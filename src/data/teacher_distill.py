"""Teacher chain-of-thought distillation client.

Calls an OpenAI-compatible endpoint (OpenRouter by default) to generate a
reasoning trace + final `\\boxed{...}` answer for each training prompt, then
keeps only the samples where the teacher's boxed answer matches the gold
answer under the competition metric.

Design goals:
  * Resumable — writes one JSONL line per completed item so re-runs skip work.
  * Parallel — `asyncio` + semaphore; OpenRouter's free tier allows bursts.
  * Safe — never fabricates an answer. If the teacher errors or mismatches,
    the record is written with `kept=False` and carries the error/mismatch.

Not intended to run on Windows laptop for all 9.5k rows; use the GPU box or a
small head-node. A tiny smoke-test (--limit 4) is cheap and validates the
pipeline end-to-end.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

# Project root importable so we can reuse metric + taxonomy.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.taxonomy import classify  # noqa: E402
from src.eval.metric import extract_answer, is_correct  # noqa: E402


SYSTEM_PROMPT = (
    "You are a careful reasoning assistant solving few-shot rule-induction puzzles. "
    "Think step by step: identify the hidden rule from the examples, verify it on every "
    "example, then apply it to the query. Keep reasoning concise and grounded in the examples. "
    "End your response with the final answer on its own line in the exact format: \\boxed{ANSWER}.\n\n"
    "Rules for the contents of \\boxed{...}:\n"
    "  * Include ONLY the final answer value — no units, no words like 'meters', no LaTeX "
    "commands such as \\text{...}, no surrounding quotes, no trailing punctuation.\n"
    "  * For numeric answers, write the bare number (e.g. \\boxed{34.94}, not \\boxed{34.94 m} "
    "or \\boxed{34.94\\text{ m}}).\n"
    "  * For string answers, write the exact string with no quotes (e.g. \\boxed{10101101}, "
    "\\boxed{VI}, \\boxed{the wise knight found}).\n"
    "  * Match the format of the gold answers implied by the examples in the prompt."
)


@dataclass
class TeacherRecord:
    id: str
    family: str
    prompt: str
    gold: str
    teacher_response: str
    teacher_answer: str | None
    kept: bool
    error: str | None = None
    latency_s: float | None = None


def _load_done_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done: set[str] = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                continue
    return done


async def _call_teacher(client, model: str, prompt: str, max_tokens: int) -> str:
    # `reasoning.effort=low` caps hidden reasoning tokens on OpenRouter-supported
    # reasoning models (gpt-oss, o-series, DeepSeek-R1, etc.). Without this the
    # paid gpt-oss-120b burns the entire max_tokens budget on hidden reasoning
    # and returns an empty visible completion. "low" matches the behavior we saw
    # on the free variant (~55 reasoning tokens, 11s latency, correct answer).
    # Non-reasoning models ignore the field.
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
        extra_body={"reasoning": {"effort": "low"}},
    )
    return resp.choices[0].message.content or ""


def _retry_after_seconds(exc: Exception) -> float | None:
    """Best-effort extraction of a Retry-After hint from an OpenAI/OpenRouter exception."""
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            hdr = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
            if hdr:
                return float(hdr)
        except Exception:  # noqa: BLE001
            pass
    return None


async def _worker(
    sem: asyncio.Semaphore,
    client,
    model: str,
    max_tokens: int,
    row: dict[str, Any],
    out_f,
    lock: asyncio.Lock,
    retries: int,
) -> None:
    async with sem:
        start = time.time()
        err: str | None = None
        text = ""
        for attempt in range(retries + 1):
            try:
                text = await _call_teacher(client, model, row["prompt"], max_tokens)
                err = None
                break
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
                if attempt >= retries:
                    break
                # Detect rate-limit / transient errors and back off more aggressively.
                status = getattr(e, "status_code", None)
                name = type(e).__name__
                is_rate_limit = status == 429 or "RateLimit" in name
                is_transient = is_rate_limit or status in (500, 502, 503, 504) or "APIConnection" in name or "Timeout" in name
                if is_rate_limit:
                    # Respect server hint if present, else exponential backoff starting at 10s.
                    delay = _retry_after_seconds(e) or (10.0 * (2 ** attempt))
                elif is_transient:
                    delay = 2.0 * (2 ** attempt)
                else:
                    # Non-transient (auth, bad request, etc.) — fail fast.
                    break
                await asyncio.sleep(delay)

        latency = time.time() - start
        teacher_answer = extract_answer(text) if text else None
        kept = err is None and teacher_answer is not None and is_correct(teacher_answer, row["answer"])

        rec = TeacherRecord(
            id=str(row["id"]),
            family=classify(row["prompt"]),
            prompt=row["prompt"],
            gold=str(row["answer"]),
            teacher_response=text,
            teacher_answer=teacher_answer,
            kept=kept,
            error=err,
            latency_s=round(latency, 2),
        )
        async with lock:
            out_f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            out_f.flush()


async def run(
    df: pd.DataFrame,
    out_path: Path,
    model: str,
    base_url: str,
    api_key: str,
    concurrency: int,
    max_tokens: int,
    retries: int,
) -> None:
    # Import here so the module stays importable without `openai` installed.
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=180.0, max_retries=0)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as out_f:
        tasks = [
            _worker(sem, client, model, max_tokens, row, out_f, lock, retries)
            for row in df.to_dict("records")
        ]
        # Simple progress counter without extra deps.
        done = 0
        for fut in asyncio.as_completed(tasks):
            await fut
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"[distill] {done}/{len(tasks)} completed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/train.csv")
    parser.add_argument("--output", default="data/distill/teacher_traces.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="0 = all rows")
    parser.add_argument("--stratify", action="store_true", help="even sample across families when --limit > 0")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    base_url = os.environ.get("TEACHER_BASE_URL") or "https://openrouter.ai/api/v1"
    api_key = os.environ.get("TEACHER_API_KEY")
    model = os.environ.get("TEACHER_MODEL") or "nvidia/nemotron-3-super-120b-a12b:free"
    if not api_key:
        raise SystemExit("TEACHER_API_KEY not set (check .env)")

    df = pd.read_csv(ROOT / args.input)
    df["family"] = df["prompt"].map(classify)

    done_ids = _load_done_ids(ROOT / args.output)
    if done_ids:
        df = df[~df["id"].astype(str).isin(done_ids)]
        print(f"[distill] skipping {len(done_ids)} already-completed rows")

    if args.limit > 0:
        if args.stratify:
            per_fam = max(1, args.limit // df["family"].nunique())
            df = (
                df.groupby("family", group_keys=False)
                .apply(lambda g: g.sample(min(per_fam, len(g)), random_state=args.seed))
            )
        else:
            df = df.sample(min(args.limit, len(df)), random_state=args.seed)

    print(f"[distill] model={model} rows={len(df)} concurrency={args.concurrency}")
    if len(df) == 0:
        print("[distill] nothing to do")
        return

    asyncio.run(
        run(
            df=df,
            out_path=ROOT / args.output,
            model=model,
            base_url=base_url,
            api_key=api_key,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            retries=args.retries,
        )
    )


if __name__ == "__main__":
    main()
