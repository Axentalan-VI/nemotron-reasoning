"""QLoRA supervised fine-tuning for NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.

Trains a LoRA adapter (rank <= 32) on teacher-distilled reasoning traces.
The adapter is the final competition submission (packaged by
``src/submit/build_submission.py``).

Inputs
------
* Base model dir : ``/workspace/hf_cache/nemotron-3-nano-30b-a3b-bf16``
* Teacher traces : ``data/distill/teacher_traces.jsonl`` (rows with
  ``kept == True``). Each provides ``prompt`` and ``teacher_response``
  (CoT + final ``\\boxed{...}``).

Output
------
``<output-dir>/adapter/`` — a PEFT adapter directory containing
``adapter_config.json`` and ``adapter_model.safetensors``.

Design
------
* 4-bit NF4 base + bf16 compute (QLoRA). One H100 80 GB fits the 30B-A3B
  MoE model in 4-bit plus rank-32 adapters on attention projections.
* LoRA rank=32, alpha=64, dropout=0.05, target_modules=q/k/v/o_proj.
  Attention-only keeps the adapter small and under the 32-rank limit.
* Completion-only loss via TRL ``SFTTrainer`` using the ``prompt`` /
  ``completion`` schema with ``completion_only_loss=True`` — prompt tokens
  are masked from the loss, only the assistant turn contributes gradient.
* Chat template comes from the base model tokenizer (``apply_chat_template``).
  The dataset's ``prompt`` is ``[system, user]`` rendered with a trailing
  assistant generation prompt; ``completion`` is the teacher response.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


SYSTEM_PROMPT = (
    "You are a careful reasoning assistant solving few-shot rule-induction puzzles. "
    "Think step by step: identify the hidden rule from the examples, verify it on every "
    "example, then apply it to the query. End your response with the final answer on its "
    "own line in the exact format: \\boxed{ANSWER}."
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _iter_kept(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("kept") and rec.get("teacher_response"):
                yield rec


def build_dataset(path: Path, tokenizer, max_len: int) -> Dataset:
    rows: list[dict[str, Any]] = []
    dropped_len = 0
    for rec in _iter_kept(path):
        prompt_msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": rec["prompt"]},
        ]
        prompt = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        completion = rec["teacher_response"]
        n_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        n_compl = len(tokenizer(completion, add_special_tokens=False)["input_ids"])
        if n_prompt + n_compl > max_len:
            dropped_len += 1
            continue
        rows.append(
            {
                "prompt": prompt,
                "completion": completion,
                "family": rec.get("family", "unknown"),
            }
        )
    print(f"[data] loaded {len(rows)} examples  (dropped {dropped_len} over {max_len} tokens)")
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_base(model_dir: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # LoRA fine-tuning only trains adapter params, so freeze the base.
    for p in model.parameters():
        p.requires_grad = False
    # Re-enable input grads so gradient checkpointing works end-to-end.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


def attach_lora(model, rank: int, alpha: int, dropout: float):
    lora = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-to", default="none")
    ap.add_argument("--run-name", default="phase1_qlora")
    args = ap.parse_args()

    if args.rank > 32:
        raise SystemExit(f"rank {args.rank} > 32 (competition hard limit)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_out = out_dir / "adapter"

    print("[tokenizer] loading")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("[data] building dataset")
    ds = build_dataset(Path(args.traces), tokenizer, args.max_seq_len)
    if len(ds) == 0:
        raise SystemExit("no training examples after filtering")
    ds = ds.shuffle(seed=args.seed)

    print("[model] loading base in 4-bit NF4")
    model = load_base(args.base_model)
    print("[model] attaching LoRA")
    model = attach_lora(model, args.rank, args.alpha, args.dropout)

    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        run_name=args.run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        optim="adamw_torch_fused",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        seed=args.seed,
        max_length=args.max_seq_len,
        packing=False,
        completion_only_loss=True,
        report_to=args.report_to,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print("[train] starting")
    trainer.train()

    print(f"[save] writing adapter to {adapter_out}")
    trainer.model.save_pretrained(adapter_out)
    tokenizer.save_pretrained(adapter_out)
    print("[done]")


if __name__ == "__main__":
    main()
