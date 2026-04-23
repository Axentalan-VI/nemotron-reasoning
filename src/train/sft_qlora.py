"""QLoRA supervised fine-tuning for NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.

Trains a LoRA adapter (rank <= 32) on teacher-distilled reasoning traces.
The adapter is the final competition submission (packaged by
``src/submit/build_submission.py``).

Inputs
------
* Base model dir : ``/workspace/hf_cache/nemotron-3-nano-30b-a3b-bf16``
* Teacher traces : ``data/distill/teacher_traces.jsonl`` (only rows with
  ``kept == True`` are used). Each record provides ``prompt``,
  ``teacher_response`` (contains CoT + final ``\\boxed{...}``).

Output
------
``outputs/phase1_qlora/adapter/`` — a PEFT adapter directory containing
``adapter_config.json`` and ``adapter_model.safetensors``.

Design
------
* 4-bit NF4 base + bf16 compute (QLoRA). One H100 80 GB is plenty for a
  30B-A3B MoE model in 4-bit + rank-32 adapters on attention projections.
* LoRA rank=32, alpha=64, dropout=0.05, target_modules = q/k/v/o_proj.
  (Attention-only keeps the adapter well under the 32-rank limit without
  touching router / expert weights, which would bloat the adapter.)
* Completions-only loss via TRL's ``SFTTrainer`` with
  ``DataCollatorForCompletionOnlyLM`` — loss is applied to assistant tokens
  only, not to the prompt.
* Chat template comes from the base model tokenizer
  (``apply_chat_template``). We build a 2-message conversation:
  ``system`` (fixed reasoning instructions) + ``user`` (the puzzle prompt);
  the ``assistant`` completion is the teacher response verbatim.

Usage
-----
    python -m src.train.sft_qlora \\
        --base-model /workspace/hf_cache/nemotron-3-nano-30b-a3b-bf16 \\
        --traces data/distill/teacher_traces.jsonl \\
        --output-dir outputs/phase1_qlora \\
        --epochs 2 --batch-size 1 --grad-accum 16
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer


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


def load_dataset(path: Path, tokenizer, max_len: int) -> Dataset:
    """Build an SFT dataset of pre-rendered chat strings.

    The tokenizer's chat template is applied once here so we can filter out
    over-length examples before training and keep the collator simple.
    """
    rows: list[dict[str, Any]] = []
    dropped_len = 0
    for rec in _iter_kept(path):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": rec["prompt"]},
            {"role": "assistant", "content": rec["teacher_response"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # Cheap pre-filter (exact length is checked by collator / trainer).
        n_tok = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if n_tok > max_len:
            dropped_len += 1
            continue
        rows.append({"text": text, "family": rec.get("family", "unknown")})
    print(f"[data] loaded {len(rows)} examples  (dropped {dropped_len} over {max_len} tokens)")
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_base(model_dir: str):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
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
# Collator helpers
# ---------------------------------------------------------------------------

def _find_assistant_response_template(tokenizer) -> str:
    """Return the literal string that precedes the assistant's content.

    TRL's ``DataCollatorForCompletionOnlyLM`` masks loss on everything up to
    and including this marker. We derive it from the tokenizer's chat
    template so this works for Nemotron and any future base model swap.
    """
    probe = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "__PROBE__"},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    idx = probe.find("__PROBE__")
    if idx < 0:
        raise RuntimeError(
            "Could not locate assistant content in chat template output; "
            "cannot derive completion template for loss masking."
        )
    # Walk back to find the boundary between the assistant header/role
    # markers and the content. We take everything from the last "user" end
    # to the probe position.
    return probe[:idx]


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
    ap.add_argument(
        "--report-to",
        default="none",
        help="wandb | tensorboard | none (default none)",
    )
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
    ds = load_dataset(Path(args.traces), tokenizer, args.max_seq_len)
    if len(ds) == 0:
        raise SystemExit("no training examples after filtering")
    ds = ds.shuffle(seed=args.seed)

    print("[model] loading base in 4-bit NF4")
    model = load_base(args.base_model)
    print("[model] attaching LoRA")
    model = attach_lora(model, args.rank, args.alpha, args.dropout)

    response_template = _find_assistant_response_template(tokenizer)
    print(f"[collator] completion template: {response_template!r}")
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template, tokenizer=tokenizer
    )

    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        run_name=args.run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        optim="paged_adamw_8bit",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        seed=args.seed,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=False,
        report_to=args.report_to,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        data_collator=collator,
    )

    print("[train] starting")
    trainer.train()

    print(f"[save] writing adapter to {adapter_out}")
    trainer.model.save_pretrained(adapter_out)
    tokenizer.save_pretrained(adapter_out)
    print("[done]")


if __name__ == "__main__":
    main()
