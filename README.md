# NVIDIA Nemotron Model Reasoning

Kaggle competition: [NVIDIA Open Models — Reasoning](https://www.kaggle.com/competitions/nvidia-open-models)

Fine-tune **NVIDIA Nemotron-3-Nano-30B-A3B-BF16** with a small (≤ rank 32) LoRA
adapter on teacher-distilled reasoning traces. The adapter — not a full
model — is the final submission, loaded by Kaggle into vLLM with
`max_lora_rank=32`.

## Approach

1. **Teacher distillation** — sample CoT solutions from a strong open
   reasoning teacher; keep only traces whose final `\boxed{...}` matches the
   ground-truth answer (`src/eval/metric.py`).
2. **QLoRA SFT** — 4-bit NF4 base + bf16 compute, LoRA rank 32, alpha 64,
   target modules `q/k/v/o_proj` (attention only). Completion-only loss via
   TRL `SFTTrainer` using the model's chat template.
3. **Submission packaging** — zip the adapter dir flat into `submission.zip`
   (validates `adapter_config.json` declares `r <= 32`).

## Layout

```
scripts/
  download_data.py          download competition data
src/
  taxonomy.py               problem-type taxonomy
  teacher_distill.py        teacher trace generation + filtering
  generators/
    bit_manip.py            synthetic bit-manipulation problems
    equations.py            synthetic equation problems
  eval/metric.py            \boxed{...} extraction + exact-match
  train/sft_qlora.py        QLoRA SFT trainer (single H100 80 GB)
  submit/build_submission.py  package adapter -> submission.zip
tests/                      pytest unit tests for generators / metric / submit
requirements.txt            CPU-only deps (data prep, tests)
requirements-gpu.txt        training deps (bitsandbytes, peft, trl, vllm)
```

## Quick commands

```powershell
# CPU env (data prep, tests)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q

# Train on a single H100 80 GB
pip install -r requirements-gpu.txt
python -m src.train.sft_qlora `
  --base-model /workspace/hf_cache/nemotron-3-nano-30b-a3b-bf16 `
  --traces data/distill/teacher_traces.jsonl `
  --output-dir outputs/phase1_qlora

# Package submission
python -m src.submit.build_submission `
  --adapter-dir outputs/phase1_qlora/adapter `
  --output submission.zip
```

## Inference constraints (Kaggle host)

The competition harness loads the adapter into vLLM with these fixed flags:

```
max_lora_rank=32  max_tokens=7680  top_p=1.0  temperature=0.0
max_num_seqs=64   gpu_memory_utilization=0.85  max_model_len=8192
```

So the adapter must keep `r <= 32` and the response must end in
`\boxed{<answer>}` for the metric to score it.

## Notes

- Nemotron-H uses Mamba blocks; bnb 4-bit + Mamba is incompatible, so for
  any attention-only or end-to-end fine-tunes that touch SSM layers, fall
  back to **bf16 + LoRA** (no 4-bit quantization).
- HF Transformers requires `attn_implementation="eager"` for NemotronH.
- `data/`, `outputs/`, and adapter weights are gitignored — too large for
  GitHub.
