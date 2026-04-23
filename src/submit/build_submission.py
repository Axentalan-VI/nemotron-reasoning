"""Build a Kaggle ``submission.zip`` from a trained LoRA adapter directory.

Per the competition Code Requirements page:

  "Submit a LoRA adapter of rank at most 32 for the NVIDIA Nemotron-3-Nano-30B
   model packaged into a ``submission.zip`` file. The adapter must include an
   ``adapter_config.json``. The model is loaded with your LoRA adapter using
   the vLLM inference engine with max_lora_rank=32, max_tokens=7680, top_p=1.0,
   temperature=0.0, max_num_seqs=64, gpu_memory_utilization=0.85,
   max_model_len=8192."

This packager:
  * Validates `adapter_config.json` exists and declares ``r <= 32``.
  * Validates a weights file (``adapter_model.safetensors`` or ``.bin``) is present.
  * Zips the adapter directory into ``submission.zip`` with a flat layout
    (files at the zip root — matches the NVIDIA reference demo).

Usage (CLI):
    python -m src.submit.build_submission \\
        --adapter-dir outputs/phase1_qlora/adapter \\
        --output submission.zip
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

MAX_LORA_RANK = 32

# Files we will include in the zip. The required trio is `REQUIRED`; optional
# tokenizer/PEFT metadata files are bundled when present (vLLM may ignore them,
# but they aid reproducibility).
REQUIRED_CONFIG = "adapter_config.json"
WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")
OPTIONAL_FILES = (
    "README.md",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    # PEFT metadata
    "training_args.bin",
)


@dataclass
class ValidationReport:
    adapter_dir: Path
    config_path: Path
    weights_path: Path
    rank: int
    target_modules: list[str]
    extra_files: list[str]

    def summary(self) -> str:
        lines = [
            f"adapter_dir     : {self.adapter_dir}",
            f"adapter_config  : {self.config_path.name}",
            f"weights         : {self.weights_path.name}  ({self.weights_path.stat().st_size / 1e6:.1f} MB)",
            f"lora rank (r)   : {self.rank}  (limit {MAX_LORA_RANK})",
            f"target modules  : {', '.join(self.target_modules) or '(unspecified)'}",
            f"extra files     : {', '.join(self.extra_files) or '(none)'}",
        ]
        return "\n".join(lines)


def validate(adapter_dir: Path) -> ValidationReport:
    """Validate a PEFT/LoRA adapter directory is submittable."""
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"adapter-dir does not exist or is not a directory: {adapter_dir}")

    config_path = adapter_dir / REQUIRED_CONFIG
    if not config_path.is_file():
        raise FileNotFoundError(
            f"missing {REQUIRED_CONFIG} in {adapter_dir} — cannot submit without it"
        )

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    rank = cfg.get("r")
    if rank is None:
        raise ValueError(f"{REQUIRED_CONFIG} missing required field 'r' (lora rank)")
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError(f"{REQUIRED_CONFIG} has invalid rank r={rank!r}")
    if rank > MAX_LORA_RANK:
        raise ValueError(
            f"LoRA rank r={rank} exceeds competition limit {MAX_LORA_RANK}. "
            f"Retrain with r<={MAX_LORA_RANK}."
        )

    weights_path: Path | None = None
    for name in WEIGHT_FILES:
        p = adapter_dir / name
        if p.is_file():
            weights_path = p
            break
    if weights_path is None:
        raise FileNotFoundError(
            f"no adapter weights found in {adapter_dir} (expected one of: {WEIGHT_FILES})"
        )

    target_modules = cfg.get("target_modules") or []
    if isinstance(target_modules, str):
        target_modules = [target_modules]

    extra = [name for name in OPTIONAL_FILES if (adapter_dir / name).is_file()]

    return ValidationReport(
        adapter_dir=adapter_dir,
        config_path=config_path,
        weights_path=weights_path,
        rank=int(rank),
        target_modules=list(target_modules),
        extra_files=extra,
    )


def _files_to_pack(report: ValidationReport) -> Iterable[Path]:
    yield report.config_path
    yield report.weights_path
    for name in report.extra_files:
        yield report.adapter_dir / name


def build(adapter_dir: Path, output: Path) -> ValidationReport:
    """Validate the adapter and write a flat-layout ``submission.zip``."""
    report = validate(adapter_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Flat layout (files at zip root) matches the reference submission demo.
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src in _files_to_pack(report):
            zf.write(src, arcname=src.name)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Package a LoRA adapter as submission.zip")
    parser.add_argument("--adapter-dir", required=True, type=Path,
                        help="Directory containing adapter_config.json + adapter_model.safetensors")
    parser.add_argument("--output", default=Path("submission.zip"), type=Path,
                        help="Output zip path (default: submission.zip in cwd)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate only, do not write the zip")
    args = parser.parse_args()

    try:
        if args.dry_run:
            report = validate(args.adapter_dir)
            print("[dry-run] validation OK")
        else:
            report = build(args.adapter_dir, args.output)
            print(f"[submit] wrote {args.output} ({args.output.stat().st_size / 1e6:.2f} MB)")
        print(report.summary())
    except (FileNotFoundError, ValueError) as e:
        print(f"[submit] ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
