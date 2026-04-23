"""Tests for src.submit.build_submission packager."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from src.submit.build_submission import (
    MAX_LORA_RANK,
    build,
    validate,
)


def _make_adapter(
    tmp_path: Path,
    *,
    rank: int = 32,
    weights_name: str = "adapter_model.safetensors",
    include_config: bool = True,
    include_weights: bool = True,
    target_modules: list[str] | None = None,
    extras: list[str] | None = None,
) -> Path:
    d = tmp_path / "adapter"
    d.mkdir()
    if include_config:
        cfg = {
            "r": rank,
            "lora_alpha": rank * 2,
            "target_modules": target_modules or ["q_proj", "v_proj"],
            "peft_type": "LORA",
            "base_model_name_or_path": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        }
        (d / "adapter_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    if include_weights:
        # Fake ~1KB of weight bytes — packager doesn't inspect contents.
        (d / weights_name).write_bytes(b"\x00" * 1024)
    for name in extras or []:
        (d / name).write_text("hello", encoding="utf-8")
    return d


def test_validate_happy_path(tmp_path: Path) -> None:
    d = _make_adapter(tmp_path, rank=16)
    report = validate(d)
    assert report.rank == 16
    assert report.weights_path.name == "adapter_model.safetensors"
    assert report.target_modules == ["q_proj", "v_proj"]


def test_validate_accepts_max_rank(tmp_path: Path) -> None:
    d = _make_adapter(tmp_path, rank=MAX_LORA_RANK)
    report = validate(d)
    assert report.rank == MAX_LORA_RANK


def test_validate_rejects_rank_over_limit(tmp_path: Path) -> None:
    d = _make_adapter(tmp_path, rank=MAX_LORA_RANK + 1)
    with pytest.raises(ValueError, match="exceeds competition limit"):
        validate(d)


def test_validate_rejects_missing_config(tmp_path: Path) -> None:
    d = _make_adapter(tmp_path, include_config=False)
    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        validate(d)


def test_validate_rejects_missing_weights(tmp_path: Path) -> None:
    d = _make_adapter(tmp_path, include_weights=False)
    with pytest.raises(FileNotFoundError, match="no adapter weights"):
        validate(d)


def test_validate_rejects_missing_rank_field(tmp_path: Path) -> None:
    d = _make_adapter(tmp_path)
    cfg_path = d / "adapter_config.json"
    cfg_path.write_text(json.dumps({"lora_alpha": 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required field 'r'"):
        validate(d)


def test_validate_accepts_bin_weights(tmp_path: Path) -> None:
    d = _make_adapter(tmp_path, weights_name="adapter_model.bin", include_weights=True)
    # Remove the default safetensors file and keep only .bin
    report = validate(d)
    assert report.weights_path.name == "adapter_model.bin"


def test_validate_rejects_nonexistent_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        validate(tmp_path / "nope")


def test_build_writes_flat_zip(tmp_path: Path) -> None:
    d = _make_adapter(
        tmp_path, rank=8, extras=["tokenizer_config.json", "README.md"]
    )
    out = tmp_path / "submission.zip"
    report = build(d, out)

    assert out.is_file()
    with zipfile.ZipFile(out) as zf:
        names = sorted(zf.namelist())
    # Flat layout (no subdirectories)
    assert names == sorted(
        [
            "README.md",
            "adapter_config.json",
            "adapter_model.safetensors",
            "tokenizer_config.json",
        ]
    )
    assert "tokenizer_config.json" in report.extra_files


def test_build_ignores_unknown_files(tmp_path: Path) -> None:
    d = _make_adapter(tmp_path, rank=8)
    # Random garbage file not in OPTIONAL_FILES should NOT be packed.
    (d / "notes.txt").write_text("secret", encoding="utf-8")
    out = tmp_path / "submission.zip"
    build(d, out)
    with zipfile.ZipFile(out) as zf:
        assert "notes.txt" not in zf.namelist()
