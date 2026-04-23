"""Synthetic puzzle generators (one module per family).

Each generator exposes a `generate(rng)` function returning a `SyntheticPuzzle`
dataclass with `{prompt, answer, family, rule}`. The prompt format mirrors the
competition surface distribution closely enough to be useful training signal.

These are used for Phase 2 synthetic data scale-up, especially for families
where the teacher's kept-rate is low (equations, bit_manip).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SyntheticPuzzle:
    prompt: str
    answer: str
    family: str
    rule: str  # tag for analysis, not included in prompt
