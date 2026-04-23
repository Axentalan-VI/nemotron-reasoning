"""Puzzle-family taxonomy for the NVIDIA Nemotron Model Reasoning Challenge.

The public train set contains exactly six narrative families, each framed as
"In Alice's Wonderland, ...". We classify each prompt by a lead-line keyword.
The classifier is intentionally simple, deterministic, and side-effect-free so
it can be reused both during EDA and as a routing hint at train/inference time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# Canonical family ids — keep stable; downstream code/configs reference these.
FAMILIES: tuple[str, ...] = (
    "bit_manip",
    "gravity",
    "unit_conv",
    "encryption",
    "numeral",
    "equations",
    "unknown",
)

# Lead-line signatures observed in train.csv (case-insensitive substring match).
_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("bit_manip", "secret bit manipulation rule"),
    ("gravity", "gravitational constant has been secretly changed"),
    ("unit_conv", "secret unit conversion is applied"),
    ("encryption", "secret encryption rules are used on text"),
    ("numeral", "numbers are secretly converted into a different numeral system"),
    ("equations", "secret set of transformation rules is applied to equations"),
)


def classify(prompt: str) -> str:
    """Return the puzzle family id for a single prompt, or 'unknown'."""
    if not isinstance(prompt, str):
        return "unknown"
    head = prompt[:400].lower()
    for fam, sig in _SIGNATURES:
        if sig in head:
            return fam
    return "unknown"


def classify_many(prompts: Iterable[str]) -> list[str]:
    return [classify(p) for p in prompts]


@dataclass(frozen=True)
class TaxonomyReport:
    counts: dict[str, int]
    total: int
    unknown_rate: float


def summarize(prompts: Iterable[str]) -> TaxonomyReport:
    labels = classify_many(prompts)
    counts: dict[str, int] = {f: 0 for f in FAMILIES}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    total = sum(counts.values())
    unknown_rate = counts.get("unknown", 0) / total if total else 0.0
    return TaxonomyReport(counts=counts, total=total, unknown_rate=unknown_rate)
