"""Local clone of the competition metric.

Mirrors the behavior documented on the competition Evaluation page
(https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview):

  "The metric extracts the final answer from the generated text, prioritizing
   content within the boxed format while falling back to other heuristic
   patterns or the last numeric value found. A prediction is graded as correct
   if it matches the ground truth either exactly as a string or within a
   relative numerical tolerance of 1e-2."

This module is the single source of truth for local evaluation. Do NOT
reimplement extraction / comparison elsewhere — import from here.

Reference (private) notebook: https://www.kaggle.com/code/metric/nvidia-nemotron-metric
If the official implementation is later made public and differs from this
clone, update this file first and re-run tests in tests/test_metric.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\s*{")
# Fallback: "answer is ...", "final answer: ..." style tags.
_ANSWER_TAG_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|:)\s*([^\n.]+)",
    re.IGNORECASE,
)
# Last-resort: any number (int / float / signed / scientific).
_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _extract_boxed(text: str) -> Optional[str]:
    """Return the contents of the last ``\\boxed{...}`` in ``text``.

    Handles nested braces by counting depth; LaTeX-safe.
    """
    last: Optional[str] = None
    for m in _BOXED_RE.finditer(text):
        start = m.end()  # position just past the opening '{'
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    last = text[start:i]
                    break
            i += 1
    return last


def extract_answer(text: str) -> str:
    """Extract the model's final answer from a generation.

    Priority:
      1. Last ``\\boxed{...}`` occurrence.
      2. "final answer: ..." / "answer is ..." tag.
      3. Last numeric token anywhere in the text.
      4. The raw text (stripped) as a last resort.
    """
    if text is None:
        return ""

    boxed = _extract_boxed(text)
    if boxed is not None:
        return _clean(boxed)

    tag = None
    for m in _ANSWER_TAG_RE.finditer(text):
        tag = m.group(1)
    if tag is not None:
        return _clean(tag)

    nums = _NUMERIC_RE.findall(text)
    if nums:
        return nums[-1]

    return text.strip()


def _clean(s: str) -> str:
    """Strip whitespace, trailing punctuation, and outer ``$`` math delimiters."""
    s = s.strip()
    # Strip matched surrounding dollar signs: $x$ or $$x$$.
    while len(s) >= 2 and s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    # Strip trailing sentence punctuation.
    s = s.rstrip(".;,: ")
    return s


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

# Characters stripped for numeric coercion attempts.
_NUM_STRIP = str.maketrans({",": "", "_": "", "$": "", "%": "", " ": ""})


def _to_float(s: str) -> Optional[float]:
    if s is None:
        return None
    try:
        return float(str(s).translate(_NUM_STRIP))
    except (TypeError, ValueError):
        return None


def _norm_str(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).lower()


def is_correct(prediction: str, ground_truth: str, rel_tol: float = 1e-2) -> bool:
    """Grade a single prediction against a ground-truth answer.

    Correct iff:
      * normalized string equality (whitespace/case-insensitive), OR
      * both coerce to floats and ``|p - g| <= rel_tol * max(1, |g|)``.
    """
    if prediction is None or ground_truth is None:
        return False

    if _norm_str(prediction) == _norm_str(ground_truth):
        return True

    p, g = _to_float(prediction), _to_float(ground_truth)
    if p is None or g is None:
        return False
    return abs(p - g) <= rel_tol * max(1.0, abs(g))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    accuracy: float
    n_correct: int
    n_total: int

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"accuracy={self.accuracy:.4f} ({self.n_correct}/{self.n_total})"


def score(
    generations: Iterable[str],
    ground_truths: Iterable[str],
    rel_tol: float = 1e-2,
) -> EvalResult:
    """Score a batch of generations against ground truths."""
    n = 0
    c = 0
    for gen, gt in zip(generations, ground_truths):
        pred = extract_answer(gen)
        if is_correct(pred, gt, rel_tol=rel_tol):
            c += 1
        n += 1
    acc = c / n if n else 0.0
    return EvalResult(accuracy=acc, n_correct=c, n_total=n)
