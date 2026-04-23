"""Synthetic generator for the `equations` family ("Alice's Wonderland").

The real puzzles show the student 3–5 few-shot examples of a secret
transformation rule over short punctuation+digit strings, then ask them to
apply it to a new query.

Observed surface format (from `data/raw/train.csv`, family=equations, n=1555):

    In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
    <LHS_1> = <RHS_1>
    <LHS_2> = <RHS_2>
    ...
    Now, determine the result for: <QUERY_LHS>

Characteristics:
  * 3–5 examples per prompt (mean ~4.0).
  * LHS is typically 5 characters: two operand chars, one operator at position
    2 (0-indexed), two more operand chars — e.g. ``%|*"|`` means operands
    ``%|`` and ``"|`` with operator ``*``. Some puzzles use shorter tokens.
  * Operators observed: ``*``, ``-``, ``+``, ``/``, ``\\``, ``|``.
  * Alphabet: digits 0–9 plus ASCII punctuation (no spaces, no letters).
  * RHS / answer length: 1–4 chars (mean ~2.94, bounded at 4).
  * Prompt length: 177–212 chars.

Each puzzle uses ONE secret rule; the student must induce it. Since the
competition rules are not public, we build a bank of plausible rules and
sample one per generated puzzle. This teaches the general *induce-then-apply*
meta-skill, which transfers across specific rule families.
"""
from __future__ import annotations

import random
from typing import Callable

from src.data.generators import SyntheticPuzzle

FAMILY = "equations"

# Operand alphabet — digits + punctuation, excluding space and letters.
# Operators are a *disjoint* subset so the operator position is unambiguous.
OPERATORS = ("*", "-", "+", "/", "\\", "|")
_NON_OPERATOR_PUNCT = "!\"#$%&'()[]{}<>:;,.?@^_`~"
DIGITS = "0123456789"
OPERAND_ALPHABET = DIGITS + _NON_OPERATOR_PUNCT
assert not any(op in OPERAND_ALPHABET for op in OPERATORS), \
    "operator chars must not appear in operand alphabet (position disambiguation)"

PROMPT_HEADER = (
    "In Alice's Wonderland, a secret set of transformation rules is applied "
    "to equations. Below are a few examples:"
)
PROMPT_QUERY_PREFIX = "Now, determine the result for: "


# ---------------------------------------------------------------------------
# Rule bank — each rule maps (left, right, op) -> output string.
# All rules are deterministic and produce 1–4 char outputs on 2-char operands.
# ---------------------------------------------------------------------------

Rule = Callable[[str, str, str], str]


def _rule_concat(left: str, right: str, op: str) -> str:
    """Concatenate operands (operator ignored)."""
    return (left + right)[:4]


def _rule_reverse_concat(left: str, right: str, op: str) -> str:
    """Right then left."""
    return (right + left)[:4]


def _rule_left_only(left: str, right: str, op: str) -> str:
    return left


def _rule_right_only(left: str, right: str, op: str) -> str:
    return right


def _rule_interleave(left: str, right: str, op: str) -> str:
    """Zip chars alternately: l0 r0 l1 r1."""
    out = []
    for a, b in zip(left, right):
        out.extend([a, b])
    return "".join(out)[:4]


def _rule_diff(left: str, right: str, op: str) -> str:
    """Chars in left that are NOT in right (preserve left order)."""
    result = "".join(c for c in left if c not in right)
    # Guarantee non-empty answer: fall back to left[0].
    return result if result else left[0]


def _rule_intersect(left: str, right: str, op: str) -> str:
    """Chars present in both (preserve left order)."""
    result = "".join(c for c in left if c in right)
    return result if result else left[-1]


def _rule_reverse_left(left: str, right: str, op: str) -> str:
    return left[::-1]


def _rule_reverse_right(left: str, right: str, op: str) -> str:
    return right[::-1]


def _rule_op_plus_last(left: str, right: str, op: str) -> str:
    """Operator char followed by last operand char."""
    return op + right[-1]


def _rule_first_last(left: str, right: str, op: str) -> str:
    """First char of left + last char of right."""
    return left[0] + right[-1]


RULES: dict[str, Rule] = {
    "concat": _rule_concat,
    "reverse_concat": _rule_reverse_concat,
    "left_only": _rule_left_only,
    "right_only": _rule_right_only,
    "interleave": _rule_interleave,
    "diff": _rule_diff,
    "intersect": _rule_intersect,
    "reverse_left": _rule_reverse_left,
    "reverse_right": _rule_reverse_right,
    "op_plus_last": _rule_op_plus_last,
    "first_last": _rule_first_last,
}


# ---------------------------------------------------------------------------
# Puzzle construction
# ---------------------------------------------------------------------------

def _sample_operand(rng: random.Random, length: int = 2) -> str:
    return "".join(rng.choices(OPERAND_ALPHABET, k=length))


def _render_equation(left: str, right: str, op: str, rhs: str) -> str:
    return f"{left}{op}{right} = {rhs}"


def _new_pair(rng: random.Random, rule: Rule, operand_len: int = 2) -> tuple[str, str, str, str]:
    """Sample (left, right, op, rhs) ensuring rhs is 1–4 chars."""
    for _ in range(20):
        left = _sample_operand(rng, operand_len)
        right = _sample_operand(rng, operand_len)
        op = rng.choice(OPERATORS)
        rhs = rule(left, right, op)
        if rhs and 1 <= len(rhs) <= 4:
            return left, right, op, rhs
    # Fallback (should be rare): truncate.
    left = _sample_operand(rng, operand_len)
    right = _sample_operand(rng, operand_len)
    op = rng.choice(OPERATORS)
    rhs = (rule(left, right, op) or left[0])[:4]
    return left, right, op, rhs


def generate(rng: random.Random | None = None) -> SyntheticPuzzle:
    """Generate one synthetic equations-family puzzle with a ground-truth answer."""
    if rng is None:
        rng = random.Random()

    rule_name = rng.choice(list(RULES.keys()))
    rule_fn = RULES[rule_name]

    n_examples = rng.randint(3, 5)
    # Use a set of unique LHS strings to avoid duplicate examples.
    seen: set[str] = set()
    example_lines: list[str] = []
    attempts = 0
    while len(example_lines) < n_examples and attempts < 100:
        attempts += 1
        left, right, op, rhs = _new_pair(rng, rule_fn)
        key = left + op + right
        if key in seen:
            continue
        seen.add(key)
        example_lines.append(_render_equation(left, right, op, rhs))

    # Query (distinct from examples).
    for _ in range(50):
        q_left, q_right, q_op, q_ans = _new_pair(rng, rule_fn)
        q_key = q_left + q_op + q_right
        if q_key not in seen:
            break

    query_lhs = q_left + q_op + q_right
    prompt = (
        PROMPT_HEADER
        + "\n"
        + "\n".join(example_lines)
        + "\n"
        + PROMPT_QUERY_PREFIX
        + query_lhs
    )
    return SyntheticPuzzle(prompt=prompt, answer=q_ans, family=FAMILY, rule=rule_name)


def generate_batch(n: int, seed: int = 0) -> list[SyntheticPuzzle]:
    rng = random.Random(seed)
    return [generate(rng) for _ in range(n)]
