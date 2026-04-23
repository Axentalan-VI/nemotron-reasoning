"""Synthetic generator for the `bit_manip` family.

Observed format (1602 real prompts in train.csv):

    In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.
    The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT,
    and possibly majority or choice functions.

    Here are some examples of input -> output:
    <8-bit> -> <8-bit>
    ...  (8–10 examples)

    Now, determine the output for: <8-bit>

Characteristics:
  * Input and output are always 8-bit binary strings (length == 8 exactly).
  * 8–10 examples per prompt (newlines 11–14; subtract header/trailing).
  * Alphabet: '0' and '1' only.
  * Prompt length 447–510 chars.
  * Header text is identical across all prompts.

We build a bank of deterministic 8-bit→8-bit functions covering: bitwise
NOT/AND/OR/XOR with constants, left/right shifts & rotates, byte-reversal,
nibble swap, bit-interleave, majority/parity.
"""
from __future__ import annotations

import random

from src.data.generators import SyntheticPuzzle

FAMILY = "bit_manip"

PROMPT_HEADER = (
    "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary "
    "numbers. The transformation involves operations like bit shifts, rotations, XOR, "
    "AND, OR, NOT, and possibly majority or choice functions.\n\n"
    "Here are some examples of input -> output:"
)
PROMPT_QUERY_PREFIX = "Now, determine the output for: "

BITS = 8
MASK = (1 << BITS) - 1  # 0xFF


# ---------------------------------------------------------------------------
# Rule bank — each rule is a pure int -> int mapping over an 8-bit domain.
# Rules that reference a constant are constructed by factory functions so a
# rule instance captures the constant as part of its identity.
# ---------------------------------------------------------------------------

def _bin(x: int) -> str:
    return format(x & MASK, "08b")


def _rotl(x: int, k: int) -> int:
    k %= BITS
    return ((x << k) | (x >> (BITS - k))) & MASK


def _rotr(x: int, k: int) -> int:
    k %= BITS
    return ((x >> k) | (x << (BITS - k))) & MASK


def _reverse_bits(x: int) -> int:
    return int(_bin(x)[::-1], 2)


def _swap_nibbles(x: int) -> int:
    return ((x & 0x0F) << 4) | ((x & 0xF0) >> 4)


def _popcount(x: int) -> int:
    return bin(x & MASK).count("1")


def _majority(x: int) -> int:
    """Set every bit to the majority bit of x (all 1s or all 0s)."""
    return MASK if _popcount(x) > BITS // 2 else 0


def _parity(x: int) -> int:
    """Lowest bit is parity, rest is input."""
    p = _popcount(x) & 1
    return ((x & 0xFE) | p)


def _interleave_halves(x: int) -> int:
    """Interleave high nibble (h) and low nibble (l) as h0 l0 h1 l1 ..."""
    hi = (x >> 4) & 0x0F
    lo = x & 0x0F
    out = 0
    for i in range(4):
        h_bit = (hi >> (3 - i)) & 1
        l_bit = (lo >> (3 - i)) & 1
        out = (out << 2) | (h_bit << 1) | l_bit
    return out & MASK


# Rule factory: returns (name, fn) pairs.
def _build_rules() -> dict[str, callable]:
    rules: dict[str, callable] = {}
    rules["not"] = lambda x: (~x) & MASK
    rules["reverse_bits"] = _reverse_bits
    rules["swap_nibbles"] = _swap_nibbles
    rules["majority"] = _majority
    rules["parity"] = _parity
    rules["interleave_halves"] = _interleave_halves
    # Shifts (with bits falling off the end).
    for k in (1, 2, 3):
        rules[f"shl_{k}"] = lambda x, k=k: (x << k) & MASK
        rules[f"shr_{k}"] = lambda x, k=k: (x >> k) & MASK
    # Rotations.
    for k in (1, 2, 3, 4):
        rules[f"rotl_{k}"] = lambda x, k=k: _rotl(x, k)
        rules[f"rotr_{k}"] = lambda x, k=k: _rotr(x, k)
    # XOR with a fixed 8-bit mask (a handful of recognizable constants).
    for c in (0x55, 0xAA, 0x0F, 0xF0, 0x33, 0xCC):
        rules[f"xor_{c:02x}"] = lambda x, c=c: (x ^ c) & MASK
    # AND / OR with masks.
    for c in (0x0F, 0xF0, 0x33, 0xCC):
        rules[f"and_{c:02x}"] = lambda x, c=c: (x & c) & MASK
        rules[f"or_{c:02x}"] = lambda x, c=c: (x | c) & MASK
    # Compositions (common in real puzzles).
    rules["not_then_rotl1"] = lambda x: _rotl((~x) & MASK, 1)
    rules["reverse_then_xor_55"] = lambda x: _reverse_bits(x) ^ 0x55
    rules["rotl1_xor_aa"] = lambda x: (_rotl(x, 1) ^ 0xAA) & MASK
    return rules


RULES: dict[str, callable] = _build_rules()


# ---------------------------------------------------------------------------
# Puzzle construction
# ---------------------------------------------------------------------------

def _sample_inputs(rng: random.Random, k: int, query_forbidden: int | None = None) -> list[int]:
    """Sample k distinct 8-bit values, avoiding a forbidden one."""
    pool = list(range(256))
    rng.shuffle(pool)
    out: list[int] = []
    for v in pool:
        if query_forbidden is not None and v == query_forbidden:
            continue
        out.append(v)
        if len(out) == k:
            break
    return out


def generate(rng: random.Random | None = None) -> SyntheticPuzzle:
    """Generate one synthetic bit_manip puzzle."""
    if rng is None:
        rng = random.Random()

    rule_name = rng.choice(list(RULES.keys()))
    rule_fn = RULES[rule_name]

    n_examples = rng.randint(8, 10)
    # Pick query first, then sample examples disjoint from it.
    query_in = rng.randrange(256)
    query_out = rule_fn(query_in) & MASK
    example_inputs = _sample_inputs(rng, n_examples, query_forbidden=query_in)

    lines = [f"{_bin(x)} -> {_bin(rule_fn(x) & MASK)}" for x in example_inputs]
    prompt = (
        PROMPT_HEADER
        + "\n"
        + "\n".join(lines)
        + "\n\n"
        + PROMPT_QUERY_PREFIX
        + _bin(query_in)
    )
    return SyntheticPuzzle(
        prompt=prompt,
        answer=_bin(query_out),
        family=FAMILY,
        rule=rule_name,
    )


def generate_batch(n: int, seed: int = 0) -> list[SyntheticPuzzle]:
    rng = random.Random(seed)
    return [generate(rng) for _ in range(n)]
