"""Tests for synthetic bit_manip generator."""
from __future__ import annotations

import random
import re

import pytest

from src.data.generators.bit_manip import (
    BITS,
    FAMILY,
    MASK,
    PROMPT_HEADER,
    PROMPT_QUERY_PREFIX,
    RULES,
    _bin,
    generate,
    generate_batch,
)
from src.data.taxonomy import classify


BIT8 = re.compile(r"^[01]{8}$")
EXAMPLE_LINE = re.compile(r"^[01]{8} -> [01]{8}$")


def test_smoke():
    p = generate(random.Random(0))
    assert p.family == FAMILY
    assert p.rule in RULES
    assert BIT8.match(p.answer), f"answer not 8-bit binary: {p.answer!r}"
    assert PROMPT_HEADER in p.prompt
    assert PROMPT_QUERY_PREFIX in p.prompt


def test_classifier_labels_correctly():
    rng = random.Random(0)
    for _ in range(50):
        p = generate(rng)
        assert classify(p.prompt) == "bit_manip", f"misclassified: {p.prompt[:100]}"


def test_examples_count_in_range():
    """8–10 examples per prompt (matches observed distribution)."""
    rng = random.Random(1)
    for _ in range(100):
        p = generate(rng)
        lines = [l for l in p.prompt.splitlines() if EXAMPLE_LINE.match(l)]
        assert 8 <= len(lines) <= 10, f"bad example count {len(lines)}"


def test_all_inputs_and_outputs_are_8_bit():
    rng = random.Random(2)
    for _ in range(50):
        p = generate(rng)
        for line in p.prompt.splitlines():
            if EXAMPLE_LINE.match(line):
                lhs, rhs = [s.strip() for s in line.split("->")]
                assert BIT8.match(lhs), f"bad LHS: {lhs!r}"
                assert BIT8.match(rhs), f"bad RHS: {rhs!r}"
        # Query is the last line after the prefix.
        q = p.prompt.splitlines()[-1].replace(PROMPT_QUERY_PREFIX, "").strip()
        assert BIT8.match(q), f"bad query: {q!r}"


def test_query_is_consistent_with_examples():
    """Applying the same hidden rule to every example must give the right RHS,
    and the query answer must match the rule applied to the query input."""
    rng = random.Random(3)
    for _ in range(30):
        p = generate(rng)
        rule_fn = RULES[p.rule]
        for line in p.prompt.splitlines():
            if EXAMPLE_LINE.match(line):
                lhs, rhs = [s.strip() for s in line.split("->")]
                x = int(lhs, 2)
                expected = _bin(rule_fn(x) & MASK)
                assert expected == rhs, (
                    f"rule {p.rule}: {lhs} -> {rhs} but rule gives {expected}"
                )
        q = p.prompt.splitlines()[-1].replace(PROMPT_QUERY_PREFIX, "").strip()
        assert _bin(rule_fn(int(q, 2)) & MASK) == p.answer


def test_query_not_in_examples():
    rng = random.Random(4)
    for _ in range(50):
        p = generate(rng)
        example_inputs = set()
        for line in p.prompt.splitlines():
            if EXAMPLE_LINE.match(line):
                example_inputs.add(line.split("->")[0].strip())
        q = p.prompt.splitlines()[-1].replace(PROMPT_QUERY_PREFIX, "").strip()
        assert q not in example_inputs, f"query {q} also appears as example input"


def test_reproducible_seed():
    a = generate_batch(20, seed=42)
    b = generate_batch(20, seed=42)
    assert [p.prompt for p in a] == [p.prompt for p in b]
    assert [p.answer for p in a] == [p.answer for p in b]


def test_rule_coverage_over_batch():
    """A batch of 1000 should cover most rules (>=20 of ~31)."""
    batch = generate_batch(1000, seed=0)
    seen = {p.rule for p in batch}
    assert len(seen) >= 20, f"only saw {len(seen)} distinct rules"


def test_prompt_length_matches_real_distribution():
    """Real prompts are 447–510 chars; synthetic should fall in similar band."""
    batch = generate_batch(200, seed=0)
    lens = [len(p.prompt) for p in batch]
    assert all(420 <= L <= 540 for L in lens), \
        f"length out of band: min={min(lens)}, max={max(lens)}"


def test_bits_constant():
    assert BITS == 8
    assert MASK == 0xFF
