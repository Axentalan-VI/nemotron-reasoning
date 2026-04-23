"""Tests for the synthetic equations-family generator."""
from __future__ import annotations

import random

import pytest

from src.data.generators.equations import (
    FAMILY,
    OPERAND_ALPHABET,
    OPERATORS,
    PROMPT_HEADER,
    PROMPT_QUERY_PREFIX,
    RULES,
    generate,
    generate_batch,
)
from src.data.taxonomy import classify


def test_smoke_single_generate():
    p = generate(random.Random(42))
    assert p.family == FAMILY
    assert p.rule in RULES
    assert 1 <= len(p.answer) <= 4
    assert PROMPT_HEADER in p.prompt
    assert PROMPT_QUERY_PREFIX in p.prompt


def test_classifier_labels_as_equations():
    """Generator output must be labeled 'equations' by our taxonomy."""
    rng = random.Random(0)
    for _ in range(50):
        p = generate(rng)
        assert classify(p.prompt) == "equations", f"misclassified: {p.prompt[:80]}"


def test_examples_count_in_range():
    """Prompts must have 3–5 example equations (matches real distribution)."""
    rng = random.Random(1)
    for _ in range(100):
        p = generate(rng)
        # Count lines with ' = ' separator (examples), excluding the query line.
        eq_lines = [l for l in p.prompt.splitlines() if " = " in l]
        assert 3 <= len(eq_lines) <= 5, f"bad example count {len(eq_lines)}"


def test_all_rules_covered_deterministic_output():
    """Each rule produces a deterministic mapping (sanity)."""
    from src.data.generators.equations import _new_pair
    rng = random.Random(7)
    for name, fn in RULES.items():
        left, right, op, rhs = _new_pair(rng, fn)
        assert fn(left, right, op)[:4] in (rhs, rhs)  # trivial self-consistency
        assert 1 <= len(rhs) <= 4, f"rule {name} produced len={len(rhs)}"


def test_query_not_in_examples():
    """The query LHS must be unique (not appear as an example LHS)."""
    rng = random.Random(3)
    for _ in range(50):
        p = generate(rng)
        lines = p.prompt.splitlines()
        example_lhs = {l.split(" = ")[0] for l in lines if " = " in l}
        query_lhs = lines[-1].replace(PROMPT_QUERY_PREFIX, "").strip()
        assert query_lhs not in example_lhs


def test_operator_disjoint_from_operand_alphabet():
    """Invariant that makes operator-position unambiguous to the student."""
    for op in OPERATORS:
        assert op not in OPERAND_ALPHABET


def test_operator_is_middle_char_of_lhs():
    """Each LHS should have an operator at position 2 (0-indexed)."""
    rng = random.Random(5)
    for _ in range(30):
        p = generate(rng)
        for line in p.prompt.splitlines():
            if " = " in line:
                lhs = line.split(" = ")[0]
                assert len(lhs) == 5
                assert lhs[2] in OPERATORS
        # query too
        qline = p.prompt.splitlines()[-1]
        q_lhs = qline.replace(PROMPT_QUERY_PREFIX, "").strip()
        assert len(q_lhs) == 5
        assert q_lhs[2] in OPERATORS


def test_reproducible_seed():
    a = generate_batch(10, seed=123)
    b = generate_batch(10, seed=123)
    assert [p.prompt for p in a] == [p.prompt for p in b]
    assert [p.answer for p in a] == [p.answer for p in b]


def test_rule_coverage_over_batch():
    """A batch of 500 should hit most rule names (stochastic, but high coverage)."""
    batch = generate_batch(500, seed=0)
    seen = {p.rule for p in batch}
    # With 11 rules and uniform sampling, seeing ≥9 in 500 draws is overwhelmingly likely.
    assert len(seen) >= 9, f"only saw {len(seen)} rules: {seen}"


def test_prompt_length_matches_distribution():
    """Real prompts are 177–212 chars; synthetic should fall in a similar window."""
    batch = generate_batch(200, seed=0)
    lens = [len(p.prompt) for p in batch]
    # Allow a slightly wider band than observed (150–230).
    assert all(140 <= L <= 240 for L in lens), \
        f"length out of band: min={min(lens)}, max={max(lens)}"


def test_no_newline_in_answer():
    for p in generate_batch(100, seed=0):
        assert "\n" not in p.answer
