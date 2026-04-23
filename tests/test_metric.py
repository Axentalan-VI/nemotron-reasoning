"""Tests for src.eval.metric. Run with: python -m pytest tests -q"""
from __future__ import annotations

import pytest

from src.eval.metric import (
    extract_answer,
    is_correct,
    score,
)


# --- extract_answer ---------------------------------------------------------

class TestExtract:
    def test_simple_boxed(self):
        assert extract_answer(r"so x=\boxed{42}") == "42"

    def test_last_boxed_wins(self):
        txt = r"first \boxed{1} scratch then final \boxed{7}"
        assert extract_answer(txt) == "7"

    def test_nested_braces(self):
        assert extract_answer(r"answer: \boxed{\frac{1}{2}}") == r"\frac{1}{2}"

    def test_boxed_with_spaces(self):
        assert extract_answer(r"x = \boxed {  3.14  }") == "3.14"

    def test_answer_tag_fallback(self):
        assert extract_answer("The final answer is 17.") == "17"

    def test_numeric_last_resort(self):
        assert extract_answer("scratch 1 and 2 and then 99") == "99"

    def test_scientific_number(self):
        assert extract_answer("result: 1.5e-3") == "1.5e-3"

    def test_strips_outer_dollars(self):
        assert extract_answer(r"\boxed{$x+1$}") == "x+1"

    def test_empty_string(self):
        assert extract_answer("") == ""

    def test_none_safe(self):
        assert extract_answer(None) == ""

    def test_string_answer(self):
        assert extract_answer(r"\boxed{hello world}") == "hello world"


# --- is_correct -------------------------------------------------------------

class TestCorrect:
    def test_exact_string(self):
        assert is_correct("foo", "foo")

    def test_case_whitespace_insensitive(self):
        assert is_correct("  Foo  Bar ", "foo bar")

    def test_numeric_tolerance_within(self):
        # |3.141 - 3.14| = 0.001 <= 1e-2 * max(1, 3.14) = 3.14e-2
        assert is_correct("3.141", "3.14")

    def test_numeric_tolerance_outside(self):
        # |3.2 - 3.14| = 0.06 > 0.0314
        assert not is_correct("3.2", "3.14")

    def test_int_vs_float(self):
        assert is_correct("42", "42.0")

    def test_comma_thousands(self):
        assert is_correct("1,000", "1000")

    def test_scientific(self):
        assert is_correct("1e3", "1000")

    def test_none_false(self):
        assert not is_correct(None, "1")
        assert not is_correct("1", None)

    def test_string_vs_number_mismatch(self):
        assert not is_correct("cat", "3")

    def test_large_number_rel_tolerance(self):
        # rel_tol scales with magnitude
        assert is_correct("1000", "1005")  # 5 <= 1e-2 * 1005 = 10.05


# --- score ------------------------------------------------------------------

class TestScore:
    def test_end_to_end(self):
        gens = [
            r"reasoning... \boxed{2}",
            r"I think \boxed{3}",
            r"finally \boxed{wrong}",
        ]
        gts = ["2", "4", "right"]
        r = score(gens, gts)
        assert r.n_total == 3
        assert r.n_correct == 1
        assert r.accuracy == pytest.approx(1 / 3)

    def test_empty(self):
        r = score([], [])
        assert r.accuracy == 0.0
        assert r.n_total == 0
