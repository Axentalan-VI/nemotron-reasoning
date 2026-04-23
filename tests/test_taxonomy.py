from __future__ import annotations

import pandas as pd
import pytest

from src.data.taxonomy import FAMILIES, classify, classify_many, summarize


def test_classify_known_families():
    samples = {
        "bit_manip": "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. ...",
        "gravity": "In Alice's Wonderland, the gravitational constant has been secretly changed. Here are some example observations:",
        "unit_conv": "In Alice's Wonderland, a secret unit conversion is applied to measurements. For example:",
        "encryption": "In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:",
        "numeral": "In Alice's Wonderland, numbers are secretly converted into a different numeral system. Some examples are given below:",
        "equations": "In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:",
    }
    for fam, prompt in samples.items():
        assert classify(prompt) == fam, f"misclassified {fam}"


def test_classify_unknown_and_bad_input():
    assert classify("random unrelated prompt") == "unknown"
    assert classify("") == "unknown"
    assert classify(None) == "unknown"  # type: ignore[arg-type]


def test_classify_many_and_summary_covers_real_train():
    path = "data/raw/train.csv"
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        pytest.skip(f"{path} not present; skipping full-dataset sanity check")
    report = summarize(df["prompt"].tolist())
    # We expect every family to be populated and unknowns to be negligible.
    assert report.total == len(df)
    assert report.unknown_rate < 0.01, report.counts
    for fam in FAMILIES:
        if fam == "unknown":
            continue
        assert report.counts[fam] > 0, f"{fam} missing from train"
