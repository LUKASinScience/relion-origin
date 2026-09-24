"""Smoke test for the QC-candidate -> RELION-loadable STAR rerun helper."""

import tempfile
from pathlib import Path

import pandas as pd

from relion_origin.io import load_particles
from relion_origin.qc import candidate_misclassified
from relion_origin.rerun import write_candidate_subset


def _fake_particles(n=200):
    import numpy as np

    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "rlnClassNumber": rng.integers(1, 3, size=n),
            "rlnMaxValueProbDistribution": rng.uniform(0, 1, size=n),
            "rlnImageName": [f"{i+1:06d}@particles.mrcs" for i in range(n)],
        }
    )


def test_write_candidate_subset_roundtrips_and_drops_internal_columns():
    df = candidate_misclassified(_fake_particles())
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "candidates.star"
        n = write_candidate_subset(df, out)
        assert n > 0

        reloaded = load_particles(out)
        assert len(reloaded) == n
        assert "_qc_candidate" not in reloaded.columns
        assert "rlnImageName" in reloaded.columns


def test_write_candidate_subset_specific_label():
    df = candidate_misclassified(_fake_particles())
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "candidates.star"
        n = write_candidate_subset(df, out, candidate_values=["in_bad_class_high_confidence"])
        reloaded = load_particles(out)
        assert len(reloaded) == n


def test_write_candidate_subset_raises_on_empty_selection():
    df = candidate_misclassified(_fake_particles())
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "candidates.star"
        try:
            write_candidate_subset(df, out, candidate_values=["not_a_real_label"])
        except ValueError:
            return
        raise AssertionError("expected ValueError for an empty candidate selection")


if __name__ == "__main__":
    test_write_candidate_subset_roundtrips_and_drops_internal_columns()
    test_write_candidate_subset_specific_label()
    test_write_candidate_subset_raises_on_empty_selection()
    print("rerun tests passed")
