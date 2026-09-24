"""Smallest possible check that the QC pipeline still works end to end
against a synthetic STAR-like DataFrame (no external test data needed)."""

import pandas as pd

from relion_origin.qc import (
    candidate_misclassified,
    classification_purity_fraction,
    flag_outliers,
    per_class_stats,
)
from relion_origin.server import _build_page_cache, render_page


def _fake_particles(n=200):
    import numpy as np

    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "rlnClassNumber": rng.integers(1, 3, size=n),
            "rlnMaxValueProbDistribution": rng.uniform(0, 1, size=n),
            "rlnCtfMaxResolution": rng.uniform(2, 10, size=n),
        }
    )


def test_per_class_stats():
    df = _fake_particles()
    stats = per_class_stats(df, "rlnCtfMaxResolution")
    assert set(stats.index) == {1, 2}
    assert (stats["n"] > 0).all()


def test_flag_outliers_is_boolean_series():
    df = _fake_particles()
    flagged = flag_outliers(df, "rlnMaxValueProbDistribution", low_is_bad=True)
    assert flagged.dtype == bool
    assert len(flagged) == len(df)


def test_candidate_misclassified_labels():
    df = _fake_particles()
    result = candidate_misclassified(df)
    assert set(result["_qc_candidate"].unique()) <= {
        "",
        "in_good_class_low_confidence",
        "in_bad_class_high_confidence",
    }


def test_classification_purity_fraction_matches_candidate_count():
    df = _fake_particles()
    result = candidate_misclassified(df)
    expected = (result["_qc_candidate"] != "").sum() / len(result)
    assert classification_purity_fraction(df) == expected


def test_render_page_shows_classes_with_standouts():
    df = _fake_particles()
    cache = _build_page_cache(df, "rlnClassNumber")
    html = render_page("test", cache)
    assert "Classes" in html
    assert "<details" in html
    assert "standout" in html


def test_render_page_shows_diagnosis_for_flagged_classes():
    df = _fake_particles()
    cache = _build_page_cache(df, "rlnClassNumber")
    html = render_page("test", cache)
    assert "diagnosis" in html


if __name__ == "__main__":
    test_per_class_stats()
    test_flag_outliers_is_boolean_series()
    test_candidate_misclassified_labels()
    test_classification_purity_fraction_matches_candidate_count()
    test_render_page_shows_classes_with_standouts()
    test_render_page_shows_diagnosis_for_flagged_classes()
    print("all smoke tests passed")
