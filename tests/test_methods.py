"""Smoke test for the methods-paragraph draft."""

import pandas as pd

from relion_origin.methods import draft_methods_paragraph
from relion_origin.qc import candidate_misclassified


def _fake_particles(n=200):
    import numpy as np

    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "rlnClassNumber": rng.integers(1, 3, size=n),
            "rlnMaxValueProbDistribution": rng.uniform(0, 1, size=n),
        }
    )


def test_paragraph_with_candidates_only():
    candidates = candidate_misclassified(_fake_particles())
    text = draft_methods_paragraph(candidates_df=candidates)
    assert "candidates for manual review" in text
    assert "not as a verdict" in text


def test_paragraph_empty_without_inputs():
    assert draft_methods_paragraph() == ""


def test_paragraph_with_resolution_comparison():
    candidates = candidate_misclassified(_fake_particles())
    comparison = {"before": 5.0, "after": 3.5, "delta_angstrom": 1.5}
    text = draft_methods_paragraph(
        candidates_df=candidates, resolution_comparison=comparison, job_before="A", job_after="B"
    )
    assert "5.0" in text and "3.5" in text
    assert "improved" in text


if __name__ == "__main__":
    test_paragraph_with_candidates_only()
    test_paragraph_empty_without_inputs()
    test_paragraph_with_resolution_comparison()
    print("methods tests passed")
