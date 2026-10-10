"""Regression test for iteration_convergence() against the real local test
project. Skips if the local test dataset isn't present."""

from pathlib import Path

import pytest

from conftest import env_path

TESTDATA = (env_path("RELION_ORIGIN_TEST_PROJECT") or Path("/nonexistent")) / "Class3D" / "job001"

pytestmark = pytest.mark.skipif(not TESTDATA.is_dir(), reason="local test dataset not present")


def test_iteration_convergence_matches_known_values():
    from relion_origin.iteration_convergence import iteration_convergence, stable_since_iteration

    df = iteration_convergence(TESTDATA).set_index("iteration")
    # Ground truth verified by hand against run_it*_data.star for this real
    # K=4 Class3D job on 2026-09-24: iteration 1 is still the single-class
    # VDAM coarse phase (0 switches, only 1 class); the split into 4 classes
    # at iteration 2 causes a large reshuffle visible at iteration 3.
    assert df.loc[1, "n_classes"] == 1
    assert df.loc[1, "switch_rate"] == 0.0
    assert df.loc[3, "n_classes"] == 4
    assert df.loc[3, "switch_rate"] > 0.5
    assert (df["n_matched"] == 8100).all()
    # This run never sustains <=5% switch rate for 3 consecutive
    # transitions -- confirms the strong missing-wedge/preferred-orientation
    # heterogeneity of this dataset.
    assert stable_since_iteration(df) is None


if __name__ == "__main__":
    test_iteration_convergence_matches_known_values()
    print("iteration_convergence regression test passed")
