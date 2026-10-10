"""Regression test for particle_flow() against the real local test project.
Skips if the local test dataset/reliontree checkout aren't present."""

from pathlib import Path

import pytest

from conftest import env_path

TESTDATA = env_path("RELION_ORIGIN_TEST_PROJECT") or Path("/nonexistent")

try:
    import reliontree  # noqa: F401

    _RELIONTREE_AVAILABLE = True
except ImportError:
    _RELIONTREE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not (TESTDATA.is_dir() and _RELIONTREE_AVAILABLE),
    reason="local test dataset or reliontree not present",
)


def test_particle_flow_matches_known_counts():
    from relion_origin.lineage import particle_flow

    df = particle_flow(TESTDATA).set_index("job")
    # Ground truth verified by hand against each job's own STAR output
    # (particles.star row count / run_data.star row count) on 2026-09-22.
    assert df.loc["Class3D/job001", "n_particles"] == 8100
    assert df.loc["Select/job002", "n_particles"] == 7091
    assert df.loc["Class3D/job003", "n_particles"] == 7091
    assert df.loc["Select/job004", "n_particles"] == 3578
    # Refine3D must report the full particle set (3578) both gold-standard
    # halves are drawn from, not one FSC half's model.star sum (1822) --
    # see lineage.py _latest_data_star docstring.
    assert df.loc["Refine3D/job006", "n_particles"] == 3578
    assert df.loc["Refine3D/job006", "loss_vs_max_parent"] == 0
