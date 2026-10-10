"""Regression test for compare_resolution() against the real local test
project. Skips if the local test dataset/reliontree checkout aren't present."""

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


def test_compare_resolution_matches_known_values():
    from relion_origin.compare import compare_resolution

    result = compare_resolution(TESTDATA, "Class3D/job001", "Refine3D/job006")
    # Ground truth verified by hand against reliontree's own job_stats() on
    # 2026-09-22: Class3D/job001 = 30.258526 A, Refine3D/job006 = 22.996479 A.
    assert result["before"] == pytest.approx(30.258526)
    assert result["after"] == pytest.approx(22.996479)
    assert result["delta_angstrom"] == pytest.approx(7.262047)
