"""Regression test for the tomogram-centering fix: without
`tomogram_dims`, RELION's corner-origin particle coordinates don't match
pytom's tomogram-centered coordinates at all. With it, they should line up
almost exactly. Skips if the local test dataset isn't present -- this
dataset is a local dev fixture, not shipped with the package.
"""

from pathlib import Path

import pytest

from relion_origin.io import load_particles, load_tomograms_star
from relion_origin.match import match_by_coordinate

from conftest import env_path

TESTDATA = env_path("RELION_ORIGIN_TEST_MATCH_DIR") or Path("/nonexistent")
TOMOGRAMS_STAR = env_path("RELION_ORIGIN_TEST_TOMOGRAMS_STAR") or Path("/nonexistent")

pytestmark = pytest.mark.skipif(not TOMOGRAMS_STAR.is_file(), reason="local test dataset not present")


def test_uncentered_match_fails():
    source = load_particles(TESTDATA / "match_source.star")
    target = load_particles(TESTDATA / "run_it025_data.star")
    result = match_by_coordinate(source, target, tolerance=5.0)
    assert (result["_match_source_index"] >= 0).sum() == 0


def test_centered_match_succeeds():
    source = load_particles(TESTDATA / "match_source.star")
    target = load_particles(TESTDATA / "run_it025_data.star")
    dims, _ctf = load_tomograms_star(TOMOGRAMS_STAR)
    result = match_by_coordinate(source, target, tolerance=10.0, tomogram_dims=dims)
    matched_fraction = (result["_match_source_index"] >= 0).mean()
    assert matched_fraction > 0.99
    assert result["_match_distance_angst"].max() < 5.0
