"""Smoke tests for orientation.py angular-coverage QC."""

import numpy as np
import pandas as pd

from relion_origin.orientation import angular_coverage_score, flag_clustered_orientation


def _isotropic(n, rng):
    tilt = np.degrees(np.arccos(rng.uniform(-1, 1, n)))
    rot = rng.uniform(-180, 180, n)
    return rot, tilt


def _clustered(n, rng):
    return rng.normal(0, 2, n), rng.normal(90, 2, n)


def test_isotropic_orientation_scores_near_zero():
    rng = np.random.default_rng(0)
    rot, tilt = _isotropic(3000, rng)
    df = pd.DataFrame({"rlnAngleRot": rot, "rlnAngleTilt": tilt, "cls": 1})
    r = angular_coverage_score(df, "cls").iloc[0]
    assert r < 0.05


def test_clustered_orientation_scores_near_one():
    rng = np.random.default_rng(0)
    rot, tilt = _clustered(3000, rng)
    df = pd.DataFrame({"rlnAngleRot": rot, "rlnAngleTilt": tilt, "cls": 1})
    r = angular_coverage_score(df, "cls").iloc[0]
    assert r > 0.95


def test_flag_relative_not_absolute():
    # three isotropic classes, one clustered -- the clustered one should
    # stand out relative to its own project even though it's not compared
    # against a fixed absolute cutoff.
    rng = np.random.default_rng(1)
    rows = []
    for cls in (1, 2, 3):
        rot, tilt = _isotropic(500, rng)
        rows.append(pd.DataFrame({"rlnAngleRot": rot, "rlnAngleTilt": tilt, "cls": cls}))
    rot, tilt = _clustered(500, rng)
    rows.append(pd.DataFrame({"rlnAngleRot": rot, "rlnAngleTilt": tilt, "cls": 4}))
    df = pd.concat(rows, ignore_index=True)

    flagged = flag_clustered_orientation(df, "cls")
    assert flagged.loc[4]
    assert not flagged.loc[1]


if __name__ == "__main__":
    test_isotropic_orientation_scores_near_zero()
    test_clustered_orientation_scores_near_one()
    test_flag_relative_not_absolute()
    print("orientation tests passed")
