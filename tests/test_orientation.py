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


def _sphere(n, rng, z_lo=-1.0, z_hi=1.0):
    z = rng.uniform(z_lo, z_hi, n)
    return pd.DataFrame({"rlnAngleRot": rng.uniform(-180, 180, n), "rlnAngleTilt": np.degrees(np.arccos(z))})


def test_even_sphere_has_no_gaps_and_hemisphere_leaves_half_empty():
    from relion_origin.orientation import orientation_coverage
    rng = np.random.default_rng(0)
    even = orientation_coverage(_sphere(20000, rng))
    assert even["poor_fraction"] < 0.02 and even["evenness"] > 0.99
    half = orientation_coverage(_sphere(20000, rng, 0.0, 1.0))
    assert 0.45 < half["poor_fraction"] < 0.55


def test_top_and_bottom_views_only_are_caught_although_the_mean_vector_cancels():
    from relion_origin.orientation import orientation_coverage
    rng = np.random.default_rng(1)
    polar = pd.concat([_sphere(10000, rng, 0.9, 1.0), _sphere(10000, rng, -1.0, -0.9)], ignore_index=True)
    assert angular_coverage_score(polar.assign(cls=1), "cls").iloc[0] < 0.1   # R says "even" ...
    assert orientation_coverage(polar)["poor_fraction"] > 0.7                  # ... the histogram does not


def test_too_few_particles_give_no_verdict():
    from relion_origin.orientation import orientation_coverage
    assert orientation_coverage(_sphere(500, np.random.default_rng(2))) is None


def test_cyclic_symmetry_folds_equivalent_views_and_unsupported_symmetry_gives_no_verdict():
    from relion_origin.orientation import orientation_coverage, parse_cyclic_symmetry
    rng = np.random.default_rng(3)
    # C6 particle seen only within one 60-degree wedge of rot: all those views are one and the same view repeated 6x
    wedge = pd.DataFrame({"rlnAngleRot": rng.uniform(0, 60, 20000), "rlnAngleTilt": np.degrees(np.arccos(rng.uniform(-1, 1, 20000)))})
    assert orientation_coverage(wedge, sym="C1")["poor_fraction"] > 0.7     # looks like a gap-ridden sphere without symmetry
    assert orientation_coverage(wedge, sym="C6")["poor_fraction"] < 0.05    # but is complete once the symmetry is used
    assert orientation_coverage(wedge, sym="D2") is None and orientation_coverage(wedge, sym="C5") is None
    assert [parse_cyclic_symmetry(x) for x in ("C1", "c6", None, "D6", "C24")] == [1, 6, 1, None, 24]
