"""Angular-coverage QC: how evenly a class's particles cover orientation
space, via standard spherical (directional) statistics -- the mean
resultant length R of each particle's viewing direction as a unit vector.

R ranges 0 (orientations spread evenly over the sphere) to 1 (all
particles looking the same way). A class with high R is a missing-wedge/
preferred-orientation red flag worth checking before trusting its
resolution: STA's missing wedge already truncates Fourier coverage
per-particle, and a class that's *also* orientation-clustered compounds
that gap rather than averaging it out across particles -- the same
concern SPA calls "preferred orientation", just compounded by the wedge
in STA. This is standard Fisher spherical statistics (Fisher, Lewis &
Embleton 1987), not something specific to relion-origin -- we're just
applying it to RELION's own AngleRot/AngleTilt per particle.
"""

import numpy as np
import pandas as pd


def _viewing_directions(df, rot_col="rlnAngleRot", tilt_col="rlnAngleTilt"):
    """Each particle's projection direction as a unit vector, from
    RELION's Euler angles (rot = azimuth, tilt = polar angle from the
    Z axis, both degrees)."""
    rot = np.radians(pd.to_numeric(df[rot_col], errors="coerce"))
    tilt = np.radians(pd.to_numeric(df[tilt_col], errors="coerce"))
    x = np.sin(tilt) * np.cos(rot)
    y = np.sin(tilt) * np.sin(rot)
    z = np.cos(tilt)
    return np.column_stack([x, y, z])


def angular_coverage_score(df, class_col="rlnClassNumber"):
    """Mean resultant length R (0=isotropic, 1=fully clustered) of viewing
    directions per class. Returns a Series indexed by class.
    """
    directions = _viewing_directions(df)
    direction_df = pd.DataFrame(directions, columns=["x", "y", "z"], index=df.index)
    direction_df = direction_df.dropna()
    mean_vectors = direction_df.groupby(df.loc[direction_df.index, class_col]).mean()
    scores = np.linalg.norm(mean_vectors.to_numpy(), axis=1)
    return pd.Series(scores, index=mean_vectors.index, name="angular_coverage_R").sort_index()


def flag_clustered_orientation(df, class_col="rlnClassNumber", z_threshold=1.5):
    """Classes whose angular_coverage_score R is a project-wide outlier
    relative to the project's *own* other classes -- not an absolute R
    cutoff. Verified on real STA data: R sits at 0.69-0.94 across all four
    classes of a real membrane-associated-particle project, because
    picking geometry/specimen orientation genuinely biases coverage for
    STA far more than a fixed "R > 0.3 is bad" threshold assumes (that
    would flag every class in a normal STA project, making the flag
    useless). Comparing classes against each other is what's actually
    actionable: "worse than your own other classes" survives that
    STA-specific baseline shift; SPA classes, which usually do sit near
    isotropic, will naturally cluster near a low z-score baseline instead.

    Returns a boolean Series indexed by class. With few classes (<~4) this
    has limited statistical power -- read `angular_coverage_score`'s raw,
    ranked numbers yourself rather than trusting the flag alone.
    """
    scores = angular_coverage_score(df, class_col)
    median = scores.median()
    mad = (scores - median).abs().median()
    if mad == 0 or pd.isna(mad):
        return pd.Series(False, index=scores.index)
    robust_z = (scores - median) / (1.4826 * mad)
    return robust_z >= z_threshold
