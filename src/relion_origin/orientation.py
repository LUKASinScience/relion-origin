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

import re

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


# ---- preferred orientation: how much of the viewing sphere is (nearly) empty ----------------------------------
# The mean resultant length R above cancels for antipodal clusters (equal top and bottom views read as "even"),
# so it cannot detect the typical preferred orientation. This works on the angular histogram instead.

Z_BANDS, PHI_BINS = 8, 24        # 192 cells of equal area (equal steps in cos(tilt) x equal steps in rot)
POOR_FRACTION_OF_EXPECTED = 0.2  # a cell is "poorly covered" below 20% of the particles an even sphere would put in it
MIN_PARTICLES_PER_CELL = 10      # fewer than this on average and the histogram is mostly noise


def parse_cyclic_symmetry(sym):
    """n for 'C<n>' (n divides 24, so the 24 rot bins fold cleanly), None for anything else (D, T, O, I, helical)."""
    m = re.fullmatch(r"[Cc](\d+)", (sym or "C1").strip())
    return int(m.group(1)) if m and 24 % int(m.group(1)) == 0 else None


def orientation_histogram(df, rot_col="rlnAngleRot", tilt_col="rlnAngleTilt", z_bands=Z_BANDS, phi_bins=PHI_BINS, fold=1):
    """Particle counts on an equal-area grid of viewing directions: array [z_bands, phi_bins // fold].
    `fold` = n for a Cn particle: viewing directions that differ by a rotation of 360/n degrees about the symmetry
    axis (rot -> rot + 360/n) show the same view of the particle, so they share a cell."""
    rot = pd.to_numeric(df[rot_col], errors="coerce").to_numpy(dtype=float)
    tilt = pd.to_numeric(df[tilt_col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(rot) & np.isfinite(tilt)
    z = np.cos(np.radians(tilt[ok]))
    zi = np.clip(((z + 1) / 2 * z_bands).astype(int), 0, z_bands - 1)
    period = 360.0 / fold
    cols = phi_bins // fold
    pi = np.clip(((rot[ok] % period) / period * cols).astype(int), 0, cols - 1)
    counts = np.zeros((z_bands, cols), dtype=int)
    np.add.at(counts, (zi, pi), 1)
    return counts


def orientation_coverage(df, rot_col="rlnAngleRot", tilt_col="rlnAngleTilt", sym="C1"):
    """{"n", "poor_fraction", "evenness"} for one set of particles, or None when there are too few particles per cell
    or the symmetry is not handled (only C1 and Cn).

    `poor_fraction`: share of the viewing sphere (by area, after folding the symmetry) holding fewer than 20% of the
    particles an even distribution would put there. 0 = no gaps; 0.5 = half of all viewing directions are nearly empty
    (all views on one hemisphere, or only top and bottom views, ...). `evenness`: normalised entropy, 1 = perfectly even.
    Whether a high poor_fraction is a problem depends on the experiment: single-particle work wants it near 0, while
    membrane-bound subtomogram averaging is restricted by the picking geometry by design."""
    fold = parse_cyclic_symmetry(sym)
    if fold is None:
        return None
    counts = orientation_histogram(df, rot_col, tilt_col, fold=fold)
    n = int(counts.sum())
    cells = counts.size
    if n < MIN_PARTICLES_PER_CELL * cells:
        return None
    expected = n / cells
    p = counts[counts > 0] / n
    return {
        "n": n,
        "poor_fraction": float((counts < POOR_FRACTION_OF_EXPECTED * expected).mean()),
        "evenness": float(-(p * np.log(p)).sum() / np.log(cells)),
    }
