"""Spatial QC: where a particle sits within its tomogram/micrograph, and
how evenly particles are distributed across micrographs/tomograms.

Three independent signals, cheap given data relion-origin already loads
(io.py's coordinate/name column resolution, tomogram/micrograph
dimensions):
- particles too close to a tomogram's Z edge get a worse effective signal
  (thinner ice / sharper missing-wedge truncation at the boundary) --
  STA-specific, needs tomogram dimensions (`edge_distance_sta`).
- particles too close to a micrograph's own image edge pick up
  extraction interpolation artifacts ("fringe"/"barcode" patterns) -- a
  distinct SPA failure mode, community-confirmed,
  not the same problem as the STA one above (`edge_distance_spa`).
- micrographs/tomograms with an unusually low or high particle count are
  worth a second look (picking failure, contamination, empty grid square)
  -- works identically for SPA and STA (`particle_density_outliers`).
"""

import numpy as np
import pandas as pd

from .io import resolve_column


def edge_distance_sta(particles_df, dims_df, tilt_series_pixel_size_col="rlnTomoTiltSeriesPixelSize", particle_pixel_size_col="rlnPixelSize"):
    """Distance (Angstrom) from each particle to the nearest tomogram
    boundary along X/Y/Z, using RELION's corner-origin pixel coordinates
    and the tomogram dimensions from `io.load_tomograms_star`'s dims_df
    (same size_px * tilt_series_pixel_size physical-size logic as
    `io.center_coordinates`, but distance-to-edge instead of
    re-centering).

    Adds `_edgeDistanceX/Y/Z_angst` and `_edgeDistanceMin_angst` (the
    smallest of the three -- the binding constraint) to a copy of
    `particles_df`. Particles whose tomogram isn't in `dims_df` get NaN.
    """
    name_col = resolve_column(particles_df, "source_name")
    x_col, y_col, z_col = (resolve_column(particles_df, c) for c in ("coord_x", "coord_y", "coord_z"))
    if not all([name_col, x_col, y_col, z_col]):
        raise ValueError("edge_distance_sta: could not resolve name/coordinate columns")

    names = particles_df[name_col].astype(str)
    missing = set(names.unique()) - set(dims_df.index)
    if missing:
        raise ValueError(
            "edge_distance_sta: %d rows reference tomograms missing from dims_df (e.g. %s)"
            % (names.isin(missing).sum(), sorted(missing)[:5])
        )
    aligned = dims_df.loc[names]

    particle_px = particles_df[particle_pixel_size_col].to_numpy(dtype=float) if particle_pixel_size_col in particles_df.columns else 1.0
    tilt_px = aligned[tilt_series_pixel_size_col].to_numpy(dtype=float)

    result = particles_df.copy()
    for axis, coord_col, size_col in (("X", x_col, "rlnTomoSizeX"), ("Y", y_col, "rlnTomoSizeY"), ("Z", z_col, "rlnTomoSizeZ")):
        coord_angst = particles_df[coord_col].to_numpy(dtype=float) * particle_px
        size_angst = aligned[size_col].to_numpy(dtype=float) * tilt_px
        distance_to_near_edge = coord_angst
        distance_to_far_edge = size_angst - coord_angst
        result["_edgeDistance%s_angst" % axis] = np.minimum(distance_to_near_edge, distance_to_far_edge)

    edge_cols = ["_edgeDistanceX_angst", "_edgeDistanceY_angst", "_edgeDistanceZ_angst"]
    result["_edgeDistanceMin_angst"] = result[edge_cols].min(axis=1)
    return result


def edge_distance_spa(particles_df, micrograph_dims_df, size_x_col="SizeX", size_y_col="SizeY", pixel_size_col="rlnPixelSize"):
    """SPA counterpart to `edge_distance_sta`: distance (Angstrom) from
    each particle to the nearest *micrograph* image boundary along X/Y --
    not the tomogram-volume-edge problem above, but a distinct, separately
    documented SPA failure mode: particles extracted
    near a micrograph's image edge or beam edge pick up interpolation
    "fringe"/"barcode" artifacts during extraction.

    `micrograph_dims_df`: a DataFrame indexed by micrograph name (matching
    `rlnMicrographName`) with pixel dimensions in `size_x_col`/`size_y_col`
    -- RELION's particle STAR doesn't carry micrograph size itself, so the
    caller supplies it (e.g. read from the micrograph .mrc headers, or
    from whatever CTF-estimation output already has it). Unlike the STA
    version, no separate tilt-series pixel size is needed: a micrograph's
    own pixel size already matches its own dimensions.

    Adds `_edgeDistanceX_angst`, `_edgeDistanceY_angst`, and
    `_edgeDistanceMin_angst` (same column name as `edge_distance_sta`, so
    `flag_near_edge` works unchanged on either) to a copy of
    `particles_df`.
    """
    name_col = resolve_column(particles_df, "source_name")
    x_col, y_col = (resolve_column(particles_df, c) for c in ("coord_x", "coord_y"))
    if not all([name_col, x_col, y_col]):
        raise ValueError("edge_distance_spa: could not resolve name/coordinate columns")

    names = particles_df[name_col].astype(str)
    missing = set(names.unique()) - set(micrograph_dims_df.index)
    if missing:
        raise ValueError(
            "edge_distance_spa: %d rows reference micrographs missing from micrograph_dims_df (e.g. %s)"
            % (names.isin(missing).sum(), sorted(missing)[:5])
        )
    aligned = micrograph_dims_df.loc[names]

    pixel_size = particles_df[pixel_size_col].to_numpy(dtype=float) if pixel_size_col in particles_df.columns else 1.0

    result = particles_df.copy()
    for axis, coord_col, size_col in (("X", x_col, size_x_col), ("Y", y_col, size_y_col)):
        coord_angst = particles_df[coord_col].to_numpy(dtype=float) * pixel_size
        size_angst = aligned[size_col].to_numpy(dtype=float) * pixel_size
        distance_to_near_edge = coord_angst
        distance_to_far_edge = size_angst - coord_angst
        result["_edgeDistance%s_angst" % axis] = np.minimum(distance_to_near_edge, distance_to_far_edge)

    edge_cols = ["_edgeDistanceX_angst", "_edgeDistanceY_angst"]
    result["_edgeDistanceMin_angst"] = result[edge_cols].min(axis=1)
    return result


def flag_near_edge(particles_df, edge_distance_col="_edgeDistanceMin_angst", fraction_of_median=0.1):
    """Flag particles whose distance to the nearest tomogram edge is
    small relative to the project's own typical (median) distance --
    relative, not an absolute Angstrom cutoff, since "close to the edge"
    depends on tomogram size, which varies by dataset/binning.

    Returns a boolean Series aligned to `particles_df.index`.
    """
    distances = particles_df[edge_distance_col]
    threshold = distances.median() * fraction_of_median
    return (distances <= threshold).fillna(False)


def particle_density_outliers(particles_df, z_threshold=2.5):
    """Flag micrographs/tomograms with an unusually low or high particle
    count -- a project-wide robust z-score (MAD-based) over per-source
    particle counts. Works for SPA (rlnMicrographName) and STA
    (rlnTomoName/rlnMicrographName) identically via io's column aliasing.

    Returns a DataFrame indexed by micrograph/tomogram name with columns
    `n_particles` and `_density_outlier`.
    """
    name_col = resolve_column(particles_df, "source_name")
    if not name_col:
        raise ValueError("particle_density_outliers: could not resolve a micrograph/tomogram name column")

    counts = particles_df[name_col].value_counts().rename("n_particles").to_frame()
    median = counts["n_particles"].median()
    mad = (counts["n_particles"] - median).abs().median()
    if mad == 0 or pd.isna(mad):
        counts["_density_outlier"] = False
        return counts
    robust_z = (counts["n_particles"] - median).abs() / (1.4826 * mad)
    counts["_density_outlier"] = robust_z >= z_threshold
    return counts
