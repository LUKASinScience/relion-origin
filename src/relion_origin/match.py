"""Match particles across STAR files that share no stable ID.

Generalizes the coordinate-matching technique the test dataset's own
transfer_lcc_values.py already did by hand: nearest-neighbour join in 3D,
grouped by tomogram/micrograph name, within a distance tolerance.
"""

import numpy as np
import pandas as pd

from .io import center_coordinates, resolve_column, tomo_base

DEFAULT_TOLERANCE_ANGSTROM = 5.0


def match_by_coordinate(
    source_df,
    target_df,
    tolerance=DEFAULT_TOLERANCE_ANGSTROM,
    target_pixel_size_col="rlnPixelSize",
    default_pixel_size=None,
    tomogram_dims=None,
):
    """Nearest-neighbour-match each row of `target_df` to a row of
    `source_df`, grouped by tomogram/micrograph name.

    Source coordinates are assumed to already be in Angstrom (STA raw
    picker output, e.g. pytom's rlnCenteredCoordinateX/Y/ZAngst, corner
    coordinates centered on the tomogram). Target coordinates are RELION's
    corner-origin pixel coordinates (rlnCoordinateX/Y/Z); converted to
    Angstrom using `target_pixel_size_col` per-row if present, else
    `default_pixel_size`.

    `tomogram_dims`: for STA data, pass the `dims_df` from
    `io.load_tomograms_star` to also re-center the target coordinates on
    the tomogram (RELION's pixel coordinates are corner-origin, pytom's
    are tomogram-centered -- without this correction nothing matches even
    though both files describe the same particles).
    Leave it None for SPA data, where both sides already share the same
    (uncentered) coordinate convention.

    Returns a copy of `target_df` with two extra columns:
      - `_match_source_index`: positional index into `source_df`, or -1
      - `_match_distance_angst`: nearest-neighbour distance, NaN if no
        tomogram overlap or nothing within `tolerance`
    """
    source_name_col = resolve_column(source_df, "source_name")
    target_name_col = resolve_column(target_df, "source_name")
    sx, sy, sz = (resolve_column(source_df, c) for c in ("coord_x", "coord_y", "coord_z"))
    tx, ty, tz = (resolve_column(target_df, c) for c in ("coord_x", "coord_y", "coord_z"))
    missing = [
        name
        for name, col in [
            ("source name column", source_name_col),
            ("target name column", target_name_col),
            ("source coordinates", sx and sy and sz),
            ("target coordinates", tx and ty and tz),
        ]
        if not col
    ]
    if missing:
        raise ValueError("match_by_coordinate: could not resolve " + ", ".join(missing))

    if target_pixel_size_col in target_df.columns:
        pixel_size = target_df[target_pixel_size_col].astype(float).to_numpy()
    elif default_pixel_size is not None:
        pixel_size = np.full(len(target_df), default_pixel_size, dtype=float)
    else:
        pixel_size = np.ones(len(target_df), dtype=float)

    if tomogram_dims is not None:
        target_names = target_df[target_name_col].astype(str)
        missing_tomos = set(target_names.unique()) - set(tomogram_dims.index)
        if missing_tomos:
            raise ValueError(
                "match_by_coordinate: %d target rows reference tomograms missing from "
                "tomogram_dims (e.g. %s) -- pass a dims_df covering every tomogram, or "
                "drop these rows first"
                % (target_names.isin(missing_tomos).sum(), sorted(missing_tomos)[:5])
            )
        dims_aligned = tomogram_dims.loc[target_names]
        cx, cy, cz = center_coordinates(
            target_df[tx].to_numpy(dtype=float),
            target_df[ty].to_numpy(dtype=float),
            target_df[tz].to_numpy(dtype=float),
            dims_aligned["rlnTomoSizeX"].to_numpy(dtype=float),
            dims_aligned["rlnTomoSizeY"].to_numpy(dtype=float),
            dims_aligned["rlnTomoSizeZ"].to_numpy(dtype=float),
            dims_aligned["rlnTomoTiltSeriesPixelSize"].to_numpy(dtype=float),
            pixel_size,
        )
        target_coords_angst = np.column_stack([cx, cy, cz])
    else:
        target_coords_angst = np.column_stack(
            [
                target_df[tx].to_numpy(dtype=float) * pixel_size,
                target_df[ty].to_numpy(dtype=float) * pixel_size,
                target_df[tz].to_numpy(dtype=float) * pixel_size,
            ]
        )
    target_keys = target_df[target_name_col].astype(str).map(tomo_base)

    source_coords = source_df[[sx, sy, sz]].to_numpy(dtype=float)
    source_keys = source_df[source_name_col].astype(str).map(tomo_base)
    source_groups = {key: np.flatnonzero(source_keys.to_numpy() == key) for key in source_keys.unique()}

    matched_index = np.full(len(target_df), -1, dtype=np.int64)
    match_distance = np.full(len(target_df), np.nan, dtype=float)

    # brute-force (n, m, 3) distance matrix per tomogram group -- fine at
    # typical group sizes, but an (m, 3) candidate block that size would
    # allocate multiple GB, so queries are chunked to cap memory rather
    # than assuming groups stay small.
    _MAX_PAIRS_PER_CHUNK = 20_000_000  # ~480 MB of float64 deltas at 3 dims

    for key, target_positions in target_keys.groupby(target_keys).groups.items():
        candidate_rows = source_groups.get(key)
        if candidate_rows is None or len(candidate_rows) == 0:
            continue
        target_positions = target_df.index.get_indexer(target_positions)
        candidates = source_coords[candidate_rows]  # (m, 3)
        queries = target_coords_angst[target_positions]  # (n, 3)

        chunk_size = max(1, _MAX_PAIRS_PER_CHUNK // max(1, len(candidates)))
        for start in range(0, len(queries), chunk_size):
            chunk = queries[start : start + chunk_size]
            deltas = chunk[:, None, :] - candidates[None, :, :]
            dist = np.sqrt((deltas**2).sum(axis=-1))
            nearest = dist.argmin(axis=1)
            nearest_dist = dist[np.arange(len(chunk)), nearest]
            chunk_positions = target_positions[start : start + chunk_size]
            matched_index[chunk_positions] = candidate_rows[nearest]
            match_distance[chunk_positions] = nearest_dist

    result = target_df.copy()
    within_tolerance = match_distance <= tolerance
    result["_match_source_index"] = np.where(within_tolerance, matched_index, -1)
    result["_match_distance_angst"] = np.where(within_tolerance, match_distance, np.nan)
    return result
