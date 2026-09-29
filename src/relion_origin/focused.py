"""Finding a small, localized class (e.g. a bound ligand, a shifted
subunit) that ordinary alignment-driven classification smears away.

Three independent signals, each a *candidate*, never a verdict (same
principle as the rest of relion-origin):

- `offset_summary`: whole-particle shifts get absorbed by alignment -- the
  difference then lives only in `rlnOriginX/YAngst`, not in the maps.
- `rank_class_differences`: which class differs from the others in a
  compact blob (ligand-like) vs. everywhere (noise / global change).
- `class_occupancy_trajectory`: classes that emptied out during a run
  (RELION 4 no-alignment runs can collapse classes, 3dem/relion#821).

Pure numpy/pandas; the MRC reader handles the modes RELION writes.
"""

import re
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import starfile

_MRC_DTYPES = {0: np.int8, 1: np.int16, 2: np.float32, 6: np.uint16, 12: np.float16}


def read_mrc(path):
    """3D array (z, y, x) of an MRC volume. Little-endian files only."""
    raw = Path(path).read_bytes()
    nx, ny, nz, mode = struct.unpack_from("<4i", raw, 0)
    (nsymbt,) = struct.unpack_from("<i", raw, 92)
    if mode not in _MRC_DTYPES:
        raise ValueError("unsupported MRC mode %d in %s" % (mode, path))
    start = 1024 + nsymbt
    data = np.frombuffer(raw, dtype=_MRC_DTYPES[mode], count=nx * ny * nz, offset=start)
    return data.reshape(nz, ny, nx).astype(np.float32)


def offset_summary(df, class_col="rlnClassNumber", pixel_size_angstrom=None, offset_range_px=None,
                   z_threshold=5.0, min_shift_angstrom=0.2):
    """Per class: n, mean/std offset (A) and how far the class's mean offset
    is from all other particles' (`shift_angstrom`, `z` = shift / standard
    error). `flag_shifted` needs both z > `z_threshold` and a shift of at
    least `min_shift_angstrom` -- a big z on a tiny shift is just a big
    dataset. `frac_at_limit`: share of the class's particles within 5% of
    the offset search limit (`offset_range_px`), meaning the range is too
    small to capture where they want to go. Uses rlnOriginX/YAngst
    (RELION 3.1+) or rlnOriginX/Y (px) x `pixel_size_angstrom`."""
    if {"rlnOriginXAngst", "rlnOriginYAngst"} <= set(df.columns):
        ox, oy = df["rlnOriginXAngst"].astype(float), df["rlnOriginYAngst"].astype(float)
    elif {"rlnOriginX", "rlnOriginY"} <= set(df.columns) and pixel_size_angstrom:
        ox, oy = df["rlnOriginX"].astype(float) * pixel_size_angstrom, df["rlnOriginY"].astype(float) * pixel_size_angstrom
    else:
        return None
    if class_col not in df.columns:
        return None
    limit = offset_range_px * pixel_size_angstrom if offset_range_px and pixel_size_angstrom else None
    rows = []
    for cls, idx in df.groupby(class_col).groups.items():
        inside = df.index.isin(idx)
        n_in, n_out = int(inside.sum()), int((~inside).sum())
        row = {"class": cls, "n": n_in,
               "mean_x": ox[inside].mean(), "mean_y": oy[inside].mean(),
               "std_x": ox[inside].std(), "std_y": oy[inside].std()}
        if n_out > 1 and n_in > 1:
            dx, dy = ox[inside].mean() - ox[~inside].mean(), oy[inside].mean() - oy[~inside].mean()
            shift = float(np.hypot(dx, dy))
            var = 0.5 * (ox.var() + oy.var())  # pooled per-axis variance
            se = float(np.sqrt(var * (1 / n_in + 1 / n_out))) or np.nan
            row.update(shift_angstrom=shift, z=shift / se)
            row["flag_shifted"] = bool(row["z"] > z_threshold and shift >= min_shift_angstrom)
        else:
            row.update(shift_angstrom=np.nan, z=np.nan, flag_shifted=False)
        if limit:
            r = np.hypot(ox[inside], oy[inside])
            row["frac_at_limit"] = float((r >= 0.95 * limit).mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("class")


def rank_class_differences(maps, compact_fraction=0.02, sigma_threshold=4.0):
    """Rank classes by how distinct they are from the mean of the others.

    `maps`: {class_number: 3D array}, same shape. For each class the
    difference to the mean of the other classes is scored by its peak in
    robust sigmas (1.4826 x MAD of the difference over the whole box).
    `n_voxels` counts voxels beyond `sigma_threshold`; `compact` is True when
    that is < `compact_fraction` of the box -- a localized blob (ligand-like)
    rather than a global change. Returns rows sorted by `peak_sigma`,
    each with the voxel position (z, y, x) of the peak. Needs >= 2 classes."""
    if len(maps) < 2:
        return []
    keys = list(maps)
    stack = np.stack([np.asarray(maps[k], dtype=np.float32) for k in keys])
    rows = []
    for i, k in enumerate(keys):
        others = np.delete(stack, i, axis=0).mean(axis=0)
        diff = stack[i] - others
        med = np.median(diff)
        sigma = 1.4826 * np.median(np.abs(diff - med)) or float(diff.std()) or 1.0
        z = (diff - med) / sigma
        peak_idx = np.unravel_index(np.argmax(np.abs(z)), z.shape)
        n_vox = int((np.abs(z) > sigma_threshold).sum())
        rows.append({
            "class": k,
            "peak_sigma": float(abs(z[peak_idx])),
            "peak_positive": bool(z[peak_idx] > 0),  # True: extra density in this class
            "peak_voxel_zyx": tuple(int(v) for v in peak_idx),
            "n_voxels": n_vox,
            "compact": 0 < n_vox < compact_fraction * z.size,
        })
    return sorted(rows, key=lambda r: -r["peak_sigma"])


def class_occupancy_trajectory(job_dir):
    """DataFrame (rows: iteration, columns: class number) of
    `rlnClassDistribution` from every run_it*_model.star of a Class3D job.
    Empty frame if none found."""
    series = {}
    for p in Path(job_dir).glob("run_it*_model.star"):
        m = re.search(r"_it(\d+)_model", p.name)
        if not m:
            continue
        blocks = starfile.read(p)
        classes = blocks.get("model_classes") if isinstance(blocks, dict) else blocks  # starfile returns a bare frame for one-block files
        if classes is None or "rlnClassDistribution" not in classes.columns:
            continue
        series[int(m.group(1))] = classes["rlnClassDistribution"].astype(float).reset_index(drop=True)
    if not series:
        return pd.DataFrame()
    df = pd.DataFrame(series).T.sort_index()
    df.columns = range(1, df.shape[1] + 1)
    return df


def collapsed_classes(trajectory, floor=0.01):
    """Class numbers whose final share is below `floor` after having held at
    least 5x that at some earlier iteration -- a class that emptied out."""
    if trajectory.empty or len(trajectory) < 2:
        return []
    last = trajectory.iloc[-1]
    return [int(c) for c in trajectory.columns if last[c] < floor and trajectory[c].max() >= 5 * floor]
