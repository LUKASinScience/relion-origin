"""Per-tomogram CTF/defocus QC for STA data.

STA CTF doesn't live on the particle table: RELION
fits *one* reference defocus per tilt image, not per particle -- the
per-particle value RELION itself uses during refinement is a geometric
projection of the 3D particle position onto that per-tilt reference,
computed from `rlnTomoProjX/Y/Z/W` inside RELION's own C++ code (not
published as a formula, confirmed by research).

Reproducing that exactly is out of scope here. What this module gives you
instead is an honest, simpler QC signal: for each tomogram, the mean and
spread of its per-tilt defocus values (`rlnDefocusU/V` in
`io.load_tomograms_star`'s `ctf_by_tomogram` blocks). A tomogram whose
tilt-series has a wildly varying or extreme defocus is worth flagging even
without reconstructing RELION's per-particle geometry -- that's the
tilt-series' own CTF-estimation quality, which every particle in it
inherits regardless of its exact 3D position.
"""

import numpy as np
import pandas as pd


def tomogram_defocus_summary(ctf_by_tomogram):
    """Mean/std/min/max of DefocusU/V per tomogram, across its tilt images.

    `ctf_by_tomogram`: the dict from `io.load_tomograms_star` (tomogram
    name -> per-tilt DataFrame with rlnDefocusU/V columns).

    Returns a DataFrame indexed by tomogram name.
    """
    rows = {}
    for name, block in ctf_by_tomogram.items():
        defocus_u = pd.to_numeric(block["rlnDefocusU"], errors="coerce")
        defocus_v = pd.to_numeric(block["rlnDefocusV"], errors="coerce")
        mean_defocus = (defocus_u + defocus_v) / 2
        rows[name] = {
            "n_tilts": len(block),
            "defocus_mean": mean_defocus.mean(),
            "defocus_std": mean_defocus.std(),
            "defocus_min": mean_defocus.min(),
            "defocus_max": mean_defocus.max(),
            "astigmatism_mean": (defocus_u - defocus_v).abs().mean(),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def flag_inconsistent_tomograms(summary, z_threshold=2.5):
    """Flag tomograms whose defocus spread or astigmatism is a project-wide
    outlier (robust z-score over `tomogram_defocus_summary`'s output).

    This is the literature-backed lever, not the MVPD/NSS-based candidate
    ranking in qc.py: NovaCTF (PMC5614107) showed 3D/per-tilt CTF quality
    is a real, measured STA resolution lever, up to 3.4 A in their
    example. A tomogram with unusually erratic per-tilt defocus is a
    direct proxy for a tilt-series whose CTF estimation itself struggled
    -- worth excluding or re-estimating before trusting its particles,
    regardless of how well any individual particle aligned.

    Returns `summary` with an added boolean `_ctf_inconsistent` column.
    """
    result = summary.copy()
    flagged = pd.Series(False, index=summary.index)
    for col in ("defocus_std", "astigmatism_mean"):
        median = summary[col].median()
        mad = (summary[col] - median).abs().median()
        if mad == 0 or pd.isna(mad):
            continue
        # 1.4826 makes MAD a consistent estimator of the standard deviation
        # for normally-distributed data -- standard robust z-score scaling.
        robust_z = (summary[col] - median).abs() / (1.4826 * mad)
        flagged = flagged | (robust_z >= z_threshold)
    result["_ctf_inconsistent"] = flagged
    return result


def flag_inconsistent_tilts(ctf_by_tomogram, z_threshold=3.5, window=5):
    """Flag individual tilt images whose defocus jumps away from its
    neighbours in the acquisition sequence -- a proxy for a per-tilt CTF
    fit that likely failed, independent of the tomogram-level summary
    above. Assumes each tomogram's block is ordered by acquisition/tilt
    (RELION5's `tomograms.star` writes it that way).

    Returns a dict {tomogram_name: boolean Series aligned to that
    tomogram's ctf_by_tomogram block, True = flagged}.
    """
    flags = {}
    for name, block in ctf_by_tomogram.items():
        mean_defocus = (pd.to_numeric(block["rlnDefocusU"], errors="coerce") + pd.to_numeric(block["rlnDefocusV"], errors="coerce")) / 2
        local_median = mean_defocus.rolling(window, center=True, min_periods=1).median()
        residual = (mean_defocus - local_median).abs()
        mad = residual.median()
        if mad == 0 or pd.isna(mad):
            flags[name] = pd.Series(False, index=block.index)
            continue
        robust_z = residual / (1.4826 * mad)
        flags[name] = robust_z >= z_threshold
    return flags


def flag_dose_gaps(ctf_by_tomogram, dose_col="rlnMicrographPreExposure", max_multiple=4.5):
    """Flag tilt images whose cumulative-dose step from their neighbour in
    the block doesn't look like a whole multiple of the tilt series' own
    per-tilt dose -- a proxy for a missing or duplicated tilt image.

    Deliberately does NOT assume the dose is monotonically increasing
    down the block: RELION5's `tomograms.star` orders tilts by geometry
    (tilt angle), not acquisition order, so a dose-symmetric scheme's
    dose *decreases* toward 0 deg from one direction and increases from
    the other -- verified on the test dataset (dose values run 163 ->
    ... -> falling, not rising, within a single tomogram's block).
    Working on the absolute step size, not its sign or trend, avoids
    baking in an acquisition-order assumption that would false-flag
    every dose-symmetric tilt series.

    The single-tilt dose unit is estimated per-tomogram as the smallest
    non-zero |diff| in the block (robust as long as most consecutive
    pairs are single-step -- true for standard dose-symmetric schemes,
    confirmed on the test dataset: steps are almost all 1x that unit,
    with rare 3x jumps at the branch crossover, well under `max_multiple`).

    Returns a dict {tomogram_name: boolean Series aligned to that
    tomogram's ctf_by_tomogram block, True = flagged}. The first tilt in
    each block is never flagged (no previous step to compare against).
    """
    flags = {}
    for name, block in ctf_by_tomogram.items():
        dose = pd.to_numeric(block[dose_col], errors="coerce")
        step = dose.diff().abs()
        nonzero_steps = step[step > 1e-6]
        if len(nonzero_steps) == 0:
            flags[name] = pd.Series(False, index=block.index)
            continue
        unit = nonzero_steps.min()
        multiple = step / unit
        nearest_multiple = multiple.round()
        off_grid = (multiple - nearest_multiple).abs() > (1.0 / max_multiple)
        too_large = nearest_multiple > max_multiple
        flags[name] = (off_grid | too_large).fillna(False)
    return flags


def attach_tomogram_ctf(particles_df, ctf_by_tomogram, name_col="rlnMicrographName"):
    """Attach each particle's tomogram-level defocus mean/spread as new
    columns, joined by tomogram name (not a per-particle geometric CTF --
    see module docstring for why).

    Adds `_tomoDefocusMean`, `_tomoDefocusStd`, `_tomoAstigmatismMean` to a
    copy of `particles_df`. Particles whose tomogram isn't in
    `ctf_by_tomogram` get NaN, not a crash -- a partially-covered project
    (e.g. a tomogram dropped after CTF estimation) is common.
    """
    summary = tomogram_defocus_summary(ctf_by_tomogram)
    result = particles_df.copy()
    joined = summary.reindex(result[name_col].astype(str))
    result["_tomoDefocusMean"] = joined["defocus_mean"].to_numpy()
    result["_tomoDefocusStd"] = joined["defocus_std"].to_numpy()
    result["_tomoAstigmatismMean"] = joined["astigmatism_mean"].to_numpy()
    return result
