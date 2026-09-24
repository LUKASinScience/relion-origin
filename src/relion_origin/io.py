"""Load RELION STAR particle tables and normalize SPA/STA column names."""

from pathlib import Path

import starfile

# STA (subtomogram averaging) and SPA (single-particle) STAR files name the
# same concepts differently. Code that doesn't care which pipeline produced
# a file asks for the canonical name and gets whichever column exists.
COLUMN_ALIASES = {
    "source_name": ("rlnTomoName", "rlnMicrographName"),
    "coord_x": ("rlnCenteredCoordinateXAngst", "rlnCoordinateX"),
    "coord_y": ("rlnCenteredCoordinateYAngst", "rlnCoordinateY"),
    "coord_z": ("rlnCenteredCoordinateZAngst", "rlnCoordinateZ"),
}


def load_particles(path):
    """Load the particle table of a RELION STAR file as a DataFrame.

    Handles both the plain single-block format and the optics-group format
    (data_optics + data_particles) newer RELION versions emit -- returns
    just the particle table either way, picking the largest block if the
    file has no block literally named "particles".
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("no such file: %s" % path)
    data = starfile.read(path)
    if isinstance(data, dict):
        if "particles" in data:
            return data["particles"].copy()
        return max(data.values(), key=len).copy()
    return data.copy()


def resolve_column(df, canonical_name):
    """Return the actual column name in `df` for a canonical alias, or None."""
    for candidate in COLUMN_ALIASES.get(canonical_name, (canonical_name,)):
        if candidate in df.columns:
            return candidate
    return None


def load_tomograms_star(path):
    """Load a RELION5 STA `*_tomograms.star` file (referenced by
    `rlnTomoTomogramsFile` in an `*_optimisation_set.star`).

    This file has one `data_global` block (per-tomogram dimensions, pixel
    size, optics) plus one `data_<tomogram_name>` block per tomogram
    holding its per-tilt-image CTF (`rlnDefocusU/V/Angle`) and projection
    geometry. For STA data this -- not the particle STAR -- is where CTF
    lives; SPA data keeps CTF on the particles/optics-group tables
    directly via `load_particles`.

    Returns (dims_df, ctf_by_tomogram): `dims_df` is the data_global table
    indexed by `rlnTomoName`; `ctf_by_tomogram` is a dict mapping each
    tomogram name to its per-tilt CTF DataFrame.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("no such file: %s" % path)
    data = starfile.read(path)
    if "global" not in data:
        raise ValueError("%s has no data_global block -- not a tomograms.star file?" % path)
    global_block = data["global"]
    duplicates = global_block["rlnTomoName"][global_block["rlnTomoName"].duplicated()].unique()
    if len(duplicates):
        raise ValueError(
            "%s has duplicate rlnTomoName entries in data_global (%s) -- indexing by name "
            "would silently return multiple rows per lookup" % (path, list(duplicates[:5]))
        )
    dims_df = global_block.set_index("rlnTomoName")
    ctf_by_tomogram = {name: block for name, block in data.items() if name != "global"}
    return dims_df, ctf_by_tomogram


def center_coordinates(x, y, z, size_x, size_y, size_z, tilt_series_pixel_size, particle_pixel_size):
    """Convert corner-origin pixel coordinates (RELION's rlnCoordinateX/Y/Z
    convention, at the particle-extraction pixel size/binning) to
    tomogram-centered Angstrom coordinates (pytom/WarpTools'
    rlnCenteredCoordinateX/Y/ZAngst convention).

    `size_x/y/z` (rlnTomoSizeX/Y/Z from load_tomograms_star's dims_df) are
    tomogram pixel dimensions at the *tilt-series* pixel size -- almost
    always a different binning level than the particle coordinates, so the
    two pixel sizes are separate arguments, not one shared value. Physical
    tomogram size in Angstrom is size_px * tilt_series_pixel_size,
    independent of the particle-extraction binning.
    """
    import numpy as np

    tomo_size_x_angst = size_x * tilt_series_pixel_size
    tomo_size_y_angst = size_y * tilt_series_pixel_size
    tomo_size_z_angst = size_z * tilt_series_pixel_size
    return (
        np.asarray(x) * particle_pixel_size - tomo_size_x_angst / 2,
        np.asarray(y) * particle_pixel_size - tomo_size_y_angst / 2,
        np.asarray(z) * particle_pixel_size - tomo_size_z_angst / 2,
    )


def tomo_base(name):
    """Strip common suffixes so tomogram/micrograph names from different
    pipeline stages (pytom vs. RELION import vs. WarpTools) collapse to the
    same matching key.

    'pos47_ts_002_unsorted_11.06Apx' -> 'pos47_ts_002_unsorted'
    'pos47_ts_002_unsorted.tomostar' -> 'pos47_ts_002_unsorted'
    """
    import re

    name = name.strip()
    name = re.sub(r"\.tomostar$", "", name)
    name = re.sub(r"_[\d.]+Apx$", "", name)
    name = re.sub(r"_[\d.]+A$", "", name)
    return name
