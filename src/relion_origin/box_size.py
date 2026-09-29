"""Aliasing-free particle-box-size calculator, ported from the project's
own `aliasing_box_calculator.js` (Cryo-EM-Guide), which follows Warp/M's
(Tegunov) CTF-delocalization approach.

At high defocus, high-resolution signal is spatially delocalized in real
space; a particle box that's too small under-samples the resulting rapid
CTF oscillations in Fourier space (CTF aliasing), losing that signal
irrecoverably. This estimates the maximum delocalization across a target
resolution range and the resulting minimum aliasing-free box.

The connection to the rest of relion-origin: this calculator needs a
defocus value, and the box size must be large enough for the *worst*
(highest) defocus in the project, not an average or a guess -- a box
sized for the mean defocus will alias exactly the highest-defocus
tomograms/micrographs. `ctf.py`'s `tomogram_defocus_summary()` (STA) or a
particle table's `rlnDefocusU/V` (SPA) already have that real maximum;
`recommended_box_size_from_ctf()` below uses it instead of a guess.
"""

import math

import numpy as np
import pandas as pd


def _wavelength_angstrom(kv):
    """Relativistic electron wavelength in Angstrom, `kv` in kilovolts."""
    return 12.2643 / math.sqrt(1000 * kv + 0.9784 * kv * kv)


def _next_even(n):
    return math.ceil(n / 2) * 2


def _next_fft_friendly(n):
    """Smallest integer >= n whose only prime factors are 2, 3, 5 (FFT-fast
    sizes RELION/Warp box sizes conventionally use)."""
    candidate = math.ceil(n)
    while True:
        temp = candidate
        for factor in (2, 3, 5):
            while temp % factor == 0:
                temp //= factor
        if temp == 1:
            return candidate
        candidate += 1


def aliasing_free_box_size(kv, defocus_angstrom, cs_mm, pixel_size_angstrom, resolution_angstrom, steps=400, diameter_angstrom=0):
    """Minimum box edge (Angstrom and pixels) that avoids CTF aliasing at
    the given defocus, up to `resolution_angstrom`.

    `defocus_angstrom`: use the *worst-case* (highest-magnitude) defocus
    in the dataset the extracted particles will come from, not a mean --
    see module docstring.

    Returns a dict: wavelength_angstrom, delocalization_angstrom,
    edge_angstrom, box_px_raw, box_px_even, box_px_fft (the practical
    recommendation: FFT-friendly and even).
    """
    wavelength = _wavelength_angstrom(kv)
    cs_angstrom = cs_mm * 1e7
    s_max = 1.0 / resolution_angstrom

    s = np.linspace(0, s_max, steps + 1)
    delocalization = wavelength * s * np.abs(defocus_angstrom - cs_angstrom * wavelength**2 * s**2)
    d_max = float(delocalization.max())

    # Warp/M: box >= particle diameter + delocalization on both sides.
    edge_angstrom = diameter_angstrom + 2 * math.sqrt(2) * d_max
    box_px_raw = edge_angstrom / pixel_size_angstrom

    return {
        "wavelength_angstrom": wavelength,
        "delocalization_angstrom": d_max,
        "edge_angstrom": edge_angstrom,
        "box_px_raw": box_px_raw,
        "box_px_even": _next_even(box_px_raw),
        "box_px_fft": _next_fft_friendly(box_px_raw),
    }


def worst_case_defocus_sta(ctf_by_tomogram):
    """Highest per-tilt defocus (max of DefocusU/V) across every tomogram
    in a `ctf_by_tomogram` dict (from `io.load_tomograms_star`). This is
    the value to size a box against, not the mean."""
    worst = None
    for block in ctf_by_tomogram.values():
        block_max = max(
            pd.to_numeric(block["rlnDefocusU"], errors="coerce").max(),
            pd.to_numeric(block["rlnDefocusV"], errors="coerce").max(),
        )
        if worst is None or block_max > worst:
            worst = block_max
    return worst


def worst_case_defocus_spa(particles_df, percentile=100):
    """Highest per-particle defocus (max of DefocusU/V) in a SPA particle
    table -- the SPA equivalent of `worst_case_defocus_sta`. `percentile`
    < 100 ignores a few outlier particles with a bad CTF fit."""
    per_particle = pd.concat([
        pd.to_numeric(particles_df["rlnDefocusU"], errors="coerce"),
        pd.to_numeric(particles_df["rlnDefocusV"], errors="coerce"),
    ], axis=1).max(axis=1)
    return float(np.nanpercentile(per_particle, percentile))


def recommended_box_size_from_ctf(kv, cs_mm, pixel_size_angstrom, resolution_angstrom, ctf_by_tomogram=None, particles_df=None, diameter_angstrom=0, percentile=100):
    """`aliasing_free_box_size` using the real worst-case defocus from
    either STA (`ctf_by_tomogram`, from `io.load_tomograms_star`) or SPA
    (`particles_df`, needs `rlnDefocusU/V`) project data, instead of a
    guessed defocus. Pass exactly one of the two.
    """
    if (ctf_by_tomogram is None) == (particles_df is None):
        raise ValueError("recommended_box_size_from_ctf: pass exactly one of ctf_by_tomogram or particles_df")

    worst_defocus = (
        worst_case_defocus_sta(ctf_by_tomogram) if ctf_by_tomogram is not None else worst_case_defocus_spa(particles_df, percentile)
    )
    result = aliasing_free_box_size(kv, worst_defocus, cs_mm, pixel_size_angstrom, resolution_angstrom, diameter_angstrom=diameter_angstrom)
    result["worst_case_defocus_angstrom"] = worst_defocus
    return result
