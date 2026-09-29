"""Regression test for box_size.py against the aliasing_box_calculator.js
reference values it was ported from (cross-checked against the JS with
node on 2026-09-22, hardcoded here so the test needs no JS runtime)."""

import pandas as pd

from relion_origin.box_size import (
    aliasing_free_box_size,
    recommended_box_size_from_ctf,
    worst_case_defocus_sta,
    worst_case_defocus_spa,
)


def test_aliasing_free_box_size_matches_js_reference():
    # kv=300, defocus=3.0 um, cs=2.7mm, pix=1.06 A/px, res=4.0 A
    r = aliasing_free_box_size(300, 3.0 * 10000, 2.7, 1.06, 4.0)
    assert abs(r["wavelength_angstrom"] - 0.019688) < 1e-5
    assert abs(r["delocalization_angstrom"] - 144.439) < 1e-2
    assert r["box_px_even"] == 386
    assert r["box_px_fft"] == 400

    # kv=300, defocus=4.27 um, cs=2.7mm, pix=11.056 A/px (bin8), res=20 A
    r2 = aliasing_free_box_size(300, 4.27 * 10000, 2.7, 11.056, 20.0)
    assert r2["box_px_even"] == 12
    assert r2["box_px_fft"] == 12


def test_worst_case_defocus_sta():
    ctf_by_tomogram = {
        "tomoA": pd.DataFrame({"rlnDefocusU": [10000, 12000], "rlnDefocusV": [9800, 11800]}),
        "tomoB": pd.DataFrame({"rlnDefocusU": [30000, 15000], "rlnDefocusV": [29500, 14800]}),
    }
    assert worst_case_defocus_sta(ctf_by_tomogram) == 30000


def test_recommended_box_size_from_ctf_uses_worst_case():
    ctf_by_tomogram = {
        "tomoA": pd.DataFrame({"rlnDefocusU": [10000], "rlnDefocusV": [9800]}),
        "tomoB": pd.DataFrame({"rlnDefocusU": [80000], "rlnDefocusV": [79500]}),
    }
    result = recommended_box_size_from_ctf(
        kv=300, cs_mm=2.7, pixel_size_angstrom=11.056, resolution_angstrom=20.0, ctf_by_tomogram=ctf_by_tomogram
    )
    assert result["worst_case_defocus_angstrom"] == 80000
    assert result["box_px_fft"] >= result["box_px_raw"]


def test_worst_case_defocus_spa():
    particles = pd.DataFrame({"rlnDefocusU": [10000, 25000, 15000], "rlnDefocusV": [9800, 24500, 14800]})
    assert worst_case_defocus_spa(particles) == 25000


def test_recommended_box_size_from_ctf_spa_path():
    # same physics as STA, different data source: rlnDefocusU/V directly on
    # the particle table instead of tomograms.star's per-tilt CTF blocks.
    particles = pd.DataFrame({"rlnDefocusU": [10000, 25000], "rlnDefocusV": [9800, 24500]})
    result = recommended_box_size_from_ctf(
        kv=300, cs_mm=2.7, pixel_size_angstrom=0.86, resolution_angstrom=2.5, particles_df=particles
    )
    assert result["worst_case_defocus_angstrom"] == 25000
    assert result["box_px_fft"] >= result["box_px_raw"]
    # a coarser target resolution needs a smaller box at the same defocus --
    # sanity check that the resolution sensitivity behaves as expected.
    coarser = aliasing_free_box_size(300, 25000, 2.7, 0.86, 10.0)
    assert coarser["box_px_fft"] < result["box_px_fft"]


def test_recommended_box_size_from_ctf_requires_exactly_one_source():
    particles = pd.DataFrame({"rlnDefocusU": [10000], "rlnDefocusV": [9800]})
    try:
        recommended_box_size_from_ctf(300, 2.7, 0.86, 2.5, ctf_by_tomogram={}, particles_df=particles)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when both ctf_by_tomogram and particles_df are given")


def test_diameter_adds_to_edge():
    base = aliasing_free_box_size(300, 30000, 2.7, 1.0, 3.0)
    with_d = aliasing_free_box_size(300, 30000, 2.7, 1.0, 3.0, diameter_angstrom=200)
    assert abs(with_d["edge_angstrom"] - base["edge_angstrom"] - 200) < 1e-6


def test_spa_percentile_ignores_outlier():
    df = pd.DataFrame({"rlnDefocusU": [20000.0] * 999 + [90000.0], "rlnDefocusV": [20000.0] * 1000})
    assert worst_case_defocus_spa(df) == 90000.0
    assert worst_case_defocus_spa(df, 99) == 20000.0


if __name__ == "__main__":
    test_aliasing_free_box_size_matches_js_reference()
    test_worst_case_defocus_sta()
    test_recommended_box_size_from_ctf_uses_worst_case()
    test_worst_case_defocus_spa()
    test_recommended_box_size_from_ctf_spa_path()
    test_recommended_box_size_from_ctf_requires_exactly_one_source()
    test_diameter_adds_to_edge()
    test_spa_percentile_ignores_outlier()
    print("box_size tests passed")
