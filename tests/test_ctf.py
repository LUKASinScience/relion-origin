"""Smoke test for the tomogram-level CTF join (ctf.py)."""

import pandas as pd

from relion_origin.ctf import attach_tomogram_ctf, tomogram_defocus_summary


def _fake_ctf_by_tomogram():
    return {
        "tomoA": pd.DataFrame({"rlnDefocusU": [10000, 10500, 9800], "rlnDefocusV": [9800, 10200, 9600]}),
        "tomoB": pd.DataFrame({"rlnDefocusU": [30000, 31000], "rlnDefocusV": [29500, 30800]}),
    }


def test_tomogram_defocus_summary():
    summary = tomogram_defocus_summary(_fake_ctf_by_tomogram())
    assert set(summary.index) == {"tomoA", "tomoB"}
    assert summary.loc["tomoA", "n_tilts"] == 3
    assert summary.loc["tomoB", "defocus_mean"] > summary.loc["tomoA", "defocus_mean"]


def test_attach_tomogram_ctf_matches_and_flags_missing():
    particles = pd.DataFrame({"rlnMicrographName": ["tomoA", "tomoA", "tomoB", "tomoC"]})
    result = attach_tomogram_ctf(particles, _fake_ctf_by_tomogram(), name_col="rlnMicrographName")
    assert result["_tomoDefocusMean"].notna().sum() == 3  # tomoC has no CTF block
    assert result.loc[result["rlnMicrographName"] == "tomoC", "_tomoDefocusMean"].isna().all()


if __name__ == "__main__":
    test_tomogram_defocus_summary()
    test_attach_tomogram_ctf_matches_and_flags_missing()
    print("all ctf tests passed")
