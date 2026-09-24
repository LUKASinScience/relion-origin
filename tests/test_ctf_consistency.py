"""Smoke test for the CTF-consistency flags (tomogram-level and per-tilt)."""

import pandas as pd

from relion_origin.ctf import flag_inconsistent_tilts, flag_inconsistent_tomograms, tomogram_defocus_summary


def _fake_ctf_by_tomogram():
    # tomoA, tomoC-tomoF: stable defocus, small realistic jitter.
    # tomoB: one wild outlier tilt + overall much higher spread than the rest.
    def stable(base, jitter):
        return pd.DataFrame(
            {
                "rlnDefocusU": [base + j for j in jitter],
                "rlnDefocusV": [base - 100 + j for j in jitter],
            }
        )

    return {
        "tomoA": stable(10000, [0, 50, -30, 20, -10, 40, 0, -20]),
        "tomoC": stable(15000, [0, 45, -25, 15, -15, 35, 5, -25]),
        "tomoD": stable(12000, [0, 55, -35, 25, -5, 45, -5, -15]),
        "tomoE": stable(18000, [0, 40, -20, 10, -20, 30, 10, -30]),
        "tomoF": stable(11000, [0, 60, -40, 30, 0, 50, -10, -10]),
        "tomoB": pd.DataFrame(
            {
                "rlnDefocusU": [20000, 20100, 20200, 60000, 20300, 20150, 20050, 20000],
                "rlnDefocusV": [19800, 19900, 20000, 59800, 20100, 19950, 19850, 19800],
            }
        ),
    }


def test_flag_inconsistent_tomograms():
    summary = tomogram_defocus_summary(_fake_ctf_by_tomogram())
    flagged = flag_inconsistent_tomograms(summary, z_threshold=1.0)
    assert flagged.loc["tomoB", "_ctf_inconsistent"]
    assert not flagged.loc["tomoA", "_ctf_inconsistent"]


def test_flag_inconsistent_tilts():
    flags = flag_inconsistent_tilts(_fake_ctf_by_tomogram())
    assert not flags["tomoA"].any()
    assert flags["tomoB"].iloc[3]  # the 60000 outlier row


if __name__ == "__main__":
    test_flag_inconsistent_tomograms()
    test_flag_inconsistent_tilts()
    print("ctf consistency tests passed")
