"""Smoke test for ctf.flag_dose_gaps()."""

import pandas as pd

from relion_origin.ctf import flag_dose_gaps


def test_no_gaps_in_clean_series():
    block = pd.DataFrame({"rlnMicrographPreExposure": [i * 2.72 for i in range(10)]})
    flags = flag_dose_gaps({"tomoA": block})
    assert not flags["tomoA"].any()


def test_dose_symmetric_direction_reversal_not_flagged():
    # decreasing toward 0 then increasing -- a dose-symmetric scheme's
    # geometric tilt order, not acquisition order. Must not false-flag.
    doses = [10 * 2.72, 8 * 2.72, 6 * 2.72, 4 * 2.72, 2 * 2.72, 1 * 2.72, 3 * 2.72, 5 * 2.72, 7 * 2.72]
    block = pd.DataFrame({"rlnMicrographPreExposure": doses})
    flags = flag_dose_gaps({"tomoA": block})
    assert not flags["tomoA"].any()


def test_missing_tilt_gap_is_flagged():
    doses = [i * 2.72 for i in range(6)]
    doses[4] += 5 * 2.72  # a jump way off the single-tilt grid
    block = pd.DataFrame({"rlnMicrographPreExposure": doses})
    flags = flag_dose_gaps({"tomoA": block})
    assert flags["tomoA"].iloc[4]


if __name__ == "__main__":
    test_no_gaps_in_clean_series()
    test_dose_symmetric_direction_reversal_not_flagged()
    test_missing_tilt_gap_is_flagged()
    print("dose gap tests passed")
