import pandas as pd

from relion_origin.diagnose import diagnose_class, diagnose_class_from_flags


def test_nyquist_limited_wins_over_everything_else():
    result = diagnose_class(mvpd_flagged_fraction=0.9, resolution_angstrom=10.0, nyquist_angstrom=9.0)
    assert result["bucket"] == "ctf_defocus_limited"


def test_ctf_inconsistent_flagged():
    result = diagnose_class(mvpd_flagged_fraction=0.1, ctf_inconsistent_fraction=0.5)
    assert result["bucket"] == "ctf_defocus_limited"


def test_heterogeneous_when_most_of_class_flagged():
    result = diagnose_class(mvpd_flagged_fraction=0.6)
    assert result["bucket"] == "heterogeneous"


def test_alignment_when_small_tail_flagged():
    result = diagnose_class(mvpd_flagged_fraction=0.05)
    assert result["bucket"] == "alignment"


def test_none_when_nothing_flagged():
    result = diagnose_class(mvpd_flagged_fraction=0.0)
    assert result["bucket"] == "none"


def test_diagnose_class_from_flags_derives_fraction():
    index = pd.RangeIndex(10)
    flags_by_metric = {
        "m1": pd.Series([True] * 7 + [False] * 3, index=index),
    }
    result = diagnose_class_from_flags(flags_by_metric, index)
    assert result["bucket"] == "heterogeneous"


if __name__ == "__main__":
    test_nyquist_limited_wins_over_everything_else()
    test_ctf_inconsistent_flagged()
    test_heterogeneous_when_most_of_class_flagged()
    test_alignment_when_small_tail_flagged()
    test_none_when_nothing_flagged()
    test_diagnose_class_from_flags_derives_fraction()
    print("all diagnose tests passed")
