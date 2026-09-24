"""Synthetic check that overlap-matching (not label equality) is what
drives the switch rate -- a pure class relabeling between iterations must
read as 0% switched, not 100%."""

import pandas as pd
import starfile

from relion_origin.iteration_convergence import iteration_convergence, stable_since_iteration


def _write_iteration(job_dir, iteration, classes):
    df = pd.DataFrame(
        {
            "rlnImageName": ["%d@a.mrcs" % i for i in range(len(classes))],
            "rlnClassNumber": classes,
        }
    )
    starfile.write(df, job_dir / ("run_it%03d_data.star" % iteration))


def test_pure_relabeling_reads_as_zero_switch_rate(tmp_path):
    _write_iteration(tmp_path, 0, [1, 1, 2, 2])
    _write_iteration(tmp_path, 1, [2, 2, 1, 1])  # same grouping, labels swapped
    df = iteration_convergence(tmp_path)
    assert len(df) == 1
    assert df.iloc[0]["switch_rate"] == 0.0


def test_real_reshuffle_is_detected(tmp_path):
    _write_iteration(tmp_path, 0, [1, 1, 2, 2])
    _write_iteration(tmp_path, 1, [1, 2, 1, 2])  # genuinely different grouping
    df = iteration_convergence(tmp_path)
    assert df.iloc[0]["switch_rate"] > 0.0


def test_stable_since_iteration_needs_sustained_low_switch_rate():
    df = pd.DataFrame(
        {
            "iteration": [1, 2, 3, 4, 5],
            "switch_rate": [0.5, 0.3, 0.02, 0.01, 0.03],
            "n_matched": [100] * 5,
            "n_classes": [4] * 5,
        }
    )
    assert stable_since_iteration(df, min_consecutive=3) == 3


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        test_pure_relabeling_reads_as_zero_switch_rate(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_real_reshuffle_is_detected(Path(d))
    test_stable_since_iteration_needs_sustained_low_switch_rate()
    print("synthetic iteration_convergence tests passed")
