import struct

import numpy as np
import pandas as pd
import pytest

import relion_origin as ro


def _write_mrc(path, arr):
    nz, ny, nx = arr.shape
    path.write_bytes(struct.pack("<4i", nx, ny, nz, 2).ljust(1024, b"\0") + arr.astype("<f4").tobytes())


def _classified(n=6000, linked=False, seed=0):
    rng = np.random.default_rng(seed)
    cls = rng.choice([1, 2, 3, 4], size=n, p=[0.4, 0.3, 0.2, 0.1])
    conf = rng.uniform(0.3, 1.0, n)
    if linked:                                    # small classes hold the confidently aligned particles, large ones the vague ones
        conf = np.where(np.isin(cls, [3, 4]), rng.uniform(0.8, 1.0, n), rng.uniform(0.3, 0.7, n))
    return pd.DataFrame({"rlnClassNumber": cls, "rlnMaxValueProbDistribution": conf})


def test_raw_fraction_is_about_half_even_without_any_structure():
    # the miscalibration: the old "purity" number is ~0.5 for pure noise
    assert 0.4 < ro.classification_purity_fraction(_classified()) < 0.6


def test_mixing_excess_is_zero_for_noise_and_positive_when_confidence_follows_class():
    assert abs(ro.classification_mixing_excess(_classified())) < 0.03
    assert ro.classification_mixing_excess(_classified(linked=True)) > 0.2


def test_mixing_excess_needs_two_classes():
    assert ro.classification_mixing_excess(_classified().assign(rlnClassNumber=1)) == 0.0


def test_map_similarity_and_class_ordering(tmp_path):
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(8, 8, 8))
    assert ro.map_similarity(ref, ref * 3 + 5) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        ro.map_similarity(ref, ref[:4])
    job = tmp_path / "Class3D" / "job001"
    job.mkdir(parents=True)
    _write_mrc(tmp_path / "ref.mrc", ref)
    _write_mrc(job / "run_it001_class001.mrc", rng.normal(size=ref.shape))          # unrelated
    _write_mrc(job / "run_it001_class002.mrc", ref + 0.2 * rng.normal(size=ref.shape))
    _write_mrc(job / "run_it002_class002.mrc", ref + 0.1 * rng.normal(size=ref.shape))  # newest iteration wins
    _write_mrc(job / "run_it002_class003.mrc", rng.normal(size=(4, 4, 4)))           # other box size -> skipped
    sims = ro.class_reference_similarity(job, tmp_path / "ref.mrc", 4)
    assert set(sims) == {1, 2} and sims[2] > 0.9 > abs(sims[1])
    assert ro.class_reference_similarity(job, tmp_path / "missing.mrc", 4) == {}
