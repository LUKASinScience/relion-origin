import struct

import numpy as np
import pandas as pd

import relion_origin as ro


def _write_mrc(path, arr):
    nz, ny, nx = arr.shape
    header = struct.pack("<4i", nx, ny, nz, 2).ljust(1024, b"\0")
    path.write_bytes(header + arr.astype("<f4").tobytes())


def test_read_mrc_roundtrip(tmp_path):
    arr = np.random.default_rng(0).normal(size=(6, 7, 8)).astype(np.float32)
    _write_mrc(tmp_path / "a.mrc", arr)
    assert np.allclose(ro.read_mrc(tmp_path / "a.mrc"), arr)


def test_rank_class_differences_finds_ligand_blob():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 0.05, size=(32, 32, 32)).astype(np.float32)
    maps = {c: base + rng.normal(0, 0.05, size=base.shape).astype(np.float32) for c in (1, 2, 3)}
    maps[2][10:13, 20:23, 5:8] += 2.0  # small extra density only in class 2
    ranked = ro.rank_class_differences(maps)
    assert ranked[0]["class"] == 2 and ranked[0]["compact"] and ranked[0]["peak_positive"]
    assert all(0 <= v < 32 for v in ranked[0]["peak_voxel_zyx"])
    assert ranked[0]["peak_sigma"] > 1.5 * ranked[1]["peak_sigma"]  # others see the same blob as *missing* density
    assert not ranked[1]["peak_positive"]


def test_offset_summary_flags_shifted_class():
    rng = np.random.default_rng(2)
    n = 2000
    cls = np.repeat([1, 2], n)
    x = rng.normal(0, 3, 2 * n)
    x[cls == 2] += 1.5  # class 2 sits 1.5 A off
    df = pd.DataFrame({"rlnClassNumber": cls, "rlnOriginXAngst": x, "rlnOriginYAngst": rng.normal(0, 3, 2 * n)})
    s = ro.offset_summary(df, offset_range_px=2, pixel_size_angstrom=1.0)
    assert s["flag_shifted"].all()  # symmetric: each class is off from the other
    assert 1.0 < s.loc[2, "shift_angstrom"] < 2.0
    same = df.assign(rlnClassNumber=rng.integers(1, 3, 2 * n))
    assert not ro.offset_summary(same)["flag_shifted"].any()


def test_offset_summary_needs_offset_columns():
    assert ro.offset_summary(pd.DataFrame({"rlnClassNumber": [1, 2]})) is None


def test_occupancy_trajectory_detects_collapse(tmp_path):
    import starfile

    for it, dist in ((1, [0.4, 0.3, 0.3]), (2, [0.6, 0.4, 0.0])):
        starfile.write({"model_classes": pd.DataFrame({"rlnClassDistribution": dist})}, tmp_path / ("run_it%03d_model.star" % it))
    traj = ro.class_occupancy_trajectory(tmp_path)
    assert traj.shape == (2, 3) and ro.collapsed_classes(traj) == [3]


def test_class_occupancy_trajectory_reads_only_the_classes_block(tmp_path):
    def model(dist):
        rows = "\n".join("class%d.mrc %s 0.5" % (i + 1, d) for i, d in enumerate(dist))
        return ("data_model_general\n\n_rlnCurrentResolution 4.0\n\n\ndata_model_classes\n\nloop_ \n"
                "_rlnReferenceImage #1 \n_rlnClassDistribution #2 \n_rlnAccuracyRotations #3 \n%s\n\n\n"
                "data_model_class_1\n\nloop_ \n_rlnSpectralIndex #1 \n_rlnClassDistribution #2 \n0 99\n\n" % rows)
    (tmp_path / "run_it001_model.star").write_text(model([0.6, 0.4]))
    (tmp_path / "run_it002_model.star").write_text(model([0.7, 0.3]))
    (tmp_path / "run_it003_model.star").write_text("data_model_general\n\n_rlnCurrentResolution 4.0\n")  # no classes block: skipped
    df = ro.class_occupancy_trajectory(tmp_path)
    assert df.shape == (2, 2) and list(df.index) == [1, 2] and df.loc[2, 1] == 0.7 and df.loc[1, 2] == 0.4
