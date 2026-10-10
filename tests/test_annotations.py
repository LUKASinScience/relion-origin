import json

import numpy as np
import pandas as pd
import pytest

from relion_origin.annotations import annotate_particles, write_annotations


def _df(n_per_class, clustered_class=None, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for cls, n in n_per_class.items():
        if cls == clustered_class:
            rot, tilt = rng.normal(0, 2, n), rng.normal(90, 2, n)
        else:
            tilt = np.degrees(np.arccos(rng.uniform(-1, 1, n))); rot = rng.uniform(-180, 180, n)
        rows.append(pd.DataFrame({"rlnClassNumber": cls, "rlnAngleRot": rot, "rlnAngleTilt": tilt,
                                  "rlnMaxValueProbDistribution": rng.uniform(0.2, 1.0, n)}))
    return pd.concat(rows, ignore_index=True)


def _views(n, z_lo, z_hi, seed=0, rot_hi=360.0, tomo=False):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"rlnAngleRot": rng.uniform(0, rot_hi, n) - 180 * (rot_hi == 360.0),
                       "rlnAngleTilt": np.degrees(np.arccos(rng.uniform(z_lo, z_hi, n)))})
    return df.assign(rlnTomoName="ts_01") if tomo else df


def test_views_gap_pill_for_single_particle_data_is_a_warning_and_even_data_gets_none():
    even = annotate_particles(_views(20000, -1, 1))
    assert not any(n["label"].startswith("views:") for n in even)
    one_hemisphere = [n for n in annotate_particles(_views(20000, 0, 1)) if n["label"].startswith("views:")]
    assert len(one_hemisphere) == 1 and one_hemisphere[0]["level"] == "warn" and "50%" in one_hemisphere[0]["label"]
    assert "preferred orientation" in one_hemisphere[0]["detail"].lower()
    assert not any("clustered" in n["label"] for n in annotate_particles(_views(20000, 0, 1)))


def test_views_gap_pill_for_subtomogram_data_is_only_informational():
    notes = [n for n in annotate_particles(_views(20000, 0, 1, tomo=True)) if n["label"].startswith("views:")]
    assert notes and notes[0]["level"] == "info" and "picking geometry" in notes[0]["detail"]


def test_symmetry_is_folded_before_gaps_are_reported():
    wedge = _views(20000, -1, 1, rot_hi=60.0)                       # one 60-degree wedge of rot
    assert any(n["label"].startswith("views:") for n in annotate_particles(wedge, sym="C1"))
    assert not any(n["label"].startswith("views:") for n in annotate_particles(wedge, sym="C6"))
    assert not any(n["label"].startswith("views:") for n in annotate_particles(wedge, sym="D2"))   # unsupported: no verdict


def test_write_annotations_roundtrip_and_name_check(tmp_path):
    path = write_annotations(tmp_path, "relion-origin", {"Class3D/job001": [{"label": "x", "level": "warn"}]})
    assert path == tmp_path / ".reliontree" / "annotations" / "relion-origin.json"
    assert json.loads(path.read_text())["Class3D/job001"][0]["label"] == "x"
    with pytest.raises(ValueError):
        write_annotations(tmp_path, "../evil", {})


def test_mixing_beyond_chance_is_annotated_only_when_it_exceeds_chance():
    linked = pd.DataFrame({"rlnClassNumber": [1] * 14 + [2] * 6,
                           "rlnMaxValueProbDistribution": [0.1 + 0.03 * i for i in range(14)] + [0.8, 0.85, 0.9, 0.9, 0.95, 0.95]})
    notes = annotate_particles(linked)
    assert notes and notes[0]["label"].startswith("mixed beyond chance") and notes[0]["level"] == "warn"


def test_project_type_overrides_the_column_guess():
    df = _views(20000, 0, 1)                                   # no rlnTomoName column
    assert [n for n in annotate_particles(df) if n["label"].startswith("views:")][0]["level"] == "warn"
    assert [n for n in annotate_particles(df, tomography=True) if n["label"].startswith("views:")][0]["level"] == "info"
