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


def test_clustered_class_is_flagged_with_five_classes():
    notes = annotate_particles(_df({1: 400, 2: 400, 3: 400, 4: 400, 5: 400}, clustered_class=3))
    labels = [n["label"] for n in notes]
    assert any(l == "clustered views: C3" for l in labels)
    assert all(n["level"] in ("good", "warn", "bad", "info") and n["detail"] for n in notes)
    assert not any("QC candidates" in l for l in labels)  # raw share is not calibrated, see annotations.py
    assert not any(l.startswith("mixed beyond chance") for l in labels)  # these classes are independent of confidence


def test_few_classes_get_an_info_line_not_a_flag():
    notes = annotate_particles(_df({1: 300, 2: 300}))
    orient = [n for n in notes if n["label"].startswith("orientation R=")]
    assert len(orient) == 1 and orient[0]["level"] == "info"


def test_single_refinement_without_class_column_still_reports_orientation():
    df = _df({1: 500}).drop(columns=["rlnClassNumber", "rlnMaxValueProbDistribution"])
    notes = annotate_particles(df)
    assert [n["label"][:14] for n in notes] == ["orientation R="]


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
