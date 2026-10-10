"""Which class looks like the reference you gave the classification?

Measured on a real STA project (14 Select jobs, 56 classes): the classes the
user kept were the ones most similar to the Class3D input reference -- ranking
by this similarity put the kept classes in the top-k 93% of the time, while
ranking by mean alignment confidence (48%) or log-likelihood (42%) was no better
than chance, and the largest class was kept every time. As a *threshold*
("keep if similarity >= x") it was poor (exact selection in 5 of 14 jobs), so
use it to order classes, not to decide.
"""

from pathlib import Path

import numpy as np

from .focused import read_mrc


def _normalise(volume):
    volume = np.asarray(volume, dtype=float)
    volume = volume - volume.mean()
    return volume / (volume.std() + 1e-12)


def map_similarity(a, b):
    """Normalised cross-correlation of two equally shaped volumes (-1..1, same frame, no alignment)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("maps differ in shape: %s vs %s" % (a.shape, b.shape))
    return float((_normalise(a) * _normalise(b)).mean())


def _latest_class_map(job_dir, number):
    """The newest `run_itNNN_classMMM.mrc` of a Class3D job (a finished job also has `run_classMMM.mrc`)."""
    candidates = sorted(Path(job_dir).glob("run_it*_class%03d.mrc" % number))
    if candidates:
        return candidates[-1]
    final = Path(job_dir) / ("run_class%03d.mrc" % number)
    return final if final.is_file() else None


def class_reference_similarity(job_dir, reference_path, n_classes):
    """{class number: similarity to `reference_path`} for classes 1..n_classes of a Class3D job;
    a class is omitted when its map is missing, unreadable or has another box size than the reference
    (e.g. the reference was rescaled), because cross-correlation then means nothing."""
    try:
        reference = read_mrc(reference_path)
    except (OSError, ValueError):
        return {}
    out = {}
    for number in range(1, n_classes + 1):
        path = _latest_class_map(job_dir, number)
        if path is None:
            continue
        try:
            out[number] = map_similarity(read_mrc(path), reference)
        except (OSError, ValueError):
            continue
    return out
