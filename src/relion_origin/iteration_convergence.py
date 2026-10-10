"""Class-assignment stability across a Class3D/Refine3D job's own
iterations, via overlap-matching between consecutive iterations -- RELION
doesn't keep class labels stable across iterations, so diffing labels
directly would misread a renumbered class as churn.

Prior art: `chenyiguang2026/RELION-Convergence-Inspector`
does this at much greater depth (angular change under Cn symmetry, map
correlation, three separate stability conclusions). This module covers only
the class-switch-rate piece, sized for what `diagnose.py` and
plateau detection in downstream tools need: a per-transition switch rate and
the iteration classification became stable, not a full convergence report.
"""

import re
from pathlib import Path

import pandas as pd

from .io import load_particles

_ITERATION_ID_COLS = ("rlnImageName", "rlnTomoParticleId", "rlnTomoParticleName")


def _iteration_data_stars(job_dir):
    """{iteration_number: path} for every run_itNNN_data.star in a job dir."""
    job_dir = Path(job_dir)
    result = {}
    for p in job_dir.glob("run_it*_data.star"):
        m = re.search(r"_it(\d+)_data", p.name)
        if m:
            result[int(m.group(1))] = p
    return dict(sorted(result.items()))


def _class_assignment(path, class_col, id_cols):
    df = load_particles(path)
    id_col = next((c for c in id_cols if c in df.columns), None)
    if id_col is None:
        raise ValueError(f"none of {id_cols} found in {path} -- no stable per-iteration particle id")
    return df.set_index(id_col)[class_col]


def _match_classes_by_overlap(prev, curr):
    """Map each `curr` class label to whichever `prev` class label most of
    its particles came from (majority vote) -- so a class-switch rate isn't
    inflated just because RELION renumbered classes between iterations."""
    mapping = {}
    for curr_class in curr.unique():
        members = prev[curr == curr_class]
        if len(members):
            mapping[curr_class] = members.value_counts().idxmax()
    return mapping


def iteration_convergence(job_dir, class_col="rlnClassNumber", id_cols=_ITERATION_ID_COLS):
    """Per-transition class-switch rate across a job's own iterations.

    Returns a DataFrame, one row per (iteration N-1 -> N) transition:
    `iteration` (the later of the pair), `switch_rate` (share of
    overlap-matched particles whose mapped class differs from the previous
    iteration), `n_matched`, `n_classes`. Empty if fewer than two
    `run_it*_data.star` files exist (job not run with K>1, or intermediate
    iteration files already cleaned up after completion).
    """
    stars = _iteration_data_stars(job_dir)
    iterations = sorted(stars)
    if len(iterations) < 2:
        return pd.DataFrame(columns=["iteration", "switch_rate", "n_matched", "n_classes"])

    rows = []
    prev_assignment = _class_assignment(stars[iterations[0]], class_col, id_cols)
    for it in iterations[1:]:
        curr_assignment = _class_assignment(stars[it], class_col, id_cols)
        common = prev_assignment.index.intersection(curr_assignment.index)
        if len(common) == 0:
            prev_assignment = curr_assignment
            continue
        mapping = _match_classes_by_overlap(prev_assignment, curr_assignment)
        mapped_curr = curr_assignment.loc[common].map(mapping)
        switched = mapped_curr.to_numpy() != prev_assignment.loc[common].to_numpy()
        rows.append(
            {
                "iteration": it,
                "switch_rate": float(switched.sum()) / len(common),
                "n_matched": len(common),
                "n_classes": int(curr_assignment.nunique()),
            }
        )
        prev_assignment = curr_assignment

    return pd.DataFrame(rows)


def stable_since_iteration(convergence_df, switch_rate_threshold=0.05, min_consecutive=3):
    """First iteration of the final sustained run of transitions at/under
    `switch_rate_threshold`, held for at least `min_consecutive` transitions
    through the end of the table -- None if it never reaches that or the
    table is shorter than `min_consecutive`.

    Same "classification became stable" definition as
    RELION-Convergence-Inspector: deliberately stricter than
    "the single quietest transition" -- one calm iteration surrounded by
    churn isn't convergence.
    """
    stable = convergence_df["switch_rate"] <= switch_rate_threshold
    if len(stable) < min_consecutive or not stable.tail(min_consecutive).all():
        return None
    idx = len(stable) - 1
    while idx > 0 and stable.iloc[idx - 1]:
        idx -= 1
    return int(convergence_df["iteration"].iloc[idx])
