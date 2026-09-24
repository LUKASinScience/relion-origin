"""Particle-flow tracking through the RELION pipeline, built on top of
reliontree's job-lineage graph instead of re-parsing default_pipeline.star.

reliontree's `build_tree()` already gives the job DAG (`parents`) and,
for Class3D/Refine3D jobs, a particle count from their `*_model.star`
(`stats[job]["n_particles"]`). Select/Extract/Import jobs don't write a
model.star, so those counts come from this module instead, by counting
rows in the job's own `particles.star` output when present -- the
RELION-standard node file name for a Select job's kept-particle subset.
"""

import re
from pathlib import Path

import pandas as pd

try:
    import reliontree
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "relion_origin.lineage needs the reliontree package (pip install reliontree) "
        "for pipeline-graph parsing -- see the build plan 'Relationship to reliontree'"
    ) from _exc

from .io import load_particles

find_project = reliontree.find_project
ProjectNotFoundError = reliontree.ProjectNotFoundError


def _latest_data_star(job_dir):
    """A finished Class3D/Refine3D job's full particle set: `run_data.star`
    once done, else the highest-numbered `run_itNNN_data.star` while
    running -- same convention reliontree's own `_latest_model_star` uses
    for `*_model.star`. Deliberately NOT the model.star particle count:
    RELION's gold-standard refinement splits particles into two
    independently-aligned random halves for unbiased FSC estimation (both
    halves still go into the final merged reconstruction -- this is
    standard RELION behavior, not a bug or a dropped-particle event), and
    a Refine3D job's `run_model.star` reports only one half's
    `_rlnGroupNrParticles` -- confirmed on the test dataset (run_model.star
    sums to 1822, run_data.star has all 3578 rows, 1822 ~ 3578/2). For
    particle-flow/loss tracking we want the full set both halves draw
    from, not one half's size, so this prefers run_data.star over
    reliontree's model.star-derived count."""
    final = job_dir / "run_data.star"
    if final.is_file():
        return final
    candidates = sorted(job_dir.glob("run_it*_data.star"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: int(re.search(r"_it(\d+)_", p.name).group(1)))


def _job_particle_count(job_dir, reliontree_count):
    """Prefer counting a job's own full particle-set STAR file over
    reliontree's best-effort model.star-derived count (see
    `_latest_data_star` docstring: for a gold-standard Refine3D job that
    count is one FSC half, not the full set flow-tracking wants). Order:
    Select/Extract's `particles.star` node file, then Class3D/Refine3D's
    `*_data.star`, then reliontree's count as a last resort for job types
    this module doesn't know the output name of."""
    for candidate in (job_dir / "particles.star", _latest_data_star(job_dir)):
        if candidate is not None and candidate.is_file():
            try:
                return len(load_particles(candidate))
            except Exception:
                continue  # malformed/unreadable -- try the next candidate
    return reliontree_count


def find_particle_file(job_dir):
    """The best available particle STAR file for a job directory: Select/
    Extract's `particles.star` node file, else Class3D/Refine3D's
    `*_data.star` (see `_latest_data_star`), else None if this job type
    doesn't carry a particle table at all (e.g. MaskCreate)."""
    job_dir = Path(job_dir)
    particles_star = job_dir / "particles.star"
    if particles_star.is_file():
        return particles_star
    return _latest_data_star(job_dir)


def list_jobs_with_particles(project_dir):
    """Every job in the project that has a usable particle STAR file, in
    pipeline order (root first). Returns a dict {job_id: Path}, meant for
    a "pick which job to analyze" UI -- e.g. relion-origin's GUI job
    selector -- so a user isn't stuck typing STAR file paths by hand.
    """
    project_dir = Path(project_dir)
    tree = reliontree.build_tree(project_dir)
    result = {}
    for job in [j for layer in tree["layers"] for j in layer]:
        particle_file = find_particle_file(tree["job_dirs"][job])
        if particle_file is not None:
            result[job] = particle_file
    return result


def particle_flow(project_dir):
    """Particle counts through the pipeline as a DataFrame: one row per
    job, with its particle count, its parent job(s), and the loss versus
    each parent (parent_count - this job's count; NaN where either side
    is unknown, e.g. a MaskCreate job that doesn't carry particles at all).

    Columns: job, n_particles, parents, resolution_angstrom, loss_vs_max_parent.
    """
    project_dir = Path(project_dir)
    tree = reliontree.build_tree(project_dir)

    counts = {}
    for job, job_dir in tree["job_dirs"].items():
        counts[job] = _job_particle_count(job_dir, tree["stats"][job]["n_particles"])

    rows = []
    for job in [j for layer in tree["layers"] for j in layer]:
        parents = sorted(tree["parents"].get(job, ()))
        parent_counts = [counts[p] for p in parents if counts.get(p) is not None]
        loss = None
        if counts.get(job) is not None and parent_counts:
            loss = max(parent_counts) - counts[job]
        rows.append(
            {
                "job": job,
                "n_particles": counts.get(job),
                "parents": ", ".join(parents) if parents else None,
                "resolution_angstrom": tree["stats"][job]["resolution_angstrom"],
                "loss_vs_max_parent": loss,
            }
        )
    return pd.DataFrame(rows)
