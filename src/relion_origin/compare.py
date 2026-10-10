"""Before/after resolution comparison between two RELION jobs -- the actual
evidence this tool's own honesty section says is
missing: a measured Angstrom number for what a QC-candidate-driven rerun
bought you, not a claim.

Typical use: run the original Refine3D job, use rerun.py to write a QC
candidate subset, run a new Refine3D job on it (or on the original set
with candidates removed), then compare the two jobs' resolution here.
"""

from pathlib import Path

try:
    import reliontree
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "relion_origin.compare needs the reliontree package (pip install reliontree) "
        "for job resolution parsing"
    ) from _exc


def _job_resolution(project_dir, job):
    """Resolution (Angstrom, gold-standard masked FSC if a PostProcess job
    ran, else the job's own current-resolution estimate) via reliontree's
    job_stats -- same parsing reliontree's own tree view uses, so this
    number matches what you'd see there."""
    tree = reliontree.build_tree(project_dir, selected=job, max_hops=0)
    if job not in tree["stats"]:
        raise ValueError("job %r not found under %s" % (job, project_dir))
    return tree["stats"][job]["resolution_angstrom"]


def compare_resolution(project_dir, job_before, job_after):
    """Resolution of `job_before` vs. `job_after` (both RELION job ids
    like "Refine3D/job006", relative to `project_dir`).

    Returns {"before": float|None, "after": float|None, "delta_angstrom":
    float|None}. `delta_angstrom` is before - after: positive means
    `job_after` resolved to a *better* (smaller Angstrom) resolution.
    None wherever a job hasn't finished / has no PostProcess or
    Refine3D/Class3D model.star to read a resolution from yet.
    """
    project_dir = Path(project_dir)
    before = _job_resolution(project_dir, job_before)
    after = _job_resolution(project_dir, job_after)
    delta = (before - after) if (before is not None and after is not None) else None
    return {"before": before, "after": after, "delta_angstrom": delta}
