"""Hand QC findings to reliontree's job tree as annotations: pills on the job
card and notes in its panel. The contract is only a file --
`<project>/.reliontree/annotations/<tool>.json`, `{job: [{label, level, detail}]}`
with level good / warn / bad / info -- so neither side imports the other for it
(reliontree reads it, any tool can write it; see reliontree's
`relion_annotations.py`).

Only signals that discriminate on real data are written. The raw QC-candidate
share (`classification_purity_fraction`) is NOT: with its default confidence
threshold (the median) it flags about half of all particles of any classification
(0.38-0.58 on 30 real jobs, the same as shuffled class labels). Its chance-corrected
form, `classification_mixing_excess`, is annotated only when it clears
MIXING_EXCESS_WARN (3 of those 30 jobs).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from .orientation import orientation_coverage
from .qc import classification_mixing_excess

TOOL = "relion-origin"
ANNOTATED_TYPES = ("Class3D", "Refine3D")
MIXING_EXCESS_WARN = 0.05   # share of particles flagged beyond what chance gives


def write_annotations(project_dir, tool, jobs):
    """Write `jobs` ({job: [annotation dicts]}) as this tool's annotation file; returns its path.
    An empty dict still writes the file, so stale annotations from a previous run disappear."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", tool):
        raise ValueError("tool name %r: use letters, digits, '_', '-', '.' only" % tool)
    folder = Path(project_dir) / ".reliontree" / "annotations"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (tool + ".json")
    path.write_text(json.dumps(jobs, indent=1), encoding="utf-8")
    return path


GAPS_NOTE = 0.2        # report views once this share of the (symmetry-folded) viewing sphere is poorly covered
GAPS_WARN = 0.5        # single-particle: more than half of all views nearly empty is a real preferred orientation


def annotate_particles(df, class_col="rlnClassNumber", sym="C1", tomography=None):
    """Annotation dicts for one job's particle table (pure; no file access). `sym`: the job's `sym_name`.
    `tomography`: True for a subtomogram-averaging project (reliontree's `tree["experiment"]`); None = guess from the
    table (a `rlnTomoName` column)."""
    out = []
    excess = classification_mixing_excess(df, class_col)
    if excess > MIXING_EXCESS_WARN:
        out.append({"label": "mixed beyond chance (+%.0f pts)" % (100 * excess), "level": "warn",
                    "detail": "%.0f points more particles than chance sit in a small class with high alignment "
                              "confidence or in a large class with low confidence. Candidates to check, not verdicts." % (100 * excess)})
    if {"rlnAngleRot", "rlnAngleTilt"} <= set(df.columns):
        cov = orientation_coverage(df, sym=sym)
        if cov is not None and cov["poor_fraction"] >= GAPS_NOTE:
            if tomography is None:
                tomography = "rlnTomoName" in df.columns
            pct = 100 * cov["poor_fraction"]
            level = "warn" if pct >= 100 * GAPS_WARN and not tomography else "info"
            why = ("Subtomogram averaging: the picking geometry (e.g. particles on a membrane) limits the views, so gaps are "
                   "expected here; check that the missing wedge is not the only information in them. "
                   if tomography else
                   "A preferred orientation: the map will be anisotropic and the resolution estimate optimistic. ")
            out.append({"label": "views: %.0f%% of sphere sparse" % pct, "level": level,
                        "detail": "%.0f%% of all viewing directions hold fewer than 20%% of the particles an even distribution would "
                                  "put there (symmetry %s taken into account, evenness %.2f of 1). %sSee the orientation globe in reliontree."
                                  % (pct, sym or "C1", cov["evenness"], why)})
    return out


def annotate_project(project_dir, class_col="rlnClassNumber"):
    """{job: annotations} for every Class3D/Refine3D job that has a readable particle table.
    Returns (annotations, skipped) where skipped = [(job, reason)]."""
    import reliontree
    from .io import load_particles
    from .lineage import find_particle_file
    tree = reliontree.build_tree(project_dir)
    result, skipped = {}, []
    for job in reliontree.ordered_jobs(tree):
        if job.split("/")[0] not in ANNOTATED_TYPES:
            continue
        star = find_particle_file(tree["job_dirs"][job])
        if star is None:
            continue
        try:
            notes = annotate_particles(load_particles(star), class_col, tree["options"].get(job, {}).get("sym_name", "C1"),
                                       tomography={"sta": True, "spa": False}.get(tree["experiment"]["type"]))
        except Exception as exc:  # one unreadable job must not stop the rest
            skipped.append((job, str(exc)))
            continue
        if notes:
            result[job] = notes
    return result, skipped


def main(argv=None):
    parser = argparse.ArgumentParser(prog="relion-origin-annotate", description=__doc__)
    parser.add_argument("project", help="RELION project directory")
    parser.add_argument("--class-column", default="rlnClassNumber")
    args = parser.parse_args(argv)
    project = Path(args.project)
    if not (project / "default_pipeline.star").is_file():
        sys.exit("no default_pipeline.star in %s" % project)
    notes, skipped = annotate_project(project, args.class_column)
    path = write_annotations(project, TOOL, notes)
    print("wrote %d job(s) to %s -- open them with `reliontree watch %s`" % (len(notes), path, project))
    for job, why in skipped:
        print("skipped %s: %s" % (job, why), file=sys.stderr)


if __name__ == "__main__":
    main()
