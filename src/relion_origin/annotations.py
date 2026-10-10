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

from .orientation import angular_coverage_score, flag_clustered_orientation
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


def annotate_particles(df, class_col="rlnClassNumber"):
    """Annotation dicts for one job's particle table (pure; no file access)."""
    out = []
    excess = classification_mixing_excess(df, class_col)
    if excess > MIXING_EXCESS_WARN:
        out.append({"label": "mixed beyond chance (+%.0f pts)" % (100 * excess), "level": "warn",
                    "detail": "%.0f points more particles than chance sit in a small class with high alignment "
                              "confidence or in a large class with low confidence. Candidates to check, not verdicts." % (100 * excess)})
    if {"rlnAngleRot", "rlnAngleTilt"} <= set(df.columns):
        work = df if class_col in df.columns else df.assign(**{class_col: 1})
        scores = angular_coverage_score(work, class_col)
        listing = ", ".join("C%s R=%.2f" % (c, r) for c, r in scores.items())
        if len(scores) >= 4:
            flagged = flag_clustered_orientation(work, class_col)
            bad = [c for c, f in flagged.items() if f]
            if bad:
                out.append({"label": "clustered views: " + ", ".join("C%s" % c for c in bad), "level": "warn",
                            "detail": "Viewing directions of these classes are more one-sided than your other classes "
                                      "(R: 0 = even, 1 = one direction). " + listing})
        else:
            out.append({"label": "orientation R=%.2f" % scores.max(), "level": "info",
                        "detail": "Mean resultant length of viewing directions (0 even, 1 one direction). " + listing})
    return out


def annotate_project(project_dir, class_col="rlnClassNumber"):
    """{job: annotations} for every Class3D/Refine3D job that has a readable particle table.
    Returns (annotations, skipped) where skipped = [(job, reason)]."""
    from .lineage import list_jobs_with_particles
    from .io import load_particles
    result, skipped = {}, []
    for job, star in list_jobs_with_particles(project_dir).items():
        if job.split("/")[0] not in ANNOTATED_TYPES:
            continue
        try:
            notes = annotate_particles(load_particles(star), class_col)
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
