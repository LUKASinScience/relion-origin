"""relion-origin CLI. `relion-origin <star-file-or-project-dir>` opens the
QC GUI. Pointed at a RELION project directory instead of a single STAR
file, the GUI gets a job picker so you can switch which job's particles
you're looking at without restarting the server."""

import argparse
import sys
from pathlib import Path

from .io import load_particles, load_tomograms_star
from .server import DEFAULT_PORT, serve


def main(argv=None):
    parser = argparse.ArgumentParser(prog="relion-origin", description=__doc__)
    parser.add_argument(
        "target",
        help="a RELION particle STAR file, or a RELION project directory (contains "
        "default_pipeline.star) to get a job picker in the GUI",
    )
    parser.add_argument("--class-column", default="rlnClassNumber")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-open", action="store_true", help="don't launch a browser (e.g. on an HPC login node)")
    parser.add_argument(
        "--job",
        default=None,
        help='when --target is a project directory, which job to open initially (e.g. "Refine3D/job006"); '
        "defaults to the last job in pipeline order",
    )
    parser.add_argument(
        "--tomograms-star",
        default=None,
        help="STA *_tomograms.star -- adds CTF-consistency, dose-gap, and tomogram-edge-distance "
        "signals to the overview (optional, STA only)",
    )
    args = parser.parse_args(argv)

    target = Path(args.target)
    project_dir, jobs, initial_job = None, None, None
    if target.is_dir():
        try:
            from .lineage import list_jobs_with_particles
        except ImportError as exc:
            sys.exit("could not use --target as a project directory: %s" % exc)
        try:
            jobs = list_jobs_with_particles(target)
        except Exception as exc:
            sys.exit("could not read the pipeline under %s: %s" % (target, exc))
        if not jobs:
            sys.exit("no job under %s has a usable particle STAR file" % target)
        if args.job and args.job not in jobs:
            sys.exit("job %r not found. Available: %s" % (args.job, sorted(jobs)))
        project_dir = target
        initial_job = args.job or list(jobs)[-1]
        star_file = jobs[initial_job]
    else:
        star_file = target

    try:
        df = load_particles(star_file)
    except Exception as exc:
        sys.exit("could not read %s: %s" % (star_file, exc))

    if args.class_column not in df.columns:
        sys.exit("column %r not found in %s. Available: %s" % (args.class_column, star_file, list(df.columns)))

    ctf_by_tomogram, tomogram_dims = None, None
    if args.tomograms_star:
        try:
            tomogram_dims, ctf_by_tomogram = load_tomograms_star(args.tomograms_star)
        except Exception as exc:
            sys.exit("could not read %s: %s" % (args.tomograms_star, exc))

    serve(
        df,
        class_col=args.class_column,
        title=str(initial_job) if project_dir else str(star_file),
        port=args.port,
        open_browser=not args.no_open,
        ctf_by_tomogram=ctf_by_tomogram,
        tomogram_dims=tomogram_dims,
        jobs=jobs,
        current_job=initial_job,
    )


if __name__ == "__main__":
    main()
