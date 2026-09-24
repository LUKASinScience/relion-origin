"""relion-origin-rerun CLI: write a QC candidate list to a RELION-loadable
particle subset STAR file. See rerun.py for what this closes."""

import argparse
import sys

from .io import load_particles
from .qc import candidate_misclassified
from .rerun import write_candidate_subset


def main(argv=None):
    parser = argparse.ArgumentParser(prog="relion-origin-rerun", description=__doc__)
    parser.add_argument("star_file", help="RELION STAR file with a particles table")
    parser.add_argument("output", help="output STAR file for the candidate subset")
    parser.add_argument("--class-column", default="rlnClassNumber")
    parser.add_argument(
        "--confidence-column",
        default="rlnMaxValueProbDistribution",
        help="alignment-confidence column candidate_misclassified() ranks on",
    )
    parser.add_argument(
        "--label",
        choices=["in_good_class_low_confidence", "in_bad_class_high_confidence"],
        default=None,
        help="only write this candidate label (default: both)",
    )
    args = parser.parse_args(argv)

    try:
        df = load_particles(args.star_file)
    except Exception as exc:
        sys.exit("could not read %s: %s" % (args.star_file, exc))

    if args.confidence_column not in df.columns:
        sys.exit(
            "column %r not found in %s -- pass --confidence-column, available: %s"
            % (args.confidence_column, args.star_file, list(df.columns))
        )

    candidates = candidate_misclassified(
        df, class_col=args.class_column, confidence_col=args.confidence_column
    )
    candidate_values = [args.label] if args.label else None
    try:
        n = write_candidate_subset(candidates, args.output, candidate_values=candidate_values)
    except ValueError as exc:
        sys.exit(str(exc))

    print("wrote %d candidate particles to %s" % (n, args.output))
    print("point a new RELION job's 'Input images STAR file' at it for a targeted rerun.")


if __name__ == "__main__":
    main()
