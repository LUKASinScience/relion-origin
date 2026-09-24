"""relion-origin-compare CLI: resolution before/after between two jobs."""

import argparse
import sys

from .compare import compare_resolution


def main(argv=None):
    parser = argparse.ArgumentParser(prog="relion-origin-compare", description=__doc__)
    parser.add_argument("project_dir", help="RELION project dir (contains default_pipeline.star)")
    parser.add_argument("job_before", help='e.g. "Refine3D/job006"')
    parser.add_argument("job_after", help='e.g. "Refine3D/job010" (a rerun on a QC-filtered subset)')
    args = parser.parse_args(argv)

    try:
        result = compare_resolution(args.project_dir, args.job_before, args.job_after)
    except ValueError as exc:
        sys.exit(str(exc))

    if result["before"] is None or result["after"] is None:
        sys.exit(
            "no resolution found for %s: %r"
            % (args.job_before if result["before"] is None else args.job_after, result)
        )

    print("%s: %.2f A" % (args.job_before, result["before"]))
    print("%s: %.2f A" % (args.job_after, result["after"]))
    sign = "better" if result["delta_angstrom"] > 0 else "worse"
    print("delta: %.2f A %s" % (abs(result["delta_angstrom"]), sign))


if __name__ == "__main__":
    main()
