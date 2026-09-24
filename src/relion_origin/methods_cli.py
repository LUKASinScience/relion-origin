"""relion-origin-methods CLI: draft a methods-section paragraph from QC
results already on disk. Run the QC steps you want described first
(cli.py's GUI, or the Python API), then point this at the same files."""

import argparse
import sys

from .compare import compare_resolution
from .ctf import flag_inconsistent_tomograms, tomogram_defocus_summary
from .io import load_particles, load_tomograms_star
from .methods import draft_methods_paragraph
from .qc import candidate_misclassified


def main(argv=None):
    parser = argparse.ArgumentParser(prog="relion-origin-methods", description=__doc__)
    parser.add_argument("particles_star", help="the particle STAR file the QC candidates were scored on")
    parser.add_argument("--class-column", default="rlnClassNumber")
    parser.add_argument("--confidence-column", default="rlnMaxValueProbDistribution")
    parser.add_argument("--tomograms-star", default=None, help="STA *_tomograms.star, for the CTF-consistency sentence")
    parser.add_argument("--project-dir", default=None, help="RELION project dir, for the resolution-comparison sentence")
    parser.add_argument("--job-before", default=None, help='e.g. "Refine3D/job006"')
    parser.add_argument("--job-after", default=None, help='e.g. "Refine3D/job010" (the rerun on the QC subset)')
    args = parser.parse_args(argv)

    try:
        df = load_particles(args.particles_star)
    except Exception as exc:
        sys.exit("could not read %s: %s" % (args.particles_star, exc))

    candidates = None
    if args.confidence_column in df.columns:
        candidates = candidate_misclassified(df, class_col=args.class_column, confidence_col=args.confidence_column)

    ctf_summary = None
    if args.tomograms_star:
        try:
            _dims, ctf_by_tomo = load_tomograms_star(args.tomograms_star)
            ctf_summary = flag_inconsistent_tomograms(tomogram_defocus_summary(ctf_by_tomo))
        except Exception as exc:
            print("warning: could not read %s (%s), skipping CTF sentence" % (args.tomograms_star, exc), file=sys.stderr)

    comparison = None
    if args.project_dir and args.job_before and args.job_after:
        try:
            comparison = compare_resolution(args.project_dir, args.job_before, args.job_after)
        except Exception as exc:
            print("warning: could not compare resolution (%s), skipping" % exc, file=sys.stderr)

    paragraph = draft_methods_paragraph(
        candidates_df=candidates,
        ctf_summary=ctf_summary,
        resolution_comparison=comparison,
        job_before=args.job_before,
        job_after=args.job_after,
    )
    if not paragraph:
        sys.exit("nothing to draft -- no QC results found for the given inputs")
    print(paragraph)


if __name__ == "__main__":
    main()
