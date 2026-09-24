"""relion-origin-boxsize CLI: aliasing-free box size from a project's real
worst-case defocus (see box_size.py for why worst-case, not a guess)."""

import argparse
import sys

from .box_size import aliasing_free_box_size, recommended_box_size_from_ctf
from .io import load_particles, load_tomograms_star


def main(argv=None):
    parser = argparse.ArgumentParser(prog="relion-origin-boxsize", description=__doc__)
    parser.add_argument("--kv", type=float, required=True, help="acceleration voltage, kV")
    parser.add_argument("--cs", type=float, required=True, help="spherical aberration, mm")
    parser.add_argument("--pixel-size", type=float, required=True, help="particle-extraction pixel size, A/px")
    parser.add_argument("--resolution", type=float, required=True, help="target resolution, A")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tomograms-star", help="STA *_tomograms.star -- worst-case defocus read from its per-tilt CTF blocks")
    group.add_argument("--particles-star", help="SPA particles STAR with rlnDefocusU/V -- worst-case defocus read from it")
    group.add_argument("--defocus", type=float, help="skip project data, use this defocus directly (A)")
    args = parser.parse_args(argv)

    try:
        if args.defocus is not None:
            result = aliasing_free_box_size(args.kv, args.defocus, args.cs, args.pixel_size, args.resolution)
            result["worst_case_defocus_angstrom"] = args.defocus
        elif args.tomograms_star:
            _dims, ctf_by_tomo = load_tomograms_star(args.tomograms_star)
            result = recommended_box_size_from_ctf(
                args.kv, args.cs, args.pixel_size, args.resolution, ctf_by_tomogram=ctf_by_tomo
            )
        else:
            df = load_particles(args.particles_star)
            result = recommended_box_size_from_ctf(
                args.kv, args.cs, args.pixel_size, args.resolution, particles_df=df
            )
    except Exception as exc:
        sys.exit(str(exc))

    print("worst-case defocus: %.0f A (%.2f um)" % (result["worst_case_defocus_angstrom"], result["worst_case_defocus_angstrom"] / 10000))
    print("wavelength: %.5f A" % result["wavelength_angstrom"])
    print("max delocalization: %.1f A" % result["delocalization_angstrom"])
    print("minimum aliasing-free box edge: %.1f A" % result["edge_angstrom"])
    print("recommended box size: %d px (raw %.1f, even %d)" % (result["box_px_fft"], result["box_px_raw"], result["box_px_even"]))


if __name__ == "__main__":
    main()
