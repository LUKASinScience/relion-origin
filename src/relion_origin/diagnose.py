"""Which of the three usual reasons is limiting a class's resolution, and
what to change in RELION to move past it -- a heuristic, not a verdict.

Theory (see the build plan §27): a class/average that isn't at the resolution its
particle count should allow is limited by one of three things, and each
needs a different fix -- more classification is useless if the real problem
is CTF, and re-extracting at a finer pixel size is useless if the real
problem is a mixed population:

1. Heterogeneity: MVPD/NSS bad across the WHOLE class (not a tail) -> the
   class mixes structures -> classify further (raise K, focused/masked
   classification) or model continuous motion (multibody/3D variability).
2. Alignment: MVPD/NSS bad only for a TAIL of particles, CTF fine for those
   -> genuine per-particle alignment noise -> finer angular/offset sampling,
   local search, tighter mask, per-particle CtfRefine.
3. CTF/defocus: resolution sits near the pixel/box Nyquist ceiling, or many
   tomograms/tilts are CTF-inconsistent -> re-extract at finer pixel size
   (see box_size.py) or fix defocus estimation before spending more compute
   on classification or alignment.
"""


def diagnose_class(
    mvpd_flagged_fraction,
    resolution_angstrom=None,
    nyquist_angstrom=None,
    ctf_inconsistent_fraction=None,
    heterogeneity_threshold=0.3,
    nyquist_margin=1.15,
):
    """Return {bucket, reason, suggestion} for one class.

    `mvpd_flagged_fraction`: share of this class's particles flagged by
    `qc.flag_class_outliers` on MVPD/NSS -- high AND spread across most of
    the class points at heterogeneity, a small share points at alignment
    noise on those specific particles.
    """
    if (
        resolution_angstrom is not None
        and nyquist_angstrom is not None
        and resolution_angstrom <= nyquist_angstrom * nyquist_margin
    ):
        return {
            "bucket": "ctf_defocus_limited",
            "reason": f"resolution ({resolution_angstrom:.1f} A) is within "
            f"{int((nyquist_margin - 1) * 100)}% of the current pixel/box "
            f"Nyquist limit ({nyquist_angstrom:.1f} A)",
            "suggestion": "re-extract at a finer pixel size / larger box "
            "(box_size.recommended_box_size_from_ctf) before doing more "
            "classification or finer alignment",
        }
    if ctf_inconsistent_fraction is not None and ctf_inconsistent_fraction > 0.2:
        return {
            "bucket": "ctf_defocus_limited",
            "reason": f"{ctf_inconsistent_fraction:.0%} of tomograms/tilts "
            "flagged as CTF-inconsistent (ctf.flag_inconsistent_tomograms)",
            "suggestion": "fix defocus estimation (re-run CTF estimation or "
            "per-particle CtfRefine) before trusting classification/alignment results",
        }
    if mvpd_flagged_fraction is not None and mvpd_flagged_fraction > heterogeneity_threshold:
        return {
            "bucket": "heterogeneous",
            "reason": f"{mvpd_flagged_fraction:.0%} of this class's particles "
            "flagged on alignment confidence -- spread across most of the "
            "class, not a small tail",
            "suggestion": "classify further: raise the number of classes (K), "
            "or run focused/masked classification on the variable region "
            "(or multibody refinement if the variation is continuous motion)",
        }
    if mvpd_flagged_fraction is not None and mvpd_flagged_fraction > 0:
        return {
            "bucket": "alignment",
            "reason": f"only {mvpd_flagged_fraction:.0%} of particles flagged, "
            "a tail rather than the whole class",
            "suggestion": "finer angular/offset sampling or local search from "
            "this run, a tighter mask to raise SNR, or per-particle CtfRefine "
            "-- or just exclude the flagged tail (qc.candidate_misclassified)",
        }
    return {
        "bucket": "none",
        "reason": "no metric crossed a flagging threshold",
        "suggestion": "class looks consistent with current settings",
    }


def diagnose_class_from_flags(flags_by_metric, class_index, **kwargs):
    """`diagnose_class`, fed from per-particle flag Series that a caller
    already computed (e.g. `qc.flag_class_outliers` per metric, as
    server.py's `_build_page_cache` and relion-autopilot's engine both do)
    instead of recomputing MVPD/NSS outlier stats here -- one shared
    definition of the fraction so the GUI and relion-autopilot agree.

    `flags_by_metric`: {metric_name: pd.Series(bool)} over the full particle
    table. `class_index`: the index of the class being diagnosed.
    `**kwargs`: passed through to `diagnose_class` (resolution_angstrom,
    nyquist_angstrom, ctf_inconsistent_fraction, ...).
    """
    if not flags_by_metric or len(class_index) == 0:
        mvpd_flagged_fraction = None
    else:
        combined = None
        for flags in flags_by_metric.values():
            f = flags.reindex(class_index).fillna(False)
            combined = f if combined is None else (combined | f)
        mvpd_flagged_fraction = float(combined.sum()) / len(class_index)
    return diagnose_class(mvpd_flagged_fraction, **kwargs)
