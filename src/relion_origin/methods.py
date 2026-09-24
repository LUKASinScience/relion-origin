"""Best-effort "Methods" paragraph draft assembled from relion-origin's
own QC results -- a starting point to edit into a paper's methods
section, not a citation-ready sentence (same spirit as reliontree's own
`draft_methods_paragraph`, which this deliberately mirrors).

Publication-readiness is also a growth lever, not just convenience: a
methods paragraph that names the tool is how a niche scientific tool like
this one actually spreads through the field (see BAUPLAN.md ##12).
"""


def _format_number(n):
    try:
        return format(int(round(float(n))), ",")
    except (TypeError, ValueError):
        return str(n)


def qc_candidate_sentence(candidates_df, candidate_col="_qc_candidate", confidence_col="rlnMaxValueProbDistribution"):
    """One sentence on the candidate-ranking step (qc.candidate_misclassified
    output), or None if `candidates_df` has no flagged particles."""
    if candidate_col not in candidates_df.columns:
        return None
    counts = candidates_df[candidate_col].value_counts()
    low_confidence = counts.get("in_good_class_low_confidence", 0)
    high_confidence = counts.get("in_bad_class_high_confidence", 0)
    if low_confidence == 0 and high_confidence == 0:
        return None
    return (
        "Of %s particles, per-particle metadata QC (relion-origin) flagged %s as "
        "low-confidence (%s) within their assigned class and %s as high-confidence "
        "despite belonging to an otherwise lower-quality class, as candidates for "
        "manual review."
        % (
            _format_number(len(candidates_df)),
            _format_number(low_confidence),
            confidence_col,
            _format_number(high_confidence),
        )
    )


def ctf_consistency_sentence(flagged_summary, flag_col="_ctf_inconsistent"):
    """One sentence on the CTF-consistency check (ctf.flag_inconsistent_tomograms
    output), or None if nothing was flagged."""
    if flag_col not in flagged_summary.columns:
        return None
    n_flagged = int(flagged_summary[flag_col].sum())
    if n_flagged == 0:
        return None
    return (
        "Tilt-series CTF estimation was screened for defocus/astigmatism outliers "
        "across the tilt series (relion-origin); %s of %s tomograms showed "
        "inconsistent per-tilt defocus and were flagged for review."
        % (n_flagged, len(flagged_summary))
    )


def resolution_comparison_sentence(comparison, job_before, job_after):
    """One sentence on a before/after resolution comparison (compare.
    compare_resolution output), or None if either side is missing a
    resolution. Assumes -- doesn't verify -- that `job_after` is actually
    a rerun on a QC-flagged subset; that framing is the caller's claim,
    this only formats the numbers."""
    before, after = comparison.get("before"), comparison.get("after")
    if before is None or after is None:
        return None
    direction = "improved" if comparison["delta_angstrom"] > 0 else "did not improve"
    return (
        "A targeted refinement on the flagged candidate subset (%s, following %s) "
        "%s resolution from %.1f Å to %.1f Å."
        % (job_after, job_before, direction, before, after)
    )


def draft_methods_paragraph(candidates_df=None, ctf_summary=None, resolution_comparison=None, job_before=None, job_after=None):
    """Assemble whichever QC steps were actually run into a short draft
    paragraph. Every argument is optional -- pass only what you have.
    Ends with a caveat sentence (candidates, not verdicts) whenever any
    QC ranking sentence is included, matching this tool's own README/
    BAUPLAN position on what its candidate lists mean."""
    sentences = []
    candidate_sentence = qc_candidate_sentence(candidates_df) if candidates_df is not None else None
    if candidate_sentence:
        sentences.append(candidate_sentence)
    if ctf_summary is not None:
        ctf_sentence = ctf_consistency_sentence(ctf_summary)
        if ctf_sentence:
            sentences.append(ctf_sentence)
    if resolution_comparison is not None and job_before and job_after:
        res_sentence = resolution_comparison_sentence(resolution_comparison, job_before, job_after)
        if res_sentence:
            sentences.append(res_sentence)

    if candidate_sentence:
        sentences.append(
            "Flagged particles were treated as candidates for review, not as a "
            "verdict, since RELION does not report a full per-particle multi-class "
            "likelihood."
        )

    return " ".join(sentences) if sentences else ""
