"""Per-particle QC scoring: outlier detection and per-class metric summaries.

Generalizes what the test dataset's plot_lcc_per_class.py already did by
hand for one metric (LCCmax) into a reusable function for any numeric
column -- CTF resolution, defocus, alignment confidence, template-matching
score, whatever the STAR file has.

This is post-hoc statistics over existing metadata: it flags *candidates*,
not verdicts. See BAUPLAN.md and README.md ("The white rabbit") for why a
hard "this particle belongs in class Y" claim needs a targeted RELION
rerun, not just a metadata score.
"""

import numpy as np
import pandas as pd


def per_class_stats(df, metric_col, class_col="rlnClassNumber"):
    """Mean/median/std/min/max/n of `metric_col`, grouped by `class_col`.

    Drops NaN values per class (mirrors plot_lcc_per_class.py's skipped_nan
    handling). Returns a DataFrame indexed by class number.
    """
    if metric_col not in df.columns:
        raise ValueError("per_class_stats: column %r not in DataFrame" % metric_col)
    if class_col not in df.columns:
        raise ValueError("per_class_stats: column %r not in DataFrame" % class_col)

    values = pd.to_numeric(df[metric_col], errors="coerce")
    grouped = values.groupby(df[class_col])
    stats = grouped.agg(n="count", mean="mean", median="median", std="std", min="min", max="max")
    return stats.sort_index()


def flag_outliers(df, metric_col, class_col="rlnClassNumber", z_threshold=2.5, low_is_bad=True):
    """Flag particles whose `metric_col` is a within-class outlier.

    Z-score computed per class (not project-wide) -- an outlier is judged
    against its own class's distribution, since classes can have
    genuinely different metric distributions (e.g. one junk class).

    `low_is_bad`: True for metrics where low = bad (rlnMaxValueProbDistribution,
    rlnCtfMaxResolution when it's a resolution *value* -- careful, lower
    resolution number is *better*, check the metric's convention before
    calling). False for metrics where high = bad (rlnNrOfSignificantSamples).

    Returns a boolean Series aligned to `df.index`: True = flagged as a
    QC outlier candidate worth checking.
    """
    values = pd.to_numeric(df[metric_col], errors="coerce")
    class_mean = values.groupby(df[class_col]).transform("mean")
    class_std = values.groupby(df[class_col]).transform("std").replace(0, np.nan)
    z = (values - class_mean) / class_std

    if low_is_bad:
        flagged = z <= -z_threshold
    else:
        flagged = z >= z_threshold
    return flagged.fillna(False)


def flag_class_outliers(df, metric_col, class_col="rlnClassNumber", z_threshold=2.5):
    """Direction-agnostic per-class outlier flag: which particles are
    furthest from their own class's typical value for `metric_col`, in
    either direction. Robust (MAD-based, like ctf.py/orientation.py's
    outlier flags) rather than `flag_outliers`'s std-based z-score, and
    deliberately doesn't assume which direction is "bad" -- useful for
    driving a generic particle browser over whatever metric the caller
    picks (CTF resolution, defocus, alignment confidence, log-likelihood)
    without having to know each metric's convention up front, unlike
    `flag_outliers` which requires `low_is_bad`.

    Returns a boolean Series aligned to `df.index`.
    """
    values = pd.to_numeric(df[metric_col], errors="coerce")
    median = values.groupby(df[class_col]).transform("median")
    mad = (values - median).abs().groupby(df[class_col]).transform("median").replace(0, np.nan)
    robust_z = (values - median).abs() / (1.4826 * mad)
    return (robust_z >= z_threshold).fillna(False)


def candidate_misclassified(
    df,
    class_col="rlnClassNumber",
    confidence_col="rlnMaxValueProbDistribution",
    good_classes=None,
    bad_classes=None,
    confidence_threshold=None,
):
    """Rank particles as "lost good particle" / "smuggled bad particle"
    candidates -- NOT a verdict, see module docstring.

    A particle in a `bad_classes` class with high alignment confidence is
    a candidate for "should probably be in a good class". A particle in a
    `good_classes` class with low alignment confidence is a candidate for
    "dragging the average down, check it".

    If `good_classes`/`bad_classes` aren't given, splits classes by
    particle count: the largest class(es) are "good", the rest "bad" --
    a rough default, override with domain knowledge (e.g. from 3D class
    inspection) when available.

    Returns df with an added `_qc_candidate` column: "lost_good",
    "smuggled_bad", or "" (not flagged).
    """
    confidence = pd.to_numeric(df[confidence_col], errors="coerce")
    if confidence_threshold is None:
        confidence_threshold = confidence.median()

    if good_classes is None or bad_classes is None:
        counts = df[class_col].value_counts()
        median_count = counts.median()
        good_classes = set(counts[counts >= median_count].index)
        bad_classes = set(counts[counts < median_count].index)

    in_good = df[class_col].isin(good_classes)
    in_bad = df[class_col].isin(bad_classes)

    result = df.copy()
    result["_qc_candidate"] = ""
    result.loc[in_good & (confidence < confidence_threshold), "_qc_candidate"] = "in_good_class_low_confidence"
    result.loc[in_bad & (confidence >= confidence_threshold), "_qc_candidate"] = "in_bad_class_high_confidence"
    return result


def classification_purity_fraction(df, class_col="rlnClassNumber", confidence_col="rlnMaxValueProbDistribution", **kwargs):
    """Fraction of particles flagged by `candidate_misclassified` (0-1)."""
    flagged = candidate_misclassified(df, class_col=class_col, confidence_col=confidence_col, **kwargs)["_qc_candidate"]
    return (flagged != "").sum() / len(flagged)
