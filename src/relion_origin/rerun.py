"""Turn a QC candidate list into a RELION-loadable particle subset STAR
file, so a candidate can actually be checked with a targeted rerun instead
of staying a number in a report.

Closes the gap the rest of this tool is explicit about (README "The white
rabbit", qc.py docstring): relion-origin can only produce *candidates*,
not verdicts, because RELION doesn't store a full multi-class likelihood
per particle. Verifying a candidate needs a real RELION job on exactly
that subset. This module makes that one command instead of manual STAR
editing -- point a new Class3D/Refine3D/Extract job's "Input images STAR
file" at the file this writes.
"""

import starfile

# Bookkeeping columns relion-origin adds during analysis (match.py, qc.py)
# -- never RELION's own vocabulary, must not leak into a file RELION will
# read, since a leading-underscore column isn't a valid rlnXxx field and
# RELION programs may warn or choke on it.
_INTERNAL_COLUMN_PREFIXES = ("_match_", "_qc_", "_tomo")


def _drop_internal_columns(df):
    keep = [c for c in df.columns if not c.startswith(_INTERNAL_COLUMN_PREFIXES)]
    return df[keep]


def write_candidate_subset(df, output_path, candidate_col="_qc_candidate", candidate_values=None):
    """Write the particles flagged by `candidate_col` (default: every
    non-empty `_qc_candidate` label from `qc.candidate_misclassified`) to
    `output_path` as a RELION-compatible particles STAR file.

    `candidate_values`: restrict to specific label(s), e.g.
    `["in_bad_class_high_confidence"]` to check only the "smuggled bad
    particle" candidates. Defaults to every flagged (non-empty) row.

    Returns the number of particles written. Raises ValueError if that's
    zero -- an empty STAR file isn't a useful rerun input and usually
    means the candidate_col/candidate_values don't match the data.
    """
    if candidate_col not in df.columns:
        raise ValueError(
            "write_candidate_subset: column %r not in df -- run qc.candidate_misclassified() "
            "first, or pass candidate_col= the column you flagged particles in" % candidate_col
        )

    if candidate_values is None:
        selected = df[df[candidate_col] != ""]
    else:
        selected = df[df[candidate_col].isin(candidate_values)]

    if len(selected) == 0:
        raise ValueError(
            "write_candidate_subset: 0 particles matched %r in %r -- nothing to write"
            % (candidate_values or "<any non-empty label>", candidate_col)
        )

    starfile.write(_drop_internal_columns(selected), output_path, overwrite=True)
    return len(selected)
