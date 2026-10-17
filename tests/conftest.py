"""Tests that need a real RELION project take its location from the environment and are skipped without it.

RELION_ORIGIN_TEST_PROJECT          a RELION STA project directory (default_pipeline.star); used by the compare, lineage and
                                    iteration-convergence tests, which expect the job ids and numbers of one particular dataset
RELION_ORIGIN_TEST_MATCH_DIR        directory with the source particle table and run_it025_data.star (match-centering test)
RELION_ORIGIN_TEST_MATCH_STAR       file name of that source table inside the directory (default: match_source.star)
RELION_ORIGIN_TEST_TOMOGRAMS_STAR   the matching *_tomograms.star
"""

import os
from pathlib import Path


def env_path(name):
    """Path from the environment variable `name`, or None when it is unset or empty."""
    value = os.environ.get(name, "").strip()
    return Path(value) if value else None
