"""relion-origin -- per-particle provenance, lineage and QC analysis for
RELION projects, the STAR-table-heavy features deliberately kept out of
reliontree's dependency-free core.

Usable as a library, not just the `relion-origin`/`relion-origin-*` CLIs --
same pattern as `reliontree`:

    import relion_origin

    df = relion_origin.load_particles("particles.star")
    flags = relion_origin.flag_class_outliers(df, "rlnDefocusU")
    candidates = relion_origin.candidate_misclassified(df)

See README.md for the full command-line tools and the build plan for the
per-module design decisions.
"""

from .annotations import annotate_particles, annotate_project, write_annotations
from .box_size import aliasing_free_box_size, recommended_box_size_from_ctf, worst_case_defocus_spa, worst_case_defocus_sta
from .compare import compare_resolution
from .ctf import attach_tomogram_ctf, flag_dose_gaps, flag_inconsistent_tilts, flag_inconsistent_tomograms, tomogram_defocus_summary
from .diagnose import diagnose_class, diagnose_class_from_flags
from .io import load_particles, load_tomograms_star, resolve_column
from .focused import class_occupancy_trajectory, collapsed_classes, offset_summary, rank_class_differences, read_mrc
from .iteration_convergence import iteration_convergence, stable_since_iteration
from .lineage import list_jobs_with_particles, particle_flow
from .match import match_by_coordinate
from .methods import draft_methods_paragraph
from .orientation import angular_coverage_score, flag_clustered_orientation
from .qc import candidate_misclassified, classification_mixing_excess, classification_purity_fraction, flag_class_outliers, flag_outliers, per_class_stats
from .reference import class_reference_similarity, map_similarity
from .rerun import write_candidate_subset
from .spatial import edge_distance_spa, edge_distance_sta, flag_near_edge, particle_density_outliers

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "annotate_particles",
    "annotate_project",
    "write_annotations",
    "load_particles",
    "load_tomograms_star",
    "resolve_column",
    "match_by_coordinate",
    "tomogram_defocus_summary",
    "flag_inconsistent_tomograms",
    "flag_inconsistent_tilts",
    "flag_dose_gaps",
    "attach_tomogram_ctf",
    "per_class_stats",
    "flag_outliers",
    "flag_class_outliers",
    "candidate_misclassified",
    "classification_purity_fraction",
    "classification_mixing_excess",
    "class_reference_similarity",
    "map_similarity",
    "diagnose_class",
    "diagnose_class_from_flags",
    "iteration_convergence",
    "stable_since_iteration",
    "particle_flow",
    "list_jobs_with_particles",
    "write_candidate_subset",
    "compare_resolution",
    "draft_methods_paragraph",
    "offset_summary",
    "rank_class_differences",
    "class_occupancy_trajectory",
    "collapsed_classes",
    "read_mrc",
    "aliasing_free_box_size",
    "worst_case_defocus_sta",
    "worst_case_defocus_spa",
    "recommended_box_size_from_ctf",
    "edge_distance_sta",
    "edge_distance_spa",
    "flag_near_edge",
    "particle_density_outliers",
    "angular_coverage_score",
    "flag_clustered_orientation",
]
