# relion-origin cheatsheet

Copy-paste commands for the most common tasks.

## Setup

```bash
# from a checkout, no install needed (reliontree checkout must be a sibling dir)
cd relion-origin
export PYTHONPATH=src:/path/to/reliontree/src

# or, properly installed (pulls reliontree along):
pip install "git+https://github.com/LUKASinScience/relion-origin"
```

## QC GUI (per-class average + its standout particles)

```bash
# single STAR file
python3 -m relion_origin.cli /path/to/particles.star

# or a whole RELION project directory -- gets a job picker in the topbar,
# and class-average thumbnails read straight from the job's own MRC output
python3 -m relion_origin.cli /path/to/relion_project --job Refine3D/job006

# HPC login node -- no browser, print the URL instead
python3 -m relion_origin.cli /path/to/particles.star --no-open
# then from your own machine:
ssh -L 8711:127.0.0.1:8711 user@cluster
# open http://127.0.0.1:8711 in your own browser

# a different class column (default: rlnClassNumber), or a different port
python3 -m relion_origin.cli /path/to/particles.star --class-column rlnClassNumber --port 9000

# STA: pass --tomograms-star to also surface CTF-consistency, dose-gap, and
# tomogram-edge-distance signals in the Overview card, plus the worst-case
# defocus hint, and attach tomogram-level defocus as a usable class metric
python3 -m relion_origin.cli /path/to/particles.star --tomograms-star particles_..._tomograms.star
```

The page has two parts:

- **Overview**: every project-wide QC dimension relion-origin can compute --
  one row per signal, a badge (green = nothing flagged, yellow/red = a
  growing fraction flagged), a one-line explanation. Signals it can't
  compute (e.g. CTF consistency without `--tomograms-star`) just don't
  appear.
- **Classes**: one collapsible section per class -- its average's
  thumbnail, its particle count, and the particles that stand out within
  it. No metric picker: relion-origin automatically checks up to three
  QC metrics that actually exist on the file (CTF resolution, defocus,
  alignment confidence, log-likelihood, in that priority order) and flags
  a particle if it's an outlier on any of them within its own class. One
  link downloads a cleaned STAR file with every flagged particle excluded
  across all classes, ready to use as input for a new RELION job.

## Particle-flow through the pipeline (Python, no CLI yet)

```python
from relion_origin.lineage import particle_flow

df = particle_flow("/path/to/relion_project")  # the dir with default_pipeline.star
print(df)
# columns: job, n_particles, parents, resolution_angstrom, loss_vs_max_parent
```

## Particle provenance across jobs with no stable ID (STA: e.g. pytom -> RELION)

```python
from relion_origin.io import load_particles, load_tomograms_star
from relion_origin.match import match_by_coordinate

source = load_particles("pytom_picks.star")          # e.g. pytom template-matching output
target = load_particles("run_it025_data.star")        # RELION Class3D/Refine3D output
dims, ctf_by_tomo = load_tomograms_star("particles_..._tomograms.star")

result = match_by_coordinate(source, target, tolerance=5.0, tomogram_dims=dims)
print((result["_match_source_index"] >= 0).mean(), "fraction matched")
# tomogram_dims=dims is STA-only (WarpTools/pytom coordinate centering fix) --
# leave it out for SPA data.
```

## STA per-tomogram CTF/defocus QC

```python
from relion_origin.io import load_tomograms_star
from relion_origin.ctf import tomogram_defocus_summary, attach_tomogram_ctf

dims, ctf_by_tomo = load_tomograms_star("particles_..._tomograms.star")
print(tomogram_defocus_summary(ctf_by_tomo))       # one row per tomogram

particles = load_particles("run_it025_data.star")
enriched = attach_tomogram_ctf(particles, ctf_by_tomo, name_col="rlnMicrographName")
# adds _tomoDefocusMean / _tomoDefocusStd / _tomoAstigmatismMean per particle
```

## CTF-consistency flagging (the literature-backed lever)

Flags tomograms/tilts whose defocus is erratic rather than just reporting
mean/spread -- NovaCTF showed CTF quality per tilt is a real, measured STA
resolution lever (up to 3.4 A in their example):

```python
from relion_origin.ctf import tomogram_defocus_summary, flag_inconsistent_tomograms, flag_inconsistent_tilts

summary = tomogram_defocus_summary(ctf_by_tomo)
flagged = flag_inconsistent_tomograms(summary)       # adds _ctf_inconsistent per tomogram
print(flagged[flagged["_ctf_inconsistent"]])

tilt_flags = flag_inconsistent_tilts(ctf_by_tomo)     # dict: tomogram -> bool Series per tilt
```

## Aliasing-free box size (from your project's real worst-case defocus)

```bash
# STA -- reads worst-case defocus from a *_tomograms.star
python3 -m relion_origin.box_size_cli --kv 300 --cs 2.7 --pixel-size 11.056 --resolution 20.0 \
    --tomograms-star particles_..._tomograms.star

# SPA -- reads it from a particles STAR's rlnDefocusU/V
python3 -m relion_origin.box_size_cli --kv 300 --cs 2.7 --pixel-size 1.06 --resolution 3.5 \
    --particles-star particles.star

# or skip project data and just plug in a defocus value directly
python3 -m relion_origin.box_size_cli --kv 300 --cs 2.7 --pixel-size 1.06 --resolution 3.5 --defocus 30000
```

```python
from relion_origin.box_size import recommended_box_size_from_ctf

recommended_box_size_from_ctf(kv=300, cs_mm=2.7, pixel_size_angstrom=11.056,
                               resolution_angstrom=20.0, ctf_by_tomogram=ctf_by_tomo)
# -> {"box_px_fft": ..., "worst_case_defocus_angstrom": ..., ...}
```

## STA edge-distance QC (particles near a tomogram boundary)

```python
from relion_origin.spatial import edge_distance_sta, flag_near_edge

enriched = edge_distance_sta(particles, dims)   # dims from load_tomograms_star
flagged = flag_near_edge(enriched)              # relative to the project's own median distance
print(flagged.sum(), "particles near a tomogram edge")
```

## SPA edge-distance QC (particles near a micrograph boundary -- a different problem than the STA one above)

```python
from relion_origin.spatial import edge_distance_spa, flag_near_edge

# micrograph_dims: a DataFrame indexed by rlnMicrographName with SizeX/SizeY
# in pixels -- you supply it (e.g. from .mrc headers), RELION's particle
# STAR doesn't carry micrograph dimensions itself.
enriched = edge_distance_spa(particles, micrograph_dims)
flagged = flag_near_edge(enriched)   # same function as the STA version, same column name
print(flagged.sum(), "particles near a micrograph edge (fringe/barcode artifact risk)")
```

## Particle-density outliers (SPA micrographs or STA tomograms)

```python
from relion_origin.spatial import particle_density_outliers

density = particle_density_outliers(particles)  # works on either via io's name-column aliasing
print(density[density["_density_outlier"]])      # picking failure / contamination / empty square
```

## Angular-coverage QC (missing-wedge / preferred-orientation flag)

```python
from relion_origin.orientation import angular_coverage_score, flag_clustered_orientation

print(angular_coverage_score(particles))          # R per class, 0=isotropic, 1=fully clustered
print(flag_clustered_orientation(particles))       # project-relative outliers, not an absolute cutoff
```

## Dose-gap QC (missing/duplicated tilt images, STA)

```python
from relion_origin.ctf import flag_dose_gaps

dose_flags = flag_dose_gaps(ctf_by_tomo)   # dict: tomogram -> bool Series per tilt
```

## Optics-group QC (facility/multi-session projects) -- no new code needed

Every `qc.py` function takes `class_col` -- point it at `rlnOpticsGroup`
instead of `rlnClassNumber`:

```python
from relion_origin.qc import per_class_stats
per_class_stats(df, "rlnCtfMaxResolution", class_col="rlnOpticsGroup")
```

```bash
relion-origin particles.star --class-column rlnOpticsGroup
```

## Resolution before/after (the actual evidence, not a claim)

Once you've run a targeted rerun on a `relion-origin-rerun` candidate
subset, compare its resolution against the original job:

```bash
python3 -m relion_origin.compare_cli /path/to/relion_project Refine3D/job006 Refine3D/job010
# Refine3D/job006: 23.00 A
# Refine3D/job010: 21.50 A
# delta: 1.50 A better
```

```python
from relion_origin.compare import compare_resolution
compare_resolution("/path/to/relion_project", "Refine3D/job006", "Refine3D/job010")
```

## Draft a methods-section paragraph

Pulls together whatever QC results are available (candidates, CTF
consistency, resolution comparison) into a short draft -- edit before
using, not citation-ready as-is:

```bash
python3 -m relion_origin.methods_cli particles.star \
    --tomograms-star particles_..._tomograms.star \
    --project-dir /path/to/relion_project \
    --job-before Refine3D/job006 --job-after Refine3D/job010
```

All the extra flags are optional -- `--tomograms-star` is STA-only, and
`--project-dir`/`--job-before`/`--job-after` only apply once you've
actually run a rerun to compare. With just `particles.star`, you get a
sentence for the candidate ranking alone.

```python
from relion_origin.methods import draft_methods_paragraph
draft_methods_paragraph(candidates_df=candidates, ctf_summary=ctf_summary,
                         resolution_comparison=comparison,
                         job_before="Refine3D/job006", job_after="Refine3D/job010")
```

## Per-class QC scoring directly (what the GUI does under the hood)

```python
from relion_origin.io import load_particles
from relion_origin.qc import per_class_stats, flag_outliers, candidate_misclassified

df = load_particles("particles.star")
print(per_class_stats(df, "rlnMaxValueProbDistribution"))

outliers = flag_outliers(df, "rlnCtfMaxResolution", low_is_bad=False)  # high resolution number = worse CTF
candidates = candidate_misclassified(df)   # adds a "_qc_candidate" column, see README "The white rabbit"
```

## Turn QC candidates into a rerun (close the "candidate, not verdict" gap)

relion-origin can only flag *candidates* (see README "The white rabbit") --
confirming one needs a real, targeted RELION job on exactly that subset.
This writes that subset as a STAR file RELION can load directly:

```bash
# both candidate labels (lost-good + smuggled-bad) in one file
python3 -m relion_origin.rerun_cli particles.star candidates.star

# just one label, e.g. only the "should probably be in a better class" ones
python3 -m relion_origin.rerun_cli particles.star candidates.star \
    --label in_bad_class_high_confidence
```

Then point a new Class3D/Refine3D/Extract job's "Input images STAR file" at
`candidates.star` in RELION for the actual verification rerun.

```python
# same thing from Python, e.g. to build the subset from your own filtering
from relion_origin.qc import candidate_misclassified
from relion_origin.rerun import write_candidate_subset

candidates = candidate_misclassified(df)
write_candidate_subset(candidates, "candidates.star")
```

## Using the installed console scripts instead of `python -m`

The install (see Setup above) also installs these directly, so you
can drop the `python3 -m relion_origin.` prefix everywhere above:

```bash
relion-origin particles.star
relion-origin-rerun particles.star candidates.star
relion-origin-compare /path/to/relion_project Refine3D/job006 Refine3D/job010
relion-origin-methods particles.star --project-dir ... --job-before ... --job-after ...
relion-origin-boxsize --kv 300 --cs 2.7 --pixel-size 11.056 --resolution 20.0 --tomograms-star ...
```
