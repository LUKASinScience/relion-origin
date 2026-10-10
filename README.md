# relion-origin

**Status: early working version, not released/packaged to PyPI yet.**
Installs cleanly from a checkout (`pip install -e .`), all automated tests
pass, and every command + the GUI has been verified end to end — but only
against one real project so far. See "Testing" below for how to try it
against your own project, STA or SPA — that's the actual next step before
trusting the numbers generally.

A separate, optional companion to
[reliontree](https://github.com/LUKASinScience/reliontree) for the analyses
that need real per-particle STAR-table parsing — things reliontree
deliberately does *not* do, because reliontree stays dependency-free (pure
Python stdlib) so it always runs on a bare HPC login node with nothing
installed. This project accepts real dependencies (`starfile`, `numpy`,
`polars`/`duckdb`) in exchange for doing the heavier per-particle analysis
reliontree's core can't.

## The white rabbit

reliontree answers "what happened" — it's the mouse chasing the cheese
through the job tree, following the pipeline graph from raw movies to the
final map, one job at a time.

relion-origin asks a different question: inside the average RELION hands
you at the end, which particles actually earned their place? RELION's
Bayesian classification is a vote, not a verdict — it optimizes a
population-level likelihood, not a certificate for every single particle.
Some particles that ended up in your "good" class have a defocus so far off
or an alignment so ambiguous (low `rlnMaxValueProbDistribution`, high
`rlnNrOfSignificantSamples`) that they're dragging the average down. Others
got voted into the "bad" class by a marginal likelihood difference despite
having a sharp, well-aligned signal — good particles lost in the noise of
a population-level decision.

Chasing those down — the true particles hiding behind an average, as
opposed to the ones that just happened to get counted — is chasing the
white rabbit instead of the cheese. Different animal, different chase,
same burrow (the same STAR files, the same pipeline).

## Why a separate project, not a reliontree feature

reliontree's whole value proposition is "runs anywhere, zero setup." The
features below need to read STAR tables with millions of rows (particle
sets) and join them across jobs by coordinate, not by a stable ID — that's
squarely `starfile`/`polars`/`duckdb` territory. Reinventing that inside
reliontree would either duplicate a well-maintained library or force every
reliontree user to accept a dependency they don't need for the (much more
common) job-tree-viewing use case.

## Current scope

What's built and working (`src/relion_origin/`):

- **`io.py`** — load a STAR file's particle table (`starfile` under the
  hood), with SPA/STA column aliasing (`rlnCoordinateX` vs.
  `rlnCenteredCoordinateXAngst`, `rlnMicrographName` vs. `rlnTomoName`).
  Also loads a RELION5-STA `*_tomograms.star` (tomogram dimensions +
  per-tilt CTF blocks) and converts RELION's corner-origin pixel
  coordinates to a tomogram-centered Angstrom frame.
- **`match.py`** — particle provenance where no stable ID exists: matches
  particles across STAR files by nearest 3D coordinate (+
  tomogram/micrograph name) within a tolerance, generalizing the technique
  the test dataset's own `transfer_lcc_values.py` used by hand. For STA
  data this needs the coordinate-centering correction from `io.py` (RELION
  and pytom/WarpTools use different coordinate origins) — verified against
  real data.
- **`ctf.py`** — STA-only: per-tomogram defocus/astigmatism summary from
  `io.load_tomograms_star`'s per-tilt CTF blocks, joined onto particles.
  Deliberately *not* a 1:1 reproduction of RELION's own per-particle
  geometric defocus (that formula lives only in RELION's C++ source) — it reports a simpler per-tomogram (not per-particle) defocus summary instead. Also flags
  tomograms/tilts with erratic defocus (`flag_inconsistent_tomograms`,
  `flag_inconsistent_tilts`) — the one lever with a published Angstrom
  number behind it (NovaCTF, up to 3.4 A).
- **`qc.py`** — per-class metric statistics, within-class outlier flagging
  (CTF/defocus/alignment-confidence), and "lost good particle" /
  "smuggled bad particle" candidate ranking — post-hoc, from existing STAR
  metadata, no RELION rerun needed. A candidate list to check, not a
  verdict: RELION doesn't store full multi-class likelihoods per particle,
  so confirming a candidate needs a targeted RELION rerun on the flagged
  subset.
- **`lineage.py`** — particle-flow tracking through the pipeline, built on
  `reliontree`'s job-DAG instead of re-parsing `default_pipeline.star`:
  particle counts per job, parent jobs, and loss versus the largest
  parent. Verified against the test project's own job outputs (see
  `tests/test_lineage.py`).
- **`rerun.py`/`rerun_cli.py`** (`relion-origin-rerun`) — writes a QC
  candidate list as a RELION-loadable particle subset STAR file, so a
  candidate is one command away from the targeted rerun that actually
  confirms it, instead of manual STAR editing. See CHEATSHEET.md.
- **`compare.py`/`compare_cli.py`** (`relion-origin-compare`) — resolution
  before/after between two jobs (e.g. the original Refine3D vs. a rerun on
  a QC-filtered subset), via reliontree's own resolution parsing. This is
  how to actually generate the Angstrom evidence the literature is
  missing, instead of asserting it.
- **`methods.py`/`methods_cli.py`** (`relion-origin-methods`) — drafts a
  methods-section paragraph from whatever QC results are available
  (candidate counts, CTF-consistency flags, resolution comparison) —
  mirrors reliontree's own `methods` command. Not just convenience: a tool
  getting named in a paper's methods section is how a niche tool like this
  actually spreads.
- **`box_size.py`/`box_size_cli.py`** (`relion-origin-boxsize`) — aliasing-
  free particle-extraction box size (Warp/M's CTF-delocalization approach,
  ported from the author's own JavaScript calculator), sized against the
  project's real worst-case defocus from `ctf.py`/particle CTF data
  instead of a guessed value — a box sized for the mean defocus aliases
  exactly the highest-defocus tomograms/micrographs. Verified bit-for-bit
  against the original JS calculator.
- **`spatial.py`** — STA edge-distance QC (`edge_distance_sta`/
  `flag_near_edge`: particles near a tomogram's Z boundary get worse
  signal from thinner effective ice / harsher missing-wedge truncation
  there), an SPA counterpart (`edge_distance_spa`: particles near a
  micrograph's own image edge pick up extraction "fringe"/"barcode"
  interpolation artifacts — a distinct, community-confirmed failure mode), and particle-density-per-source outlier detection
  (`particle_density_outliers`, works for SPA micrographs or STA
  tomograms identically — flags a picking failure/contamination/empty
  grid square).
- **`orientation.py`** — angular-coverage QC (`angular_coverage_score`,
  standard spherical/directional statistics, Fisher et al. 1987): how
  clustered a class's viewing directions are, a missing-wedge/preferred-
  orientation red flag. Flags project-relative outliers, not an absolute
  cutoff — STA classes commonly sit at R=0.7-0.9 even when nothing's
  wrong, an absolute threshold would flag everything
- **`ctf.flag_dose_gaps()`** — flags tilt images whose cumulative-dose step
  doesn't look like a whole multiple of the tilt series' own per-tilt
  dose (missing/duplicated tilt proxy). Works on the *absolute* step size,
  not assuming dose increases monotonically down the block — RELION5's
  `tomograms.star` orders tilts by geometry, and a dose-symmetric scheme's
  dose genuinely falls then rises within one tomogram's block, confirmed
  on the test dataset
- **Optics-group QC is already free**: every `qc.py` function takes a
  `class_col` argument — `class_col="rlnOpticsGroup"` works today, no new
  code, including in the GUI (`--class-column`). Useful for
  multi-grid/multi-session facility projects.
- **`server.py`/`cli.py`** — the QC GUI (see "GUI" below).

Everything above works generically on any RELION STAR file with a class
column and numeric metric columns — it was built STA-first (that's the
only real project available while developing), but nothing in `io.py`,
`qc.py`, or `server.py` is STA-specific. Only `ctf.py` and the coordinate
centering in `match.py`/`io.py` are.

## Explicitly not planned

No STAR editor, no RELION GUI replacement, no particle picker, no
heterogeneity algorithm, no tomography framework, and no neural
classifier — see [ANTIDOTE](https://github.com/herzik-lab/ANTIDOTE) for
that (metadata-driven NN for 3D particle sorting; complementary to this
tool, not something to rebuild here).

## Relationship to reliontree

Standalone package, own `pyproject.toml`, own dependencies — not a
reliontree extra. Imports `reliontree` for pipeline-graph parsing
(`build_tree`, `find_project`) instead of re-parsing `default_pipeline.star`.

## GUI

Same pattern as reliontree, on purpose: a stdlib `http.server` only,
binding to `127.0.0.1`, no CDN scripts. That's what makes it work
identically on a local Mac, a workstation, and an HPC login node with no
browser and no open inbound port — on the login node, tunnel it from your
own machine:

```bash
ssh -L 8711:127.0.0.1:8711 user@cluster
```

then open `http://127.0.0.1:8711` locally. Nothing else needs installing
on the remote side beyond the Python dependencies (no X11, no separate
server process, no internet access required on the cluster). Point it at
a whole RELION project directory instead of a single STAR file to get a
job picker in the topbar (switches which job's particles you're looking
at without restarting the server).

Two things on the page:

- **Overview**: every project-wide QC dimension relion-origin can compute
  for the given file(s) — angular coverage, and (with `--tomograms-star`)
  CTF consistency, dose gaps, tomogram-edge distance, and a worst-case
  defocus hint for the box-size calculator — one row per signal, a
  color-coded badge, a one-line explanation.
- **Classes**: the actual point of the tool, and the actual unit a user
  thinks in — not a metric picker. Every class gets its own section: its
  average's thumbnail, its particle count, and the particles that stand
  out within it, already flagged (CTF resolution, defocus, alignment
  confidence, log-likelihood — whichever of those exist, no configuring
  required). One link downloads a cleaned STAR file with every flagged
  particle excluded, across all classes.

## Annotating the reliontree job tree

`relion-origin-annotate <project>` writes orientation findings (viewing-direction
clustering per class, angular coverage of a refinement) to
`<project>/.reliontree/annotations/relion-origin.json`; reliontree (>=0.2.0 plus
the annotations reader) shows them as pills on the job cards. The QC-candidate
share is deliberately not annotated: with the default median confidence threshold
it flags about half of all particles of any classification.

## Install

`reliontree` (>= 0.3.0) is a dependency and is not on PyPI yet, so `pip install -e .` alone fails with
`No matching distribution found for reliontree`. Install it first, from its checkout or from the tagged release:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "git+https://github.com/LUKASinScience/reliontree@v0.3.0"     # or: pip install -e ../reliontree
pip install -e .                                                         # relion-origin itself (+ starfile, numpy, pandas)
python -c "import relion_origin; print(relion_origin.__version__)"       # it is an importable module, not only a CLI
```

reliontree does not depend on relion-origin and never imports it: it works alone. relion-origin only hands it findings
through a file (`<project>/.reliontree/annotations/relion-origin.json`, written by `relion-origin-annotate <project>`);
reliontree shows them as pills on the job cards when the file exists.

See `CHEATSHEET.md` for copy-paste commands (GUI, particle-flow tracking,
provenance matching, CTF QC, rerun helper, methods draft). Short
version:

```bash
cd relion-origin
pip install -e .                       # installs starfile + reliontree (>=0.3.0) + the console scripts
relion-origin /path/to/a/particles.star            # opens the QC GUI
relion-origin-rerun particles.star candidates.star # write a QC candidate subset
relion-origin-compare /path/to/project JobA JobB   # resolution before/after
relion-origin-methods particles.star               # draft a methods paragraph
relion-origin-annotate /path/to/project            # write orientation/QC findings for reliontree's job cards
```

or, without installing, straight from a checkout (see CHEATSHEET.md for the
full `PYTHONPATH` form).

## Credits and third-party material

- No third-party source code is bundled. Runtime dependencies keep their own licenses: starfile (BSD-3-Clause), numpy and
  pandas (BSD-3-Clause), all compatible with AGPL-3.0-or-later; reliontree is AGPL-3.0-or-later.
- The box-size calculator follows the CTF-delocalization approach of Warp/M (Tegunov et al.); the angular-sampling and
  regularisation heuristics are the textbook Crowther criterion and RELION's published guidance.
- Reading RELION's STAR/`.bild` files and calling its programs is plain use of RELION's documented file formats (RELION is GPL).
- Prior art is cited where relevant (RELION-Convergence-Inspector, ANTIDOTE); nothing was copied from it.

## License

AGPL-3.0-or-later, the same as [reliontree](https://github.com/LUKASinScience/reliontree), which it imports.
