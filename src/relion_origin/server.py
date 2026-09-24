"""Local QC-GUI, same pattern as reliontree: stdlib http.server only, binds
127.0.0.1. That's the deliberate choice that makes it work everywhere with
one command and no setup:

- **Local Mac/workstation**: opens straight in the default browser.
- **Workstation on the LAN**: same, browser and server on the same machine.
- **HPC login node** (no browser, often no open inbound port): the server
  still binds 127.0.0.1 and does nothing else network-wise; reach it with
  an SSH local-port-forward from your own machine --
  `ssh -L 8711:127.0.0.1:8711 user@cluster` -- then open
  http://127.0.0.1:8711 in your own browser. No CDN scripts, no external
  assets: every image is server-rendered inline, so it also works on an
  air-gapped cluster with no outbound internet.

Design intent (BAUPLAN.md ##21, ##22): the class is the unit, not the
metric. A user looks at one average -- shown large, not a 32px icon --
sees the particles that make it up, paginated instead of truncated, and
sees which QC metrics matter with a red/yellow/green read at a glance,
not just a raw flagged-count.
"""

import tempfile
import webbrowser
from html import escape as _esc
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

import pandas as pd
import starfile

from .ctf import attach_tomogram_ctf, flag_dose_gaps, flag_inconsistent_tomograms, tomogram_defocus_summary
from .diagnose import diagnose_class_from_flags
from .orientation import flag_clustered_orientation
from .qc import flag_class_outliers
from .spatial import edge_distance_sta, flag_near_edge
from .theme import THEME_CSS, badge, metric as metric_tile

DEFAULT_PORT = 8711
Z_THRESHOLD = 2.5
PAGE_SIZE = 25
THUMB_SIZE = 220

# The metrics a user actually cares about for "does this particle belong
# in this average", in priority order -- CTF/defocus first (what BAUPLAN
# ##1 identifies as the clearest, most literature-backed lever), then
# alignment quality. Whichever of these exist on the loaded particles get
# used; nothing else does, deliberately, so the page never turns into a
# dropdown of forty columns nobody asked for.
_QC_METRIC_PRIORITY = [
    "rlnCtfMaxResolution",
    "_tomoDefocusMean",
    "rlnDefocusU",
    "rlnMaxValueProbDistribution",
    "rlnLogLikeliContribution",
]
_METRIC_LABELS = {
    "rlnCtfMaxResolution": "CTF resolution",
    "_tomoDefocusMean": "tomogram defocus",
    "rlnDefocusU": "defocus",
    "rlnMaxValueProbDistribution": "alignment confidence",
    "rlnLogLikeliContribution": "log-likelihood",
}


def _default_qc_metrics(df, max_metrics=3):
    return [m for m in _QC_METRIC_PRIORITY if m in df.columns][:max_metrics]


def _identifier_column(df):
    """The best available per-particle identifier for display -- rlnImageName
    (stack index + file, stable within a RELION job chain) if present,
    else a coordinate-based fallback."""
    for candidate in ("rlnImageName", "rlnMicrographName", "rlnTomoName"):
        if candidate in df.columns:
            return candidate
    return None


def _class_thumbnails(job_dir):
    if job_dir is None:
        return {}
    try:
        from .class_images import class_average_thumbnails

        return class_average_thumbnails(job_dir, size=THUMB_SIZE)
    except Exception:
        return {}  # no reliontree, no MRC files, or a decode failure -- the page still works without images


def _query_href(query, **overrides):
    """Build `?...` preserving every current query param except the ones
    being overridden -- so a page-forward link for one class doesn't reset
    the job picker, other classes' pagination, etc."""
    merged = {k: v[0] for k, v in query.items() if v}
    for k, v in overrides.items():
        if v is None:
            merged.pop(k, None)
        else:
            merged[k] = v
    return "?" + urlencode(merged) if merged else "?"


def _level_for_fraction(fraction):
    """The red/yellow/green read BAUPLAN.md ##22 asks for: green = clean,
    yellow = a minority flagged, red = a large chunk flagged. 5%/20% are
    round, defensible cut points -- not a claim about a physically
    "correct" threshold, just a legible three-tier scale."""
    if fraction <= 0:
        return "good"
    if fraction <= 0.05:
        return "warn"
    return "bad" if fraction > 0.2 else "warn"


_DIAGNOSIS_LEVEL = {
    "heterogeneous": "warn",
    "alignment": "warn",
    "ctf_defocus_limited": "bad",
    "none": "good",
}


def _diagnosis_html(flags_by_metric, group_index, ctf_inconsistent_fraction):
    diagnosis = diagnose_class_from_flags(
        flags_by_metric, group_index, ctf_inconsistent_fraction=ctf_inconsistent_fraction
    )
    if diagnosis["bucket"] == "none":
        return ""
    level = _DIAGNOSIS_LEVEL.get(diagnosis["bucket"], "neutral")
    return '<p class="diagnosis">%s <span class="signal-detail">%s &mdash; %s</span></p>' % (
        badge(diagnosis["bucket"].replace("_", " "), level),
        _esc(diagnosis["reason"]),
        _esc(diagnosis["suggestion"]),
    )


def _class_section_html(df, class_col, cls, metrics, flags_by_metric, id_col, thumbnails, query, ctf_inconsistent_fraction=None):
    """One class: its average image (large -- this is what the user is
    actually judging), particle count, and the particles that stand out
    on at least one QC metric, paginated so "see all of them" doesn't mean
    one giant unreadable page."""
    group_index = df.index[df[class_col] == cls]
    n_total = len(group_index)

    n_flags = pd.Series(0, index=group_index)
    for m in metrics:
        n_flags = n_flags + flags_by_metric[m].reindex(group_index).fillna(False).astype(int)
    standout_index = n_flags[n_flags > 0].sort_values(ascending=False).index
    n_standout = len(standout_index)

    try:
        thumb_key = int(cls)
    except (TypeError, ValueError):
        thumb_key = cls
    image_html = (
        '<img class="class-image" src="%s" alt="class %s average">' % (thumbnails[thumb_key], _esc(str(cls)))
        if thumb_key in thumbnails
        else ""
    )

    if n_standout == 0:
        table_html = "<p>No standouts on %s.</p>" % ", ".join(_METRIC_LABELS.get(m, m) for m in metrics)
    else:
        page_param = "page_%s" % cls
        page = max(1, int(query.get(page_param, ["1"])[0] or 1))
        n_pages = -(-n_standout // PAGE_SIZE)  # ceil
        page = min(page, n_pages)
        start = (page - 1) * PAGE_SIZE
        shown = standout_index[start : start + PAGE_SIZE]

        header = "<tr><th>%s</th><th>flagged for</th>%s</tr>" % (
            _esc(id_col) if id_col else "row",
            "".join("<th>%s</th>" % _esc(_METRIC_LABELS.get(m, m)) for m in metrics),
        )
        rows = [header]
        for idx in shown:
            identifier = _esc(str(df.at[idx, id_col])) if id_col else str(idx)
            flagged_here = [m for m in metrics if bool(flags_by_metric[m].get(idx, False))]
            flag_badges = " ".join(badge(_METRIC_LABELS.get(m, m), "bad") for m in flagged_here)
            value_cells = []
            for m in metrics:
                v = pd.to_numeric(pd.Series([df.at[idx, m]]), errors="coerce").iloc[0]
                cell = "%.4g" % v if pd.notna(v) else "-"
                cls_attr = ' class="cell-bad"' if m in flagged_here else ""
                value_cells.append("<td%s>%s</td>" % (cls_attr, cell))
            rows.append("<tr><td>%s</td><td>%s</td>%s</tr>" % (identifier, flag_badges, "".join(value_cells)))

        pager = ""
        if n_pages > 1:
            prev_link = (
                '<a href="%s">&larr; previous 25</a>' % _query_href(query, **{page_param: str(page - 1)})
                if page > 1
                else '<span class="pager-disabled">&larr; previous 25</span>'
            )
            next_link = (
                '<a href="%s">next 25 &rarr;</a>' % _query_href(query, **{page_param: str(page + 1)})
                if page < n_pages
                else '<span class="pager-disabled">next 25 &rarr;</span>'
            )
            pager = '<div class="pager">%s<span>page %d / %d</span>%s</div>' % (prev_link, page, n_pages, next_link)

        table_html = "<table>%s</table>%s<p class=\"signal-detail\">%d of %d particles stand out</p>" % (
            "".join(rows),
            pager,
            n_standout,
            n_total,
        )

    diagnosis_html = _diagnosis_html(flags_by_metric, group_index, ctf_inconsistent_fraction)

    return (
        '<details%s><summary>class %s &mdash; %d standout / %d particles</summary>'
        '<div class="class-body">%s%s%s</div></details>'
        % (" open" if n_standout > 0 else "", _esc(str(cls)), n_standout, n_total, image_html, diagnosis_html, table_html)
    )


def _signal_row(label, flagged_count, total_count, detail):
    """One row of the overview signals card: a label, a red/yellow/green
    badge, and a one-line explanation. Same shape for every QC dimension
    so the overview reads as one consistent scale instead of ad-hoc
    per-signal formatting."""
    if total_count == 0:
        level, text = "neutral", "n/a"
    else:
        level = _level_for_fraction(flagged_count / total_count)
        text = "%d / %d flagged" % (flagged_count, total_count)
    return (
        '<div class="signal-row"><div class="signal-label">%s</div>%s'
        '<div class="signal-detail">%s</div></div>' % (_esc(label), badge(text, level), _esc(detail))
    )


def _info_row(label, value_text, detail):
    """Same shape as `_signal_row`, for a plain fact rather than a
    flagged-count -- e.g. the worst-case defocus, which isn't itself a
    problem to fix, just a number the box-size calculator needs."""
    return (
        '<div class="signal-row"><div class="signal-label">%s</div>%s'
        '<div class="signal-detail">%s</div></div>' % (_esc(label), badge(value_text, "neutral"), _esc(detail))
    )


def _build_page_cache(df, class_col, ctf_by_tomogram=None, tomogram_dims=None, job_dir=None):
    """Precompute everything that doesn't depend on which page of which
    class the current request asked for -- flags, thumbnails, the
    project-wide overview. Class-section HTML itself is rendered per
    request in `render_page` (it needs the request's pagination state).

    `ctf_by_tomogram`/`tomogram_dims` (from `io.load_tomograms_star`) are
    optional and STA-only. When `ctf_by_tomogram` is given, tomogram-level
    defocus gets attached to every particle here (`_tomoDefocusMean`) so
    it's available as a QC metric, not just a project-wide overview
    number. `job_dir` (the job's own directory) is optional -- when given,
    class-average images are rendered from its MRC output.
    """
    if ctf_by_tomogram is not None:
        df = attach_tomogram_ctf(df, ctf_by_tomogram)

    n_classes = df[class_col].nunique()
    stat_tiles = [metric_tile("%d" % len(df), "particles"), metric_tile("%d" % n_classes, str(class_col))]

    metrics = _default_qc_metrics(df)
    id_col = _identifier_column(df)
    flags_by_metric = {m: flag_class_outliers(df, m, class_col, z_threshold=Z_THRESHOLD) for m in metrics}
    combined_flag = pd.Series(False, index=df.index)
    for f in flags_by_metric.values():
        combined_flag = combined_flag | f
    n_standout_total = int(combined_flag.sum())
    stat_tiles.append(metric_tile("%d" % n_standout_total, "particles stand out"))

    # One overview row PER analyzed parameter with its own red/yellow/green
    # read (BAUPLAN.md ##22) -- not one combined count, so a user can see
    # which specific metric is the problem for this job.
    signal_rows = [
        _signal_row(
            _METRIC_LABELS.get(m, m),
            int(flags_by_metric[m].sum()),
            len(df),
            "particles that are within-class outliers on this metric -- candidates to inspect/remove, not a verdict",
        )
        for m in metrics
    ]

    if {"rlnAngleRot", "rlnAngleTilt"} <= set(df.columns):
        clustered = flag_clustered_orientation(df, class_col=class_col)
        signal_rows.append(
            _signal_row(
                "Angular coverage",
                int(clustered.sum()),
                len(clustered),
                "classes whose viewing-direction coverage is a project-relative outlier (missing-wedge/preferred-orientation risk)",
            )
        )

    ctf_inconsistent_fraction = None
    if ctf_by_tomogram is not None:
        summary = flag_inconsistent_tomograms(tomogram_defocus_summary(ctf_by_tomogram))
        ctf_inconsistent_fraction = float(summary["_ctf_inconsistent"].sum()) / len(summary) if len(summary) else None
        signal_rows.append(
            _signal_row(
                "CTF consistency",
                int(summary["_ctf_inconsistent"].sum()),
                len(summary),
                "tomograms with erratic per-tilt defocus/astigmatism vs. the rest of the project",
            )
        )
        try:
            from .box_size import worst_case_defocus_sta

            worst_defocus = worst_case_defocus_sta(ctf_by_tomogram)
            signal_rows.append(
                _info_row(
                    "Worst-case defocus",
                    "%.1f um" % (worst_defocus / 10000),
                    "size an extraction box against this, not the mean -- "
                    "relion-origin-boxsize --defocus %d ..." % round(worst_defocus),
                )
            )
        except Exception:
            pass  # missing rlnDefocusU/V in a block -- skip the hint, don't break the page
        dose_flags = flag_dose_gaps(ctf_by_tomogram)
        total_tilts = sum(len(v) for v in dose_flags.values())
        flagged_tilts = sum(int(v.sum()) for v in dose_flags.values())
        signal_rows.append(
            _signal_row("Dose gaps", flagged_tilts, total_tilts, "tilt images with a missing/duplicated-tilt-shaped dose step")
        )

    if tomogram_dims is not None:
        try:
            enriched = edge_distance_sta(df, tomogram_dims)
            near_edge = flag_near_edge(enriched)
            signal_rows.append(
                _signal_row(
                    "Tomogram-edge distance",
                    int(near_edge.sum()),
                    len(near_edge),
                    "particles close to a tomogram boundary (thinner ice / harsher missing-wedge truncation)",
                )
            )
        except ValueError:
            pass  # a tomogram referenced in df isn't in tomogram_dims -- skip this signal, don't break the page

    return {
        "df": df,
        "class_col": class_col,
        "metrics": metrics,
        "id_col": id_col,
        "flags_by_metric": flags_by_metric,
        "thumbnails": _class_thumbnails(job_dir),
        "classes": sorted(df[class_col].dropna().unique(), key=str),
        "stat_grid_html": '<div class="stat-grid">%s</div>' % "".join(stat_tiles),
        "signals_html": "".join(signal_rows),
        "n_standout_total": n_standout_total,
        "combined_flag": combined_flag,
        # ponytail: project-wide, not per-class (tomogram-level flag has no
        # clean per-class join here) -- upgrade to per-class if a project
        # shows classes with meaningfully different tomogram membership.
        "ctf_inconsistent_fraction": ctf_inconsistent_fraction,
    }


def render_page(title, cache, query=None, job_picker_html=""):
    query = query or {}
    overview_card = (
        '<div class="card"><h2>Overview</h2>%s</div>' % cache["signals_html"] if cache["signals_html"] else ""
    )
    export_html = (
        '<p><a href="/export">Download cleaned STAR (excludes the %d standout particles below)</a></p>'
        % cache["n_standout_total"]
        if cache["n_standout_total"]
        else ""
    )
    if cache["metrics"]:
        class_sections = "".join(
            _class_section_html(
                cache["df"],
                cache["class_col"],
                cls,
                cache["metrics"],
                cache["flags_by_metric"],
                cache["id_col"],
                cache["thumbnails"],
                query,
                ctf_inconsistent_fraction=cache["ctf_inconsistent_fraction"],
            )
            for cls in cache["classes"]
        )
    else:
        class_sections = "<p>No CTF/defocus/alignment metric columns found on this particle table.</p>"
    classes_card = '<div class="card"><h2>Classes</h2>%s%s</div>' % (export_html, class_sections)

    safe_title = _esc(str(title))
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>%s</title><style>%s"
        "body{margin:0;}"
        "#topbar{position:sticky;top:0;z-index:1;background:var(--surface);"
        "border-bottom:1px solid var(--border);padding:var(--space-3) var(--space-6);"
        "display:flex;align-items:center;justify-content:space-between;gap:var(--space-3);}"
        "#topbar h1{font-size:16px;margin:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}"
        "#page{max-width:960px;margin:0 auto;padding:var(--space-6) var(--space-4);}"
        ".card{margin-bottom:var(--space-4);}"
        ".signal-row{display:grid;grid-template-columns:180px auto 1fr;align-items:center;"
        "gap:var(--space-3);padding:var(--space-2) 0;border-bottom:1px solid var(--border);}"
        ".signal-row:last-child{border-bottom:none;}"
        ".signal-label{font-weight:600;}"
        ".signal-detail{color:var(--text-muted);font-size:12px;}"
        "details{margin-bottom:var(--space-2);border:1px solid var(--border);border-radius:8px;overflow:hidden;}"
        "details summary{padding:var(--space-2) var(--space-3);cursor:pointer;font-weight:600;background:var(--neutral-bg);}"
        ".class-image{display:block;width:%dpx;height:%dpx;object-fit:contain;background:#000;"
        "border-radius:6px;margin:var(--space-3) 0;image-rendering:pixelated;}"
        ".class-body{padding:0 var(--space-3) var(--space-3);}"
        ".diagnosis{margin:var(--space-2) 0;}"
        "details table{margin:var(--space-2) 0;}"
        "details th,details td{padding:var(--space-2) var(--space-3);}"
        "td.cell-bad{color:var(--bad);font-weight:600;}"
        ".pager{display:flex;align-items:center;gap:var(--space-3);font-size:12px;margin:var(--space-2) 0;}"
        ".pager span{color:var(--text-muted);}"
        ".pager-disabled{color:var(--border);}"
        "</style></head><body>"
        '<div id="topbar"><h1>%s</h1>%s</div>'
        '<div id="page">%s%s%s</div>'
        "</body></html>"
        % (
            safe_title,
            THEME_CSS,
            THUMB_SIZE,
            THUMB_SIZE,
            safe_title,
            job_picker_html,
            cache["stat_grid_html"],
            overview_card,
            classes_card,
        )
    )


def _job_picker_html(jobs, current_job):
    """A `<select>` to switch which job's particles the GUI shows, when
    relion-origin was pointed at a whole project directory instead of a
    single STAR file -- otherwise every job change needs restarting the
    server. Submits via GET so it works with no JS beyond the browser's
    own form handling."""
    if not jobs:
        return ""
    options = "".join(
        '<option value="%s"%s>%s</option>' % (_esc(j), " selected" if j == current_job else "", _esc(j)) for j in jobs
    )
    return '<form method="get" id="jobPicker"><select name="job" onchange="this.form.submit()">%s</select></form>' % options


def serve(
    df,
    class_col="rlnClassNumber",
    title="relion-origin",
    port=DEFAULT_PORT,
    open_browser=True,
    ctf_by_tomogram=None,
    tomogram_dims=None,
    jobs=None,
    current_job=None,
):
    """Start the QC GUI. Blocks until Ctrl-C. Same binding as reliontree
    (127.0.0.1 only) -- see module docstring for HPC/workstation/local
    access via SSH tunnel.

    `ctf_by_tomogram`/`tomogram_dims` (from `io.load_tomograms_star`) are
    optional and STA-only -- see `_build_page_cache`.

    `jobs`/`current_job`: `{job_id: Path}` (from `lineage.list_jobs_with_particles`)
    and the initially-selected job id -- when given, the page shows a job
    picker and switching it re-loads that job's particles on demand
    (cached per job after first load, so flipping back to a job already
    seen doesn't re-parse its STAR file)."""
    from .io import load_particles

    def _job_dir_for(job_id):
        return jobs[job_id].parent if jobs else None

    cache = _build_page_cache(
        df, class_col, ctf_by_tomogram=ctf_by_tomogram, tomogram_dims=tomogram_dims, job_dir=_job_dir_for(current_job)
    )
    job_state = {"current": current_job, "cache": {current_job: cache}} if jobs else None

    def _get_job_cache(job_id):
        if job_id not in job_state["cache"]:
            job_df = load_particles(jobs[job_id])
            job_state["cache"][job_id] = _build_page_cache(
                job_df,
                class_col,
                ctf_by_tomogram=ctf_by_tomogram,
                tomogram_dims=tomogram_dims,
                job_dir=_job_dir_for(job_id),
            )
        return job_state["cache"][job_id]

    class Handler(BaseHTTPRequestHandler):
        def _active(self, query):
            if jobs:
                requested_job = query.get("job", [None])[0]
                job_id = requested_job if requested_job in jobs else job_state["current"]
                job_state["current"] = job_id
                return _get_job_cache(job_id), job_id, _job_picker_html(jobs, job_id)
            return cache, title, ""

        def do_GET(self):
            try:
                parsed = urlsplit(self.path)
                query = parse_qs(parsed.query)
                active_cache, page_title, job_picker = self._active(query)

                if parsed.path == "/export":
                    cleaned = active_cache["df"].loc[~active_cache["combined_flag"]]
                    with tempfile.TemporaryDirectory() as tmp:
                        tmp_path = Path(tmp) / "cleaned_particles.star"
                        starfile.write(cleaned, tmp_path, overwrite=True)
                        body = tmp_path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Disposition", 'attachment; filename="cleaned_particles.star"')
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                body = render_page(page_title, active_cache, query=query, job_picker_html=job_picker).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                body = ("relion-origin error: %s" % exc).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, format, *a):
            pass

    url = "http://127.0.0.1:%d/" % port
    print("relion-origin: %s (Ctrl-C to stop)" % url)
    print("On an HPC login node with no browser: ssh -L %d:127.0.0.1:%d user@cluster, then open %s locally." % (port, port, url))
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
