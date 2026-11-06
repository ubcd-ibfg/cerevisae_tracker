"""
app_tracker.py  -  Particle Tracker (multi-mode)
  Mode A: Single cell  — each TIFF = one cell
  Mode B: Full field   — each TIFF = field with multiple cells
  Mode C: Statistical analysis — compare conditions from CSV exports
Run:  streamlit run app_tracker.py
"""
from stats_page import render_stats_page
from field_analyzer import (
    make_reference_frame, segment_cells, detect_drift, detect_drift_from_cells,
    apply_drift_correction, track_cell_roi, quality_tests,
)
from particle_tracker import (
    load_tiff, detect_signal_end, detect_peak,
    process_trajectory, compute_statistics,
)
from visualization import (
    fig_to_bytes, make_traj_fig, make_signal_fig, make_msd_log,
    make_gif, make_gif_roi, make_field_overview, make_drift_fig,
)
from reporting import build_report, build_cell_report_batch, render_quality_tests
from export import build_selected_cell_rois_zip, build_raw_tif_zip
import streamlit as st
from scipy import stats
import matplotlib.pyplot as plt
import io
import tempfile
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

# ── CONFIG ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Particle Tracker", page_icon="🔬",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.stApp{background-color:#0d0d1a;color:#e0e0f0}
section[data-testid="stSidebar"]{background-color:#111128}
.stTabs [role="tab"]{color:#9090c0;font-weight:600;font-size:14px}
.stTabs [aria-selected="true"]{color:#00d4ff;border-bottom:2px solid #00d4ff}
div[data-testid="metric-container"]{background:#1a1a32;border:1px solid #2a2a50;border-radius:10px;padding:10px 14px}
.sc{background:#0d3326;border:1px solid #22c55e;border-radius:6px;padding:5px 12px;color:#22c55e;font-weight:700;display:inline-block}
.si{background:#2d2000;border:1px solid #f59e0b;border-radius:6px;padding:5px 12px;color:#f59e0b;font-weight:700;display:inline-block}
.sd{background:#3b0d0d;border:1px solid #ef4444;border-radius:6px;padding:5px 12px;color:#ef4444;font-weight:700;display:inline-block}
.hbox{background:linear-gradient(135deg,#1a1a3e,#0d1f3c);border-radius:14px;padding:20px 26px;margin-bottom:16px;border:1px solid #2a2a6e}
.cbox{background:#141430;border:1px solid #2a2a6e;border-radius:10px;padding:14px 18px;margin-bottom:14px}
.qt-pass{color:#22c55e;font-weight:700}
.qt-fail{color:#ef4444;font-weight:700}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Particle Tracker")
    st.markdown("---")
    APP_MODE = st.radio(
        "Mode",
        ["🧬 Single cell", "🔭 Full field (batch)", "📊 Statistical analysis"],
        label_visibility="collapsed",
    )
    st.markdown(f"**Active mode:** {APP_MODE}")
    st.markdown("---")
    if APP_MODE == "📊 Statistical analysis":
        st.markdown(
            "Upload two CSV files (batch mode output) in the main panel.")
        uploaded_files = []
    elif APP_MODE == "🧬 Single cell":
        st.markdown("### Cells")
        uploaded_files = st.file_uploader("One TIFF per cell", type=["tif", "tiff"],
                                          accept_multiple_files=True, key="uf_cell")
        st.markdown("---")
    else:
        st.markdown("### Fields")
        uploaded_files = st.file_uploader("One TIFF per complete field",
                                          type=["tif", "tiff"], accept_multiple_files=True,
                                          key="uf_field")
        st.markdown("---")
        st.markdown("### Segmentation")
        seg_min_size = st.slider(
            "Min. cell size (px²)", 50, 1000, 150, step=50)
        seg_max_size = st.slider("Max. cell size (px²)", 200, 20000, 5000, step=100,
                                 help="Cells above this area are automatically excluded (possible merges)")
        seg_sigma = st.slider("Segmentation smoothing",
                              0.5, 8.0, 2.5, step=0.25)
        seg_dilation = st.slider("Edge dilation (px)", 0, 12, 4)
        seg_watershed_diam = st.slider(
            "Min. cell diameter — cluster split (px)", 5, 500, 15, step=1,
            help="Set this to the diameter of your cells in pixels "
                 "(measure from the Phase 1 overview: diameter ≈ 2·√(area/π)). "
                 "Controls the minimum distance between watershed seeds when "
                 "splitting a cell cluster. Too small → over-splits noise; "
                 "too large → fails to separate touching cells.")
        st.markdown("---")
        st.markdown("### Drift")
        drift_thr = st.slider("Drift threshold (px)", 1.0, 20.0, 3.0, step=0.5,
                              help="Drift > this value is considered significant")
        drift_sigma = st.slider("Drift detection smoothing", 0.5, 5.0, 2.0, step=0.25,
                                help="Higher = more robust but less precise. Adjust if drift isn't detected properly")
        drift_detect_method = st.radio(
            "Detection method",
            ["Phase correlation", "Segmentation (first↔last)"],
            horizontal=True,
            help=(
                "**Phase correlation** (FFT): fast, sub-pixel, works frame-by-frame. "
                "**Segmentation** (recommended for systematic drift): segments cell bodies "
                "in the first and last frame, runs phase correlation on the binary masks "
                "to find the total displacement vector, then interpolates linearly. "
                "Insensitive to intensity changes and photobleaching."
            ),
        )
        drift_correct_method = st.radio("Correction method",
                                        ["Direct", "Smoothed"],
                                        horizontal=True,
                                        help="Smoothed interpolates drift for smoother correction")
        st.markdown("---")
    if APP_MODE != "📊 Statistical analysis":
        st.markdown("### Signal loss")
        auto_cut = st.toggle("Automatic cut", value=True)
        sig_thr = st.slider("Threshold (% intensity)", 3,
                            60, 15, disabled=not auto_cut)
        sig_win = st.slider("Consecutive frames", 2, 20,
                            7, disabled=not auto_cut)
        sig_smo = st.slider("Trace smoothing", 1, 15, 5, disabled=not auto_cut)
        st.markdown("---")
        st.markdown("### Detection")
        thr_pct = st.slider("Min. intensity percentile",
                            50.0, 99.9, 95.0, step=0.5)
        roi_r = st.slider("ROI radius (px)", 2, 20, 5)
        g_sigma = st.slider("Gaussian sigma", 0.5, 5.0, 1.5, step=0.25)
        st.markdown("---")
        st.markdown("### Interpolation")
        max_gap = st.slider("Max. missing (%)", 0, 80, 20, step=5)
        st.markdown("---")
        st.markdown("### GIF")
        gen_gif = st.toggle("Generate GIF", value=True)
        tail_len = st.slider("Tail (frames)", 5, 50, 15,
                             step=5, disabled=not gen_gif)
        gif_fps = st.slider("FPS", 3, 30, 10, disabled=not gen_gif)
        st.markdown("---")
        if APP_MODE == "🧬 Single cell":
            run_btn = st.button("Analyze all cells",
                                type="primary", use_container_width=True)
        else:
            run_btn = st.button("Scan fields (Phase 1)",
                                type="primary", use_container_width=True)
    else:
        auto_cut = sig_thr = sig_win = sig_smo = None
        thr_pct = roi_r = g_sigma = max_gap = None
        gen_gif = tail_len = gif_fps = None
        seg_min_size = 150; seg_max_size = 5000; seg_sigma = 2.5; seg_dilation = 4; seg_watershed_diam = 18
        drift_thr = 3.0; drift_sigma = 2.0; drift_correct_method = "Direct"
        drift_detect_method = "Phase correlation"
        run_btn = False

# ── HEADER ────────────────────────────────────────────────────────────────────
if APP_MODE == "📊 Statistical analysis":
    render_stats_page()
    st.stop()
elif APP_MODE == "🧬 Single cell":
    st.markdown("""<div class="hbox"><h1 style="margin:0;color:#00d4ff;font-size:2rem;">
    🔬 Particle Tracker <span style="font-size:1rem;color:#555588;">— single cell</span></h1>
    <p style="margin:6px 0 0;color:#8888cc;font-size:0.9rem;">
    Each TIFF = one independent cell &nbsp;·&nbsp; automatic signal cut &nbsp;·&nbsp; interpolation &nbsp;·&nbsp; MSD
    </p></div>""", unsafe_allow_html=True)
else:
    st.markdown("""<div class="hbox"><h1 style="margin:0;color:#00d4ff;font-size:2rem;">
    🔬 Particle Tracker <span style="font-size:1rem;color:#555588;">— full field (batch)</span></h1>
    <p style="margin:6px 0 0;color:#8888cc;font-size:0.9rem;">
    Each TIFF = field with multiple cells &nbsp;·&nbsp; segmentation &nbsp;·&nbsp;
    drift detection &nbsp;·&nbsp; relative tracking &nbsp;·&nbsp; quality tests
    </p></div>""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "cells" not in st.session_state:
    st.session_state.cells = {}
    st.session_state.sel = None
for key in ("batch_phase", "batch_fields", "batch_results", "batch_sel_field", "batch_sel_cell"):
    if key not in st.session_state:
        st.session_state[key] = "idle" if key == "batch_phase" else (
            {} if "fields" in key or "results" in key else None)


# ══════════════════════════════════════════════════════════════════════════════
# MODE A — SINGLE CELL
# ══════════════════════════════════════════════════════════════════════════════
if APP_MODE == "🧬 Single cell":

    if run_btn:
        if not uploaded_files:
            st.error("Upload at least one TIFF file.")
            st.stop()
        cells = {}
        n_cells = len(uploaded_files)
        pg = st.progress(0, text=f"Analyzing 0/{n_cells}...")
        sb = st.empty()
        params_used = {"thr_pct": thr_pct, "max_gap": max_gap, "roi": roi_r,
                       "sigma": g_sigma, "auto_cut": auto_cut,
                       "sig_thr": sig_thr if auto_cut else "-"}
        with tempfile.TemporaryDirectory() as tmp:
            tp = Path(tmp)
            for idx, uf in enumerate(uploaded_files):
                cname = Path(uf.name).stem
                sb.info(f"Processing **{cname}** ({idx+1}/{n_cells})...")
                fpath = tp / uf.name
                fpath.write_bytes(uf.read())
                try:
                    sfull = load_tiff(fpath)
                    n_full = sfull.shape[0]
                    # Signal end detection: identify frame where signal drops below threshold
                    # Useful for discarding frames with weak/no signal at end of acquisition
                    if auto_cut:
                        si = detect_signal_end(sfull, threshold_ratio=sig_thr/100,
                                               window=sig_win, gaussian_sigma=g_sigma,
                                               smooth_window=sig_smo)
                        ef = si["end_frame"]
                        stack = sfull[:ef+1]
                    else:
                        si = None
                        ef = n_full-1
                        stack = sfull
                    n_fr = stack.shape[0]
                    # Peak detection and trajectory processing per frame
                    raw = []
                    for i in range(n_fr):
                        d = detect_peak(stack[i], threshold_pct=thr_pct,
                                        roi_radius=roi_r, gaussian_sigma=g_sigma)
                        raw.append((i, d[0], d[1], d[2]) if d else None)
                    # Link detections into trajectory, interpolate gaps
                    res = process_trajectory(raw, max_gap_ratio=max_gap/100)
                    sd = compute_statistics(
                        res["trajectory"]) if res["status"] != "discarded" else None
                    tf_b = None
                    gf_b = None
                    if sd:
                        fig = make_traj_fig(res, sd, stack=stack)
                        tf_b = fig_to_bytes(fig)
                        plt.close(fig)
                        if gen_gif:
                            gf_b = make_gif(stack, res, tail_len, gif_fps)
                    cells[cname] = {"result": res, "sd": sd, "si": si, "stack": stack, "n_full": n_full,
                                    "tf_b": tf_b, "gif": gf_b,
                                    "report": build_report(cname, res, sd, si,
                                                           {"Loaded frames": n_full, "Analyzed frames": n_fr, **params_used})}
                except Exception as e:
                    cells[cname] = {"error": str(e)}
                pg.progress((idx+1)/n_cells,
                            text=f"Analyzing {idx+1}/{n_cells}...")
        pg.empty()
        sb.empty()
        st.session_state.cells = cells
        st.session_state.sel = list(cells.keys())[0] if cells else None
        st.success(f"Analysis complete — {n_cells} cells processed.")

    cells = st.session_state.cells
    if not cells:
        st.markdown("""<div style="text-align:center;padding:60px 0;">
        <div style="font-size:4rem;">🧬</div>
        <h3 style="color:#5555bb;">Upload your TIFFs and press <b>Analyze</b></h3>
        <p style="color:#4444aa;max-width:480px;margin:auto;">
        One TIFF per cell. The app analyzes all in batch and lets you navigate through results.
        </p></div>""", unsafe_allow_html=True)
        st.stop()

    with st.expander("📋 Summary — all cells", expanded=True):
        rows = []
        for name, d in cells.items():
            if "error" in d:
                rows.append({"Cell": name, "Status": "ERROR", "Loaded": "—", "Anal.": "—",
                             "Det.": "—", "Interp.": "—", "alpha": "—", "Motion": d["error"]})
                continue
            r = d["result"]
            sd2 = d["sd"]
            rows.append({"Cell": name, "Status": r["status"].upper(),
                         "Loaded": d["n_full"], "Anal.": r["n_frames"],
                         "Det.": r["n_detected"], "Interp.": r["n_frames"]-r["n_detected"],
                         "alpha": f"{sd2['alpha_msd']:.2f}" if sd2 and not np.isnan(sd2["alpha_msd"]) else "—",
                         "Motion": sd2["motion_type"] if sd2 else "—"})

        def cs(v):
            if v == "COMPLETE":
                return "color:#22c55e;font-weight:bold"
            if v == "INTERPOLATED":
                return "color:#f59e0b;font-weight:bold"
            if v in ("DISCARDED", "ERROR"):
                return "color:#ef4444;font-weight:bold"
            return ""
        df_sum = pd.DataFrame(rows)
        st.dataframe(df_sum.style.applymap(cs, subset=["Status"]),
                     use_container_width=True, height=min(300, 60+35*len(cells)))
        csv_rows = []
        for name, d in cells.items():
            if "error" in d or d["result"]["status"] == "discarded":
                continue
            for t in d["result"]["trajectory"]:
                csv_rows.append({"cell_id": name, "frame": t[0], "x_px": round(t[1], 4),
                                 "y_px": round(t[2], 4),
                                 "intensity": round(t[3], 2) if not np.isnan(t[3]) else np.nan,
                                 "interpolated": t[4]})
        if csv_rows:
            st.download_button("Download combined CSV",
                               data=pd.DataFrame(csv_rows).to_csv(
                                   index=False).encode("utf-8"),
                               file_name="all_cells.csv", mime="text/csv")

    st.markdown('<div class="cbox">', unsafe_allow_html=True)
    names = list(cells.keys())
    idx0 = names.index(
        st.session_state.sel) if st.session_state.sel in names else 0
    cs1, cs2, cs3 = st.columns([6, 1, 1])
    with cs2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("◀", use_container_width=True, disabled=idx0 == 0):
            st.session_state.sel = names[idx0-1]
            st.rerun()
    with cs3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("▶", use_container_width=True, disabled=idx0 == len(names)-1):
            st.session_state.sel = names[idx0+1]
            st.rerun()
    with cs1:
        sel = st.selectbox(
            f"Cell ({len(names)} total)", options=names, index=idx0)
        if sel != st.session_state.sel:
            st.session_state.sel = sel
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    cname = st.session_state.sel
    data = cells[cname]
    st.markdown(f"### 🧬 `{cname}`  <small style='color:#555588;'>({idx0+1}/{len(names)})</small>",
                unsafe_allow_html=True)
    if "error" in data:
        st.error(f"Error: {data['error']}")
        st.stop()
    res = data["result"]
    sd = data["sd"]
    si = data["si"]
    stack = data["stack"]
    n_full = data["n_full"]
    icns = {"complete": "✅", "interpolated": "🔄", "discarded": "❌"}
    lbls = {"complete": "COMPLETE trajectory",
            "interpolated": f"INTERPOLATED — {res['n_frames']-res['n_detected']} interp. ({res['missing_ratio']*100:.1f}%)",
            "discarded": f"DISCARDED — {res['missing_ratio']*100:.1f}% missing"}
    css_map = {"complete": "sc", "interpolated": "si", "discarded": "sd"}
    st.markdown(f"<div class='{css_map[res['status']]}' style='margin-bottom:10px'>"
                f"{icns[res['status']]}  {lbls[res['status']]}</div>", unsafe_allow_html=True)
    if si and si["auto_cut"]:
        pct = 100*(1-res["n_frames"]/n_full)
        st.info(
            f"📉 Cut at frame **{si['end_frame']}** — {res['n_frames']}/{n_full} frames ({pct:.0f}% saved)")
    if res["status"] == "discarded":
        st.warning("Cell discarded. Adjust parameters and re-analyze.")
        if res.get("gap_info"):
            for g in res["gap_info"]:
                st.markdown(f"- Gap **{g[0]}-{g[1]}** ({g[2]} frames)")
        st.stop()
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Loaded", n_full)
    with m2:
        dlt = f"-{n_full-res['n_frames']}" if res["n_frames"] < n_full else None
        st.metric("Analyzed", res["n_frames"], delta=dlt, delta_color="off")
    with m3:
        st.metric("Detected", res["n_detected"])
    with m4:
        st.metric("Interpolated", res["n_frames"]-res["n_detected"])
    with m5:
        a = sd["alpha_msd"]
        st.metric("Alpha α", f"{a:.2f}" if not np.isnan(a) else "—")
    with m6:
        st.metric("Motion type", sd["motion_type"])
    st.markdown("")
    t1, t2, t3, t4, t5, t6 = st.tabs(
        ["📈 Trajectory", "📉 Signal", "📊 MSD log-log", "🎬 GIF", "🗃️ CSV", "📄 Report"])
    with t1:
        if data["tf_b"]:
            st.image(data["tf_b"], use_container_width=True)
            st.markdown("<small style='color:#666'>o=start | *=end | x=interpolated | color=time</small>",
                        unsafe_allow_html=True)
            st.markdown("---")
            ca, cb = st.columns(2)
            with ca:
                st.markdown("**Geometry**")
                st.markdown(
                    f"- Total path length: **{sd['total_path_px']:.2f} px**")
                st.markdown(
                    f"- End-to-end disp.: **{sd['end_to_end_px']:.2f} px**")
                st.markdown(
                    f"- Confinement: **{sd['confinement_ratio']:.3f}**")
                st.markdown(
                    f"- Gyration radius: **{sd['gyration_radius_px']:.2f} px**")
            with cb:
                st.markdown("**Speed**")
                st.markdown(
                    f"- Mean: **{sd['mean_speed_px_frame']:.4f} px/frame**")
                st.markdown(
                    f"- Max: **{sd['max_speed_px_frame']:.4f} px/frame**")
                st.markdown("**Intensity**")
                st.markdown(f"- Mean: **{sd['mean_intensity']:.1f}**")
                st.markdown(f"- Std. dev.: **{sd['std_intensity']:.1f}**")
    with t2:
        if si:
            sf = make_signal_fig(si, n_full)
            st.pyplot(sf, use_container_width=True)
            plt.close(sf)
            st.markdown("---")
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.metric("Cut frame", si["end_frame"])
            with sc2:
                st.metric("Abs. threshold", f"{si['threshold_abs']:.0f}")
            with sc3:
                pct2 = 100*(1-res["n_frames"]/n_full)
                st.metric("Frames saved", f"{pct2:.0f}%")
        else:
            st.info("Enable automatic cut in the sidebar.")
    with t3:
        mf = make_msd_log(sd)
        st.pyplot(mf, use_container_width=False)
        plt.close(mf)
        st.markdown("---")
        mm1, mm2, mm3 = st.columns(3)
        a2 = sd["alpha_msd"]
        D2 = sd["diffusion_coefficient"]
        di2 = sd["directionality_index"]
        with mm1:
            st.metric("Alpha α", f"{a2:.3f}" if not np.isnan(a2) else "—")
        with mm2:
            st.metric("D (px²/frame)",
                      f"{D2:.4f}" if not np.isnan(D2) else "—")
        with mm3:
            st.metric("Direct. index",
                      f"{di2:.3f}" if not np.isnan(di2) else "—")
        st.markdown(f"""
| Regime | α | Interpretation |
|---|---|---|
| Subdiffusive | < 0.7 | Confined / trapped |
| Diffusive | 0.7–1.3 | Free Brownian |
| Superdiffusive | 1.3–1.8 | Directed / propelled |
| Ballistic | > 1.8 | Active motor |

**{cname}: α={f'{a2:.2f}' if not np.isnan(a2) else 'N/A'} → {sd['motion_type']}**
""")
    with t4:
        if data["gif"]:
            st.image(data["gif"], caption=f"GIF — {cname}")
            st.download_button(f"Download GIF — {cname}", data=data["gif"],
                               file_name=f"{cname}_annotated.gif", mime="image/gif")
        elif not gen_gif:
            st.info("GIF generation disabled.")
        else:
            st.info("No GIF available.")
    with t5:
        trows = [{"cell_id": cname, "frame": t[0], "x_px": round(t[1], 4), "y_px": round(t[2], 4),
                  "intensity": round(t[3], 2) if not np.isnan(t[3]) else np.nan, "interpolated": t[4]}
                 for t in res["trajectory"]]
        dfc = pd.DataFrame(trows)
        st.dataframe(dfc.style.applymap(
            lambda v: "background-color:#2d2000;color:#f59e0b" if v is True else "",
            subset=["interpolated"]), use_container_width=True, height=380)
        st.download_button(f"Download CSV — {cname}",
                           data=dfc.to_csv(index=False).encode("utf-8"),
                           file_name=f"{cname}_trajectory.csv", mime="text/csv")
    with t6:
        st.code(data["report"], language=None)
        st.download_button(f"Download report — {cname}", data=data["report"].encode("utf-8"),
                           file_name=f"{cname}_report.txt", mime="text/plain")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if data["tf_b"]:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(f"{cname}_trajectory.png", data["tf_b"])
                zf.writestr(f"{cname}_trajectory.csv", pd.DataFrame(
                    trows).to_csv(index=False).encode("utf-8"))
                if data["gif"]:
                    zf.writestr(f"{cname}_annotated.gif", data["gif"])
                zf.writestr(f"{cname}_report.txt",
                            data["report"].encode("utf-8"))
            buf.seek(0)
            st.download_button(f"📦 ZIP — {cname}", data=buf.read(),
                               file_name=f"{cname}_results.zip", mime="application/zip",
                               use_container_width=True)
    with col2:
        buf_all = io.BytesIO()
        all_csv = []
        with zipfile.ZipFile(buf_all, "w", zipfile.ZIP_DEFLATED) as zf:
            for nm, d in cells.items():
                if "error" in d:
                    continue
                if d["tf_b"]:
                    zf.writestr(f"{nm}/{nm}_trajectory.png", d["tf_b"])
                if d["gif"]:
                    zf.writestr(f"{nm}/{nm}_annotated.gif",  d["gif"])
                zf.writestr(f"{nm}/{nm}_report.txt",
                            d["report"].encode("utf-8"))
                for t in d["result"]["trajectory"]:
                    all_csv.append({"cell_id": nm, "frame": t[0], "x_px": round(t[1], 4),
                                    "y_px": round(t[2], 4),
                                    "intensity": round(t[3], 2) if not np.isnan(t[3]) else np.nan,
                                    "interpolated": t[4]})
            if all_csv:
                zf.writestr("all_cells.csv", pd.DataFrame(
                    all_csv).to_csv(index=False).encode("utf-8"))
        buf_all.seek(0)
        st.download_button("📦 ZIP — all cells", data=buf_all.read(),
                           file_name="all_cells_results.zip", mime="application/zip",
                           use_container_width=True, type="primary")
    st.markdown("<div style='text-align:center;margin-top:28px;color:#333355;font-size:0.78rem;'>"
                "Particle Tracker · single cell · automatic cut · MSD</div>",
                unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# MODE B — FULL FIELD (BATCH)
# ══════════════════════════════════════════════════════════════════════════════
else:

    # ── PHASE 1: SCAN ─────────────────────────────────────────────────────────
    if run_btn:
        if not uploaded_files:
            st.error("Upload at least one field TIFF.")
            st.stop()
        n_fields = len(uploaded_files)
        fields = {}
        pg = st.progress(0, text="Scanning fields...")
        sb = st.empty()
        with tempfile.TemporaryDirectory() as tmp:
            tp = Path(tmp)
            for idx, uf in enumerate(uploaded_files):
                fname = Path(uf.name).stem
                sb.info(f"Scanning **{fname}** ({idx+1}/{n_fields})...")
                fpath = tp / uf.name
                fpath.write_bytes(uf.read())
                try:
                    sfull = load_tiff(fpath)
                    n_full = sfull.shape[0]
                    # Apply signal end detection if enabled (same as Mode A)
                    if auto_cut:
                        si = detect_signal_end(sfull, threshold_ratio=sig_thr/100,
                                               window=sig_win, gaussian_sigma=g_sigma,
                                               smooth_window=sig_smo)
                        stack = sfull[:si["end_frame"]+1]
                    else:
                        si = None
                        stack = sfull
                    # Create reference frame and segment cells
                    ref_frame = make_reference_frame(stack)
                    cells_list, labeled = segment_cells(
                        ref_frame, min_size=seg_min_size, max_size=seg_max_size,
                        sigma=seg_sigma, dilation_px=seg_dilation,
                        min_cell_diameter=seg_watershed_diam)
                    # Detect field-level drift for all cells
                    if drift_detect_method == "Segmentation (first↔last)":
                        drift_info = detect_drift_from_cells(
                            stack, sigma=seg_sigma,
                            min_size=seg_min_size, max_size=seg_max_size,
                            dilation_px=seg_dilation,
                            drift_threshold_px=drift_thr)
                    else:
                        drift_info = detect_drift(
                            stack, sigma=drift_sigma, drift_threshold_px=drift_thr)
                    ov_fig = make_field_overview(
                        ref_frame, cells_list, labeled)
                    ov_bytes = fig_to_bytes(ov_fig)
                    plt.close(ov_fig)
                    if drift_info["is_significant"]:
                        df_fig = make_drift_fig(drift_info, fname)
                        df_bytes = fig_to_bytes(df_fig)
                        plt.close(df_fig)
                    else:
                        df_bytes = None
                    fields[fname] = {
                        "stack": stack, "si": si, "n_full": n_full,
                        "ref_frame": ref_frame, "cells": cells_list,
                        "labeled": labeled, "drift_info": drift_info,
                        "ov_bytes": ov_bytes, "drift_bytes": df_bytes,
                        "decision": "continue",
                    }
                except Exception as e:
                    fields[fname] = {"error": str(e)}
                pg.progress((idx+1)/n_fields,
                            text=f"Scanning {idx+1}/{n_fields}...")
        pg.empty()
        sb.empty()
        for key in list(st.session_state.keys()):
            if key.startswith("cell_inc_"):
                del st.session_state[key]
        st.session_state.batch_fields = fields
        st.session_state.batch_results = {}
        st.session_state.batch_phase = "scan_done"
        st.session_state.batch_sel_field = next(
            (fn for fn in fields if "error" not in fields[fn]), None)
        st.session_state.batch_sel_cell = None
        st.success(f"Scan complete — {n_fields} fields processed.")
        st.rerun()

    phase = st.session_state.batch_phase
    fields = st.session_state.batch_fields

    # ── IDLE ─────────────────────────────────────────────────────────────────
    if phase == "idle":
        st.markdown("""<div style="text-align:center;padding:60px 0;">
        <div style="font-size:4rem;">🔭</div>
        <h3 style="color:#5555bb;">Upload your fields and press <b>Scan fields (Phase 1)</b></h3>
        <p style="color:#4444aa;max-width:520px;margin:auto;">
        Each TIFF must be a complete field with multiple cells.
        The app segments cells, detects drift, and tracks each one independently.
        </p></div>""", unsafe_allow_html=True)
        st.stop()

    # ── PHASE 2: DRIFT REVIEW ─────────────────────────────────────────────────
    if phase == "scan_done":
        st.markdown("## 📊 Scan results")
        fields_drift = {fn: fd for fn, fd in fields.items(
        ) if "error" not in fd and fd["drift_info"]["is_significant"]}
        fields_ok = {fn: fd for fn, fd in fields.items(
        ) if "error" not in fd and not fd["drift_info"]["is_significant"]}
        fields_err = {fn: fd for fn, fd in fields.items() if "error" in fd}
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Total fields", len(fields))
        with col_b:
            st.metric("With drift", len(fields_drift))
        with col_c:
            st.metric("Without drift", len(fields_ok))
        with col_d:
            st.metric("Errors", len(fields_err))

        # Drift detection help
        with st.expander("❓ Drift detection & correction guide"):
            st.markdown("""
### How to detect drift correctly

**Drift parameters in sidebar:**
- **Drift threshold (px)**: Minimum drift to flag as "significant". Lower = more sensitive, may flag noise
- **Drift detection smoothing**: Controls sensitivity. 
  - Lower (0.5-1.0): More sensitive, picks up weak drift
  - Higher (2.0-3.0): More robust, ignores noise
  
**If drift isn't detected:**
1. Lower the **drift threshold** (try 1.0-2.0 px)
2. Lower the **detection smoothing** to make it more sensitive
3. Check if your fields have actual drift (compare first and last frames)

**If false positives (drift detected when there is none):**
1. Raise the **drift threshold** (try 5.0-8.0 px)
2. Raise the **detection smoothing** to 3.0-4.0 for robustness
3. Choose "Smoothed" correction method for stability

**Correction method:**
- **Direct**: Applies detected drift directly (faster)
- **Smoothed**: Interpolates drift trajectory for smoother correction (better for noisy drift)
            """)

        if fields_err:
            with st.expander("❌ Fields with errors"):
                for fn, fd in fields_err.items():
                    st.error(f"**{fn}**: {fd['error']}")
        st.markdown("### 🗺️ Segmented fields")
        valid_fns = [fn for fn, fd in fields.items() if "error" not in fd]
        n_show = min(len(valid_fns), 3)
        if n_show:
            ov_cols = st.columns(n_show)
            for i, fn in enumerate(valid_fns[:n_show]):
                fd = fields[fn]
                with ov_cols[i]:
                    st.markdown(
                        f"**{fn}** — {len(fd['cells'])} segmented cells")
                    st.image(fd["ov_bytes"], use_container_width=True)
        st.markdown("### 🔬 Manual cell exclusion")
        st.info(
            "Uncheck cells you want to exclude from tracking "
            "(e.g. large regions that are merges of multiple cells). "
            "Cells marked ⚠️ have an area >2× the field mean."
        )
        for fn in valid_fns:
            fd = fields[fn]
            if not fd["cells"]:
                continue
            areas = [c["area"] for c in fd["cells"]]
            mean_area = float(np.mean(areas)) if areas else 1.0
            with st.expander(
                f"**{fn}** — {len(fd['cells'])} cells  "
                f"(mean area: {mean_area:.0f} px²  |  "
                f"min: {min(areas)} px²  |  max: {max(areas)} px²)",
                expanded=True,
            ):
                cols_per_row = 5
                for row_start in range(0, len(fd["cells"]), cols_per_row):
                    row_cells = fd["cells"][row_start: row_start + cols_per_row]
                    row_cols = st.columns(len(row_cells))
                    for col_widget, cell in zip(row_cols, row_cells):
                        with col_widget:
                            clabel = f"C{cell['label']}"
                            area = cell["area"]
                            flag = " ⚠️" if area > mean_area * 2 else ""
                            st.checkbox(
                                f"{clabel}: {area} px²{flag}",
                                value=True,
                                key=f"cell_inc_{fn}_{clabel}",
                            )
        st.markdown("---")

        if fields_drift:
            st.markdown("### ⚠️ Fields with significant drift")
            st.info(
                "Drift was detected in the following fields. Choose how to proceed with each one.")
            with st.form("drift_review_form"):
                decisions = {}
                for fn, fd in fields_drift.items():
                    di = fd["drift_info"]
                    st.markdown(f"**Field: `{fn}`**")
                    dc1, dc2 = st.columns([2, 3])
                    with dc1:
                        st.markdown(
                            f"- Method: **{di.get('method', 'phase_correlation')}**")
                        st.markdown(
                            f"- Max drift: **{di['max_drift_px']:.2f} px**")
                        st.markdown(
                            f"- Cumulative: **{di['cumulative_px']:.2f} px**")
                        st.markdown(
                            f"- Direction: dy={di['slope_y']:.3f} px/f  dx={di['slope_x']:.3f} px/f")
                        if di.get("method") == "segmentation_mask":
                            st.markdown(
                                f"- Cells (first/last): "
                                f"**{di.get('n_cells_first', '?')}** / **{di.get('n_cells_last', '?')}**")
                            st.markdown(
                                f"- Total vector: dy={di.get('total_dy', 0):.1f}  "
                                f"dx={di.get('total_dx', 0):.1f} px")
                    with dc2:
                        if fd["drift_bytes"]:
                            st.image(fd["drift_bytes"],
                                     use_container_width=True)
                    decisions[fn] = st.radio(
                        f"Action — {fn}",
                        ["Correct drift automatically",
                         "Continue without correcting",
                         "Skip this field"],
                        horizontal=True, label_visibility="collapsed", key=f"dr_{fn}")
                    st.markdown("---")
                submitted = st.form_submit_button("✅ Confirm and track all cells",
                                                  type="primary", use_container_width=True)
                if submitted:
                    dec_map = {"Correct drift automatically": "correct",
                               "Continue without correcting": "continue",
                               "Skip this field": "skip"}
                    for fn in fields_drift:
                        fields[fn]["decision"] = dec_map[decisions[fn]]
                    st.session_state.batch_fields = fields
                    st.session_state.batch_phase = "tracking"
                    st.rerun()
        else:
            st.success("No field has significant drift.")
            if st.button("▶ Track all cells", type="primary"):
                st.session_state.batch_phase = "tracking"
                st.rerun()
        st.stop()

    # ── PHASE 3: TRACKING ────────────────────────────────────────────────────
    if phase == "tracking":
        st.markdown("## 🔭 Tracking cells...")
        results = {}
        total_c = sum(
            sum(1 for cell in fd["cells"]
                if st.session_state.get(f"cell_inc_{fn}_C{cell['label']}", True))
            for fn, fd in fields.items()
            if "error" not in fd and fd.get("decision", "continue") != "skip"
        )
        done_c = 0
        pg = st.progress(0, text="Starting...")
        sb = st.empty()
        params_u = {"thr_pct": thr_pct, "max_gap": max_gap,
                    "roi": roi_r, "sigma": g_sigma}
        for fname, fd in fields.items():
            if "error" in fd:
                continue
            decision = fd.get("decision", "continue")
            if decision == "skip":
                results[fname] = {"skipped": True, "cells": {}, "ov_bytes": fd["ov_bytes"],
                                  "drift_bytes": fd.get("drift_bytes"),
                                  "drift_info": fd["drift_info"], "decision": "skip",
                                  "n_cells_segmented": len(fd["cells"])}
                continue
            stack = fd["stack"]
            if decision == "correct":
                method = "spline" if drift_correct_method == "Smoothed" else "linear"
                stack = apply_drift_correction(
                    stack, fd["drift_info"], method=method)
                drift_used = None
            else:
                drift_used = fd["drift_info"]
            field_cells = {}
            for cell in fd["cells"]:
                clabel = f"C{cell['label']}"
                if not st.session_state.get(f"cell_inc_{fname}_{clabel}", True):
                    continue
                sb.info(
                    f"Tracking **{fname}** → **{clabel}** ({done_c+1}/{total_c})...")
                try:
                    raw_det = track_cell_roi(stack, cell, drift_info=drift_used,
                                             threshold_pct=thr_pct, roi_radius=roi_r,
                                             gaussian_sigma=g_sigma)
                    proc_in = [(d[0], d[1], d[2], d[3])
                               if d else None for d in raw_det]
                    res = process_trajectory(
                        proc_in, max_gap_ratio=max_gap/100)
                    sd = compute_statistics(
                        res["trajectory"]) if res["status"] != "discarded" else None
                    qt = quality_tests(res, sd, fd.get("si"))
                    tf_b = None
                    gf_b = None
                    if sd:
                        fig = make_traj_fig(res, sd, cell_label=clabel)
                        tf_b = fig_to_bytes(fig)
                        plt.close(fig)
                        if gen_gif:
                            gf_b = make_gif_roi(
                                stack, cell, raw_det, res, drift_used, tail_len, gif_fps)
                    rpt = build_cell_report_batch(
                        fname, clabel, res, sd, qt, params_u)
                    field_cells[clabel] = {"result": res, "sd": sd, "qt": qt, "raw_det": raw_det,
                                           "tf_b": tf_b, "gif": gf_b, "report": rpt, "cell_meta": cell}
                except Exception as e:
                    field_cells[clabel] = {"error": str(e)}
                done_c += 1
                pg.progress(done_c/max(total_c, 1),
                            text=f"Tracking {done_c}/{total_c}...")
            results[fname] = {"skipped": False, "cells": field_cells, "drift_info": fd["drift_info"],
                              "decision": decision, "ov_bytes": fd["ov_bytes"],
                              "drift_bytes": fd.get("drift_bytes"),
                              "n_cells_segmented": len(fd["cells"])}
        pg.empty()
        sb.empty()
        st.session_state.batch_results = results
        st.session_state.batch_phase = "done"
        sel_f = next((fn for fn, r in results.items()
                     if not r.get("skipped")), None)
        st.session_state.batch_sel_field = sel_f
        st.session_state.batch_sel_cell = next(
            iter(results[sel_f]["cells"]), None) if sel_f else None
        st.success("Tracking complete.")
        st.rerun()

    # ── RESULTS ───────────────────────────────────────────────────────────────
    if phase == "done":
        results = st.session_state.batch_results
        if not results:
            st.warning("No results. Re-analyze.")
            st.stop()

        all_rows = []
        for fname, fr in results.items():
            if fr.get("skipped"):
                continue
            for clabel, cd in fr["cells"].items():
                if "error" in cd:
                    all_rows.append({"Field": fname, "Cell": clabel, "Status": "ERROR",
                                     "Frames": "—", "Det.": "—", "alpha": "—", "Motion": "—", "QT": "—"})
                    continue
                r2 = cd["result"]
                sd2 = cd["sd"]
                qt2 = cd["qt"]
                ovr = qt2.get("_summary", {}).get(
                    "overall", "?") if qt2 else "?"
                all_rows.append({"Field": fname, "Cell": clabel, "Status": r2["status"].upper(),
                                 "Frames": r2["n_frames"], "Det.": r2["n_detected"],
                                 "Interp": r2["n_frames"] - r2["n_detected"],
                                 "alpha": f"{sd2['alpha_msd']:.2f}" if sd2 and not np.isnan(sd2["alpha_msd"]) else "—",
                                 "Motion": sd2["motion_type"] if sd2 else "—", "QT": ovr})

        with st.expander("📋 Global summary — all fields and cells", expanded=True):
            total_frames = 0
            total_interpolated = 0
            selection_options = []
            for fname, fr in results.items():
                if fr.get("skipped"):
                    continue
                for clabel, cd in fr["cells"].items():
                    if "error" in cd:
                        continue
                    total_frames += cd["result"]["n_frames"]
                    total_interpolated += cd["result"]["n_frames"] - \
                        cd["result"]["n_detected"]
                    selection_options.append(f"{fname}/{clabel}")

            st.markdown(
                f"**Interpolated frames:** {total_interpolated} / {total_frames}  "
                f"({total_interpolated*100/total_frames:.1f}%)"
                if total_frames else ""
            )
            st.markdown("---")

            ac1, ac2 = st.columns([1, 4])
            with ac1:
                max_interp_sel = st.slider(
                    "Max interpolated frames for auto-select", 0, 50, 0,
                    help="Allow cells with up to this many interpolated frames to be auto-selected.")
                qt_filter = st.multiselect(
                    "QC status filter",
                    options=["PASS", "WARN", "FAIL"],
                    default=["PASS"],
                    help="Auto-select cells with these QC statuses.")
                if st.button(
                        f"⭐ Auto-select: QT {'/'.join(qt_filter)}, ≤ {max_interp_sel} interp. frames",
                        use_container_width=True):
                    auto_sel = []
                    for _fn, _fr in results.items():
                        if _fr.get("skipped"):
                            continue
                        for _cl, _cd in _fr["cells"].items():
                            if "error" in _cd:
                                continue
                            _qt = _cd["qt"]
                            _r = _cd["result"]
                            _ovr = _qt.get("_summary", {}).get(
                                "overall", "") if _qt else ""
                            if _ovr in qt_filter and (
                                    _r["n_frames"] - _r["n_detected"] <= max_interp_sel):
                                auto_sel.append(f"{_fn}/{_cl}")
                    st.session_state.batch_global_csv_sel = auto_sel
                    st.rerun()

            st.markdown("---")

            selected_cells = st.multiselect(
                "Select cells to export",
                options=selection_options,
                default=st.session_state.get(
                    "batch_global_csv_sel", selection_options),
                help="Select the rows to include in the combined CSV.",
                key="batch_global_csv_sel"
            )

            if all_rows:
                def cs2(v):
                    if v == "COMPLETE":
                        return "color:#22c55e;font-weight:bold"
                    if v == "INTERPOLATED":
                        return "color:#f59e0b;font-weight:bold"
                    if v in ("DISCARDED", "ERROR"):
                        return "color:#ef4444;font-weight:bold"
                    return ""

                def cq(v):
                    if v == "PASS":
                        return "color:#22c55e;font-weight:bold"
                    if v == "WARN":
                        return "color:#f59e0b;font-weight:bold"
                    if v == "FAIL":
                        return "color:#ef4444;font-weight:bold"
                    return ""

                df_all = pd.DataFrame(all_rows)
                st.dataframe(
                    df_all.style
                    .applymap(cs2, subset=["Status"])
                    .applymap(cq, subset=["QT"]),
                    use_container_width=True,
                    height=min(400, 60 + 35 * len(all_rows)),
                )

            selected_set = set(selected_cells)
            csv_all = []
            for fname, fr in results.items():
                if fr.get("skipped"):
                    continue
                for clabel, cd in fr["cells"].items():
                    if "error" in cd:
                        continue
                    key = f"{fname}/{clabel}"
                    if selected_set and key not in selected_set:
                        continue
                    prev_xy = None
                    for t in cd["result"]["trajectory"]:
                        speed = np.nan
                        if prev_xy is not None:
                            speed = float(
                                np.hypot(t[1] - prev_xy[0], t[2] - prev_xy[1]))
                        csv_all.append({
                            "field_id": fname,
                            "cell_id": clabel,
                            "frame": t[0],
                            "x_rel_px": round(t[1], 4),
                            "y_rel_px": round(t[2], 4),
                            "intensity": round(t[3], 2) if not np.isnan(t[3]) else np.nan,
                            "interpolated": t[4],
                            "speed_px_frame": round(speed, 4) if not np.isnan(speed) else np.nan,
                        })
                        prev_xy = (t[1], t[2])
            if csv_all:
                st.download_button(
                    "Download global CSV (selected)",
                    data=pd.DataFrame(csv_all).to_csv(
                        index=False).encode("utf-8"),
                    file_name="fields_cells_selected.csv",
                    mime="text/csv",
                )
            if selected_cells:
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    roi_zip = build_selected_cell_rois_zip(
                        selected_cells, results, st.session_state.batch_fields,
                        tail_len, gif_fps)
                    if roi_zip:
                        st.download_button(
                            "Download selected ROI GIFs",
                            data=roi_zip,
                            file_name="selected_cells_rois.zip",
                            mime="application/zip",
                            help="ZIP with annotated ROI GIFs for the selected cells.",
                        )
                    else:
                        st.info("No ROI GIF available for the current selection.")
                with dl_col2:
                    if st.button("Prepare raw TIFFs", key="btn_prep_tif",
                                 use_container_width=True,
                                 help="Generate TIFF crops for the current selection. "
                                      "Only runs when you click — not on every UI change."):
                        with st.spinner(f"Cropping {len(selected_cells)} cell(s)…"):
                            tif_zip = build_raw_tif_zip(
                                selected_cells, results, st.session_state.batch_fields)
                        st.session_state["_raw_tif_zip"] = tif_zip
                        st.session_state["_raw_tif_sel"] = list(selected_cells)

                    cached_zip = st.session_state.get("_raw_tif_zip")
                    cached_sel = st.session_state.get("_raw_tif_sel", [])
                    if cached_zip:
                        if sorted(cached_sel) != sorted(selected_cells):
                            st.caption(
                                "Selection changed — click Prepare to regenerate.")
                        st.download_button(
                            f"Download raw TIFFs ({len(cached_sel)} cell(s))",
                            data=cached_zip,
                            file_name="selected_cells_raw.zip",
                            mime="application/zip",
                            help="Unprocessed TIFF crops (original bit-depth). "
                                 "Open in Fiji / ImageJ for manual QC.",
                            use_container_width=True,
                            key="btn_dl_tif",
                        )

        st.markdown("---")
        valid_fields = [fn for fn, fr in results.items()
                        if not fr.get("skipped")]
        if not valid_fields:
            st.warning("All fields were skipped.")
            st.stop()

        st.markdown('<div class="cbox">', unsafe_allow_html=True)
        fi_names = valid_fields
        fi_idx = fi_names.index(st.session_state.batch_sel_field) \
            if st.session_state.batch_sel_field in fi_names else 0
        fi1, fi2, fi3 = st.columns([5, 1, 1])
        with fi2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("◀", key="fi_p", use_container_width=True, disabled=fi_idx == 0):
                st.session_state.batch_sel_field = fi_names[fi_idx-1]
                st.session_state.batch_sel_cell = next(
                    iter(results[fi_names[fi_idx-1]]["cells"]), None)
                st.rerun()
        with fi3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("▶", key="fi_n", use_container_width=True, disabled=fi_idx == len(fi_names)-1):
                st.session_state.batch_sel_field = fi_names[fi_idx+1]
                st.session_state.batch_sel_cell = next(
                    iter(results[fi_names[fi_idx+1]]["cells"]), None)
                st.rerun()
        with fi1:
            sel_field = st.selectbox(
                f"Field ({len(fi_names)} total)", options=fi_names, index=fi_idx)
            if sel_field != st.session_state.batch_sel_field:
                st.session_state.batch_sel_field = sel_field
                st.session_state.batch_sel_cell = next(
                    iter(results[sel_field]["cells"]), None)
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        fname = st.session_state.batch_sel_field
        fr = results[fname]
        st.markdown(f"### 🔭 Field: `{fname}`")
        dm1, dm2, dm3, dm4 = st.columns(4)
        with dm1:
            st.metric("Detected cells", fr["n_cells_segmented"])
        with dm2:
            st.metric("Tracked cells", len(fr["cells"]))
        with dm3:
            st.metric("Max drift",
                      f"{fr['drift_info']['max_drift_px']:.2f} px")
        with dm4:
            st.metric("Drift action", fr["decision"])
        ov1, ov2 = st.columns(2)
        with ov1:
            if fr["ov_bytes"]:
                st.image(
                    fr["ov_bytes"], caption="Field segmentation", use_container_width=True)
        with ov2:
            if fr["drift_bytes"]:
                st.image(fr["drift_bytes"], caption="Detected drift",
                         use_container_width=True)
            else:
                st.info("No significant drift in this field.")

        st.markdown("---")
        cell_names = list(fr["cells"].keys())
        if not cell_names:
            st.warning("No tracked cells in this field.")
            st.stop()
        ci_idx = cell_names.index(st.session_state.batch_sel_cell) \
            if st.session_state.batch_sel_cell in cell_names else 0
        ci1, ci2, ci3 = st.columns([5, 1, 1])
        with ci2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("◀", key="ci_p", use_container_width=True, disabled=ci_idx == 0):
                st.session_state.batch_sel_cell = cell_names[ci_idx-1]
                st.rerun()
        with ci3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("▶", key="ci_n", use_container_width=True, disabled=ci_idx == len(cell_names)-1):
                st.session_state.batch_sel_cell = cell_names[ci_idx+1]
                st.rerun()
        with ci1:
            sel_cell = st.selectbox(f"Cell in this field ({len(cell_names)} total)",
                                    options=cell_names, index=ci_idx)
            if sel_cell != st.session_state.batch_sel_cell:
                st.session_state.batch_sel_cell = sel_cell
                st.rerun()

        clabel = st.session_state.batch_sel_cell
        cd = fr["cells"][clabel]
        st.markdown(f"### 🧬 `{clabel}`  <small style='color:#555588;'>en {fname} ({ci_idx+1}/{len(cell_names)})</small>",
                    unsafe_allow_html=True)
        if "error" in cd:
            st.error(f"Error in this cell: {cd['error']}")
            st.stop()
        res = cd["result"]
        sd = cd["sd"]
        qt = cd["qt"]
        css_map = {"complete": "sc", "interpolated": "si", "discarded": "sd"}
        icns = {"complete": "✅", "interpolated": "🔄", "discarded": "❌"}
        lbls = {"complete": "COMPLETE trajectory",
                "interpolated": f"INTERPOLATED ({res['missing_ratio']*100:.1f}%)",
                "discarded": f"DISCARDED ({res['missing_ratio']*100:.1f}% missing)"}
        st.markdown(f"<div class='{css_map[res['status']]}' style='margin-bottom:10px'>"
                    f"{icns[res['status']]}  {lbls[res['status']]}</div>", unsafe_allow_html=True)
        bm1, bm2, bm3, bm4, bm5, bm6 = st.columns(6)
        with bm1:
            st.metric("Frames", res["n_frames"])
        with bm2:
            st.metric("Detected", res["n_detected"])
        with bm3:
            st.metric("Interpolated", res["n_frames"]-res["n_detected"])
        with bm4:
            a2 = sd["alpha_msd"] if sd else float("nan")
            st.metric("Alpha α", f"{a2:.2f}" if not np.isnan(a2) else "—")
        with bm5:
            st.metric("Motion type", sd["motion_type"] if sd else "—")
        with bm6:
            ovr = qt.get("_summary", {}).get("overall", "?") if qt else "?"
            col_map2 = {"PASS": "#22c55e",
                        "WARN": "#f59e0b", "FAIL": "#ef4444"}
            oc = col_map2.get(ovr, "#aaa")
            st.markdown(f"<div style='text-align:center;margin-top:8px;'>"
                        f"<span style='font-size:1.3rem;font-weight:700;color:{oc};'>{ovr}</span><br>"
                        f"<span style='color:#888;font-size:0.8rem'>QC overall</span></div>",
                        unsafe_allow_html=True)
        if res["status"] == "discarded":
            st.warning("Cell discarded.")
            render_quality_tests(qt)
            st.stop()
        tb1, tb2, tb3, tb4, tb5, tb6, tb7 = st.tabs([
            "📈 Relative trajectory", "📊 MSD log-log", "🔬 Quality tests",
            "🎬 GIF (ROI)", "🗃️ CSV", "📄 Report", "🗺️ Field"])
        with tb1:
            if cd["tf_b"]:
                st.image(cd["tf_b"], use_container_width=True)
                st.markdown("<small style='color:#666'>Relative to cell centroid · + = centroid · o=start · *=end</small>",
                            unsafe_allow_html=True)
                if sd:
                    st.markdown("---")
                    ca, cb2 = st.columns(2)
                    with ca:
                        st.markdown("**Geometry (relative to centroid)**")
                        st.markdown(
                            f"- Total path length: **{sd['total_path_px']:.2f} px**")
                        st.markdown(
                            f"- End-to-end disp.: **{sd['end_to_end_px']:.2f} px**")
                        st.markdown(
                            f"- Confinement: **{sd['confinement_ratio']:.3f}**")
                        st.markdown(
                            f"- Gyration radius: **{sd['gyration_radius_px']:.2f} px**")
                    with cb2:
                        st.markdown("**Speed**")
                        st.markdown(
                            f"- Mean: **{sd['mean_speed_px_frame']:.4f} px/frame**")
                        st.markdown(
                            f"- Max: **{sd['max_speed_px_frame']:.4f} px/frame**")
                        st.markdown("**Intensity**")
                        st.markdown(f"- Mean: **{sd['mean_intensity']:.1f}**")
                        st.markdown(
                            f"- Std. dev.: **{sd['std_intensity']:.1f}**")
        with tb2:
            mf = make_msd_log(sd)
            st.pyplot(mf, use_container_width=False)
            plt.close(mf)
            mm1, mm2, mm3 = st.columns(3)
            a2 = sd["alpha_msd"]
            D2 = sd["diffusion_coefficient"]
            di2 = sd["directionality_index"]
            with mm1:
                st.metric("Alpha α", f"{a2:.3f}" if not np.isnan(a2) else "—")
            with mm2:
                st.metric("D (px²/frame)",
                          f"{D2:.4f}" if not np.isnan(D2) else "—")
            with mm3:
                st.metric("Direct. index",
                          f"{di2:.3f}" if not np.isnan(di2) else "—")
        with tb3:
            render_quality_tests(qt)
        with tb4:
            if cd["gif"]:
                st.image(cd["gif"], caption=f"{fname} → {clabel} (ROI)")
                st.download_button(f"Download GIF — {fname}_{clabel}", data=cd["gif"],
                                   file_name=f"{fname}_{clabel}_roi.gif", mime="image/gif")
            elif not gen_gif:
                st.info("GIF generation disabled.")
            else:
                st.info("No GIF available.")
        with tb5:
            trows_b = [{"field_id": fname, "cell_id": clabel, "frame": t[0],
                        "x_rel_px": round(t[1], 4), "y_rel_px": round(t[2], 4),
                        "intensity": round(t[3], 2) if not np.isnan(t[3]) else np.nan,
                        "interpolated": t[4]} for t in res["trajectory"]]
            dfc_b = pd.DataFrame(trows_b)
            st.dataframe(dfc_b.style.applymap(
                lambda v: "background-color:#2d2000;color:#f59e0b" if v is True else "",
                subset=["interpolated"]), use_container_width=True, height=380)
            st.download_button(f"Download CSV — {fname}_{clabel}",
                               data=dfc_b.to_csv(index=False).encode("utf-8"),
                               file_name=f"{fname}_{clabel}_trajectory.csv", mime="text/csv")
        with tb6:
            st.code(cd["report"], language=None)
            st.download_button(f"Download report — {fname}_{clabel}",
                               data=cd["report"].encode("utf-8"),
                               file_name=f"{fname}_{clabel}_report.txt", mime="text/plain")
        with tb7:
            if fr["ov_bytes"]:
                st.image(
                    fr["ov_bytes"], caption=f"Field segmentation — {fname}", use_container_width=True)

        st.markdown("---")
        zip1, zip2 = st.columns(2)
        with zip1:
            buf_f = io.BytesIO()
            with zipfile.ZipFile(buf_f, "w", zipfile.ZIP_DEFLATED) as zf:
                if fr["ov_bytes"]:
                    zf.writestr(f"{fname}_overview.png", fr["ov_bytes"])
                fc = []
                for cl2, cd2 in fr["cells"].items():
                    if "error" in cd2:
                        continue
                    if cd2["tf_b"]:
                        zf.writestr(f"{fname}_{cl2}_traj.png", cd2["tf_b"])
                    if cd2["gif"]:
                        zf.writestr(f"{fname}_{cl2}_roi.gif",  cd2["gif"])
                    zf.writestr(f"{fname}_{cl2}_report.txt",
                                cd2["report"].encode("utf-8"))
                    for t in cd2["result"]["trajectory"]:
                        fc.append({"field_id": fname, "cell_id": cl2, "frame": t[0],
                                   "x_rel_px": round(t[1], 4), "y_rel_px": round(t[2], 4),
                                   "intensity": round(t[3], 2) if not np.isnan(t[3]) else np.nan,
                                   "interpolated": t[4]})
                if fc:
                    zf.writestr(f"{fname}_trajectories.csv",
                                pd.DataFrame(fc).to_csv(index=False).encode("utf-8"))
            buf_f.seek(0)
            st.download_button(f"📦 ZIP — field {fname}", data=buf_f.read(),
                               file_name=f"{fname}_results.zip", mime="application/zip",
                               use_container_width=True)
        with zip2:
            buf_all = io.BytesIO()
            ac = []
            with zipfile.ZipFile(buf_all, "w", zipfile.ZIP_DEFLATED) as zf:
                for fn2, fr2 in results.items():
                    if fr2.get("skipped"):
                        continue
                    if fr2["ov_bytes"]:
                        zf.writestr(f"{fn2}/{fn2}_overview.png",
                                    fr2["ov_bytes"])
                    for cl2, cd2 in fr2["cells"].items():
                        if "error" in cd2:
                            continue
                        if cd2["tf_b"]:
                            zf.writestr(
                                f"{fn2}/{fn2}_{cl2}_traj.png", cd2["tf_b"])
                        if cd2["gif"]:
                            zf.writestr(
                                f"{fn2}/{fn2}_{cl2}_roi.gif",  cd2["gif"])
                        zf.writestr(f"{fn2}/{fn2}_{cl2}_report.txt",
                                    cd2["report"].encode("utf-8"))
                        for t in cd2["result"]["trajectory"]:
                            ac.append({"field_id": fn2, "cell_id": cl2, "frame": t[0],
                                       "x_rel_px": round(t[1], 4), "y_rel_px": round(t[2], 4),
                                       "intensity": round(t[3], 2) if not np.isnan(t[3]) else np.nan,
                                       "interpolated": t[4]})
                if ac:
                    zf.writestr("all_fields_cells.csv",
                                pd.DataFrame(ac).to_csv(index=False).encode("utf-8"))
            buf_all.seek(0)
            st.download_button("📦 ZIP — all fields", data=buf_all.read(),
                               file_name="all_fields_results.zip", mime="application/zip",
                               use_container_width=True, type="primary")

    st.markdown("<div style='text-align:center;margin-top:28px;color:#333355;font-size:0.78rem;'>"
                "Particle Tracker · full field · segmentation · drift · relative tracking · quality tests</div>",
                unsafe_allow_html=True)




