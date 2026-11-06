"""
reporting.py - Report generation and quality test rendering for Particle Tracker.

Functions for building formatted text reports and rendering quality test results in Streamlit.
"""

import streamlit as st
import numpy as np
import pandas as pd


def build_report(name, result, sd, sig, params):
    """Build a text report for a single cell trajectory analysis.

    Compiles results and statistics into a formatted text report including
    trajectory status, frame counts, gaps, and motion statistics.

    Args:
        name: Cell identifier string.
        result: Dictionary with 'status', 'n_frames', 'n_detected', 'gap_info'.
        sd: Dictionary with statistics (path length, displacement, alpha_msd, etc.)
            or None if trajectory was discarded.
        sig: Dictionary with signal info or None if signal cutting was not performed.
        params: Dictionary of analysis parameters used.

    Returns:
        str: Formatted text report ready for display or export.
    """
    L = ["="*60, f"  CELL: {name}", "="*60]
    for k, v in params.items():
        L.append(f"  {k:<28}: {v}")
    L += [f"\n  Status: {result['status'].upper()}",
          f"  Analyzed frames: {result['n_frames']}",
          f"  Detected: {result['n_detected']}",
          f"  Interpolated: {result['n_frames']-result['n_detected']}"]
    if sig:
        nf = len(sig["intensity_trace"])
        L.append(f"  Loaded frames: {nf}")
        L.append(
            f"  Signal cut: frame {sig['end_frame']} ({'yes' if sig['auto_cut'] else 'no'})")
    if result.get("gap_info"):
        for g in result["gap_info"]:
            L.append(f"  Gap {g[0]}-{g[1]} ({g[2]} frames)")
    if result["status"] != "discarded" and sd:
        L += [f"\n  Total path length: {sd['total_path_px']:.2f} px",
              f"  End-to-end disp.:  {sd['end_to_end_px']:.2f} px",
              f"  Confinement:       {sd['confinement_ratio']:.3f}",
              f"  Mean speed:        {sd['mean_speed_px_frame']:.4f} px/fr",
              (f"  Alpha MSD:         {sd['alpha_msd']:.4f}" if not np.isnan(
                  sd["alpha_msd"]) else "  Alpha MSD: N/A"),
              f"  Motion type:       {sd['motion_type']}",
              (f"  Diffusion coeff.:  {sd['diffusion_coefficient']:.6f}" if not np.isnan(
                  sd["diffusion_coefficient"]) else "  D: N/A"),
              f"  Mean intensity:    {sd['mean_intensity']:.1f}"]
    L.append("\n"+"="*60)
    return "\n".join(L)


def build_cell_report_batch(field_name, cell_label, result, sd, qt, params):
    """Build a text report for a cell from batch (full field) analysis.

    Similar to build_report but includes quality test results and field information.

    Args:
        field_name: Field identifier string.
        cell_label: Cell identifier string (e.g., 'C1').
        result: Dictionary with 'status', 'n_frames', 'n_detected', 'gap_info'.
        sd: Dictionary with statistics or None if trajectory was discarded.
        qt: Dictionary with quality test results or None.
        params: Dictionary of analysis parameters used.

    Returns:
        str: Formatted text report with quality test summary.
    """
    L = ["="*60, f"  FIELD:  {field_name}", f"  CELL:   {cell_label}", "="*60]
    for k, v in params.items():
        L.append(f"  {k:<28}: {v}")
    L += [f"\n  Status: {result['status'].upper()}",
          f"  Analyzed frames: {result['n_frames']}",
          f"  Detected: {result['n_detected']}",
          f"  Interpolated: {result['n_frames']-result['n_detected']}"]
    if result.get("gap_info"):
        for g in result["gap_info"]:
            L.append(f"  Gap {g[0]}-{g[1]} ({g[2]} frames)")
    if result["status"] != "discarded" and sd:
        L += [f"\n  Total path length: {sd['total_path_px']:.2f} px",
              f"  End-to-end disp.:  {sd['end_to_end_px']:.2f} px",
              f"  Confinement:       {sd['confinement_ratio']:.3f}",
              f"  Mean speed:        {sd['mean_speed_px_frame']:.4f} px/fr",
              (f"  Alpha MSD:         {sd['alpha_msd']:.4f}" if not np.isnan(
                  sd["alpha_msd"]) else "  Alpha MSD: N/A"),
              f"  Motion type:       {sd['motion_type']}",
              (f"  Diffusion coeff.:  {sd['diffusion_coefficient']:.6f}" if not np.isnan(
                  sd["diffusion_coefficient"]) else "  D: N/A"),
              f"  Mean intensity:    {sd['mean_intensity']:.1f}"]
    if qt:
        L.append("\n  Quality tests:")
        for k, v in qt.items():
            if k == "_summary":
                continue
            sym = "OK" if v["pass"] else "!!"
            val = v["value"] if v["value"] is not None else "N/A"
            L.append(f"    [{sym}] {k}: {val}  ({v['label']})")
        s = qt.get("_summary", {})
        L.append(
            f"  Overall: {s.get('overall','?')} ({s.get('n_pass',0)}/{s.get('n_total',0)})")
    L.append("\n"+"="*60)
    return "\n".join(L)


def render_quality_tests(qt):
    """Render quality test results as Streamlit UI components.

    Displays quality test summary and individual test results in a grid layout
    with pass/fail styling.

    Args:
        qt: Dictionary with quality test results. Keys are test names, values are
            dicts with 'pass' (bool), 'value' (float or None), and 'label' (str).
            Special key '_summary' contains overall status and counts.
    """
    if not qt:
        return
    s = qt.get("_summary", {})
    overall = s.get("overall", "?")
    n_pass = s.get("n_pass", 0)
    n_total = s.get("n_total", 0)
    col_map = {"PASS": "#22c55e", "WARN": "#f59e0b", "FAIL": "#ef4444"}
    oc = col_map.get(overall, "#aaa")
    st.markdown(f"<h4 style='color:{oc};margin-bottom:8px'>Quality tests: "
                f"{overall} ({n_pass}/{n_total})</h4>", unsafe_allow_html=True)
    tests_items = [(k, v) for k, v in qt.items() if k != "_summary"]
    n_cols = min(len(tests_items), 3)
    if n_cols == 0:
        return
    cols = st.columns(n_cols)
    for ci, (k, v) in enumerate(tests_items):
        cls = "qt-pass" if v["pass"] else "qt-fail"
        sym = "✅" if v["pass"] else "❌"
        val = f"{v['value']}" if v["value"] is not None else "N/A"
        with cols[ci % n_cols]:
            st.markdown(
                f"<div style='background:#141430;border:1px solid #2a2a6e;"
                f"border-radius:8px;padding:10px;margin-bottom:8px;'>"
                f"<div class='{cls}'>{sym} {k}</div>"
                f"<div style='color:#aaa;font-size:0.82rem'>{v['label']}</div>"
                f"<div style='color:#eee;font-weight:700;font-size:1.05rem'>{val}</div>"
                f"</div>", unsafe_allow_html=True)


# ── DataFrame style helpers ───────────────────────────────────────────────────

def style_status(v: str) -> str:
    """CSS for the Status column (COMPLETE / INTERPOLATED / MISSING)."""
    if v == "COMPLETE":
        return "color:#22c55e;font-weight:bold"
    if v == "INTERPOLATED":
        return "color:#f59e0b;font-weight:bold"
    if v == "MISSING":
        return "color:#ef4444;font-weight:bold"
    return ""


def style_qt(v: str) -> str:
    """CSS for the QT (quality-test) column (PASS / WARN / FAIL)."""
    if v == "PASS":
        return "color:#22c55e;font-weight:bold"
    if v == "WARN":
        return "color:#f59e0b;font-weight:bold"
    if v == "FAIL":
        return "color:#ef4444;font-weight:bold"
    return ""


def style_interp(v: object) -> str:
    """CSS for boolean interpolated columns."""
    return "background-color:#2d2000;color:#f59e0b" if v is True else ""
