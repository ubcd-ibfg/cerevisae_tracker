"""
config.py
=========
Central configuration for Particle Tracker.

All numeric thresholds, display constants, colour palette, figure sizes and
session-state keys live here so they can be changed in one place without
hunting through application code.
"""

from __future__ import annotations

# ── Application metadata ──────────────────────────────────────────────────────
APP_TITLE   = "Particle Tracker"
APP_VERSION = "0.4.0"

# ── Dark-theme colour palette ─────────────────────────────────────────────────
DARK_BG        = "#0d0d1a"   # main canvas background
SIDEBAR_BG     = "#111128"   # sidebar background
PANEL_BG       = "#141430"   # card / metric container background
COLOR_ACCENT   = "#00d4ff"   # primary accent (cyan)
COLOR_WARN     = "#f59e0b"   # warning / interpolated
COLOR_OK       = "#22c55e"   # success / pass
COLOR_ERROR    = "#ef4444"   # error / fail
COLOR_INTERP   = "#7c4dff"   # interpolated points
COLOR_FIT      = "#ff6b6b"   # secondary / fit line
COLOR_DX       = "#00d4ff"   # drift X component
COLOR_DY       = "#ff6b6b"   # drift Y component
COLOR_HIST     = "#7c4dff"   # histogram bars
COLOR_SPINE    = "#333333"   # axis spine
COLOR_TICK     = "#aaaaaa"   # tick labels
COLOR_LABEL    = "#cccccc"   # axis labels
COLOR_TITLE    = "#eeeeee"   # panel titles

# Cluster palette (up to 8 groups)
CLUSTER_PALETTE: list[str] = [
    "#00d4ff", "#ff6b6b", "#ffd700", "#00ff88",
    "#ff88ff", "#ff8800", "#88ffff", "#ff0088",
]

# ── Figure sizes (width, height) in inches ───────────────────────────────────
FIGSIZE_TRAJ_3PANEL    = (17, 5)    # make_traj_fig — 3-panel trajectory figure
FIGSIZE_SIGNAL         = (10, 3.5)  # make_signal_fig — intensity trace
FIGSIZE_MSD_LOG        = (6,  4.5)  # make_msd_log — log-log MSD
FIGSIZE_FIELD_OVERVIEW = (14, 6)    # make_field_overview — segmentation overview
FIGSIZE_DRIFT          = (15, 4)    # make_drift_fig — drift panels

# ── Rendering ─────────────────────────────────────────────────────────────────
FIG_DPI = 150   # export DPI for all PNG outputs

# ── Motion-type classification (MSD power-law exponent α) ────────────────────
ALPHA_SUBDIFFUSIVE   = 0.7   # α < this  → Subdiffusive / confined
ALPHA_SUPERDIFFUSIVE = 1.3   # α < this  → Diffusive (Brownian)
ALPHA_BALLISTIC      = 1.8   # α < this  → Superdiffusive; ≥ → Ballistic / directed

# ── MSD fitting ───────────────────────────────────────────────────────────────
MSD_FIT_LEN  = 8    # maximum number of lag points used for log-log linear fit
MSD_MAX_LAG  = 30   # hard cap on lag (frames) for MSD calculation

# ── Quality-test thresholds ───────────────────────────────────────────────────
QT_MIN_COMPLETENESS   = 0.70  # T1 – fraction of frames with detections
QT_MAX_GAP_FRACTION   = 0.20  # T4 – largest single gap as fraction of trajectory
QT_MIN_MSD_R2         = 0.80  # T5 – log-log MSD fit R²
QT_MAX_INTENSITY_CV   = 0.60  # T6 – intensity coefficient of variation
QT_MIN_SNR            = 3.0   # T2 – signal-to-noise ratio (ref / threshold)
QT_WARN_PASS_FRACTION = 0.60  # minimum fraction of tests passing for WARN (vs FAIL)

# ── Segmentation & tracking defaults ─────────────────────────────────────────
DEFAULT_BBOX_PAD      = 12    # pixel padding added around cell bbox during tracking
DEFAULT_N_REF_FRAMES  = 5     # frames used to build the reference (median) projection
DRIFT_SLOPE_THRESHOLD = 0.05  # px/frame — minimum slope considered systematic drift

# ── Streamlit session-state keys ─────────────────────────────────────────────
SK_CELLS           = "cells"
SK_SEL             = "sel"
SK_BATCH_PHASE     = "batch_phase"
SK_BATCH_FIELDS    = "batch_fields"
SK_BATCH_RESULTS   = "batch_results"
SK_BATCH_SEL_FIELD = "batch_sel_field"
SK_BATCH_SEL_CELL  = "batch_sel_cell"
SK_BATCH_CSV_SEL   = "batch_global_csv_sel"
SK_RAW_TIF_ZIP     = "_raw_tif_zip"
SK_RAW_TIF_SEL     = "_raw_tif_sel"


def cell_include_key(field_name: str, clabel: str) -> str:
    """Session-state key controlling whether a cell is included in tracking."""
    return f"cell_inc_{field_name}_{clabel}"


def drift_radio_key(field_name: str) -> str:
    """Session-state key for the per-field drift-correction radio button."""
    return f"dr_{field_name}"
