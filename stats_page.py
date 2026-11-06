"""
stats_page.py  –  Statistical comparison of movement between two conditions.
Input: CSV files produced by the tracker (batch mode output).
"""

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

TEST_FILES_DIR = Path(__file__).parent / "test_files"

METRICS_LABELS = {
    # Speed / kinetics
    "mean_speed":      "Mean speed (px/frame)",
    "speed_std":       "Speed variability (std, px/frame)",
    "max_speed":       "Max speed (px/frame)",
    # Geometry
    "confinement":     "Confinement ratio",
    "gyration_radius": "Gyration radius (px)",
    "directionality":  "Angular persistence (cos turning)",
    # MSD
    "alpha_msd":       "MSD exponent (α)  [R²≥0.5 only]",
}
METRIC_COLS = list(METRICS_LABELS.keys())

# Threshold below which alpha_msd is treated as NaN (unreliable log-log fit)
ALPHA_R2_MIN = 0.5

ALL_FEATURES = {
    "mean_speed":      "Mean speed (px/frame)",
    "speed_std":       "Speed variability (std)",
    "straightness":    "Straightness (displacement/path)",
    "ang_persistence": "Angular persistence (cos turning)",
    "alpha_local":     "Local MSD exponent (α)",
}

DARK_BG  = "#0d0d1a"
COLOR_A  = "#00d4ff"
COLOR_B  = "#ff6b6b"
CLUSTER_PALETTE = [
    "#00d4ff", "#ff6b6b", "#ffd700", "#00ff88",
    "#ff88ff", "#ff8800", "#88ffff", "#ff0088",
]


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _dark_ax(ax):
    ax.set_facecolor(DARK_BG)
    for sp in ax.spines.values():
        sp.set_color("#333")
    ax.tick_params(colors="#aaa")
    ax.xaxis.label.set_color("#ccc")
    ax.yaxis.label.set_color("#ccc")
    ax.title.set_color("#ddd")


def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


# ── DATA PROCESSING ───────────────────────────────────────────────────────────

def _load_csv(source) -> pd.DataFrame:
    df = pd.read_csv(source)

    # Mode A normalisation: has x_px/y_px but not the Mode B column layout
    if "x_px" in df.columns and "x_rel_px" not in df.columns:
        df = df.rename(columns={"x_px": "x_rel_px", "y_px": "y_rel_px"})
        df["field_id"] = "single"
        df = df.sort_values(["cell_id", "frame"]).reset_index(drop=True)
        speed_col = np.full(len(df), np.nan)
        for _, g in df.groupby("cell_id", sort=False):
            idx = g.index.values
            dx = np.diff(g["x_rel_px"].values)
            dy = np.diff(g["y_rel_px"].values)
            speed_col[idx[1:]] = np.hypot(dx, dy)
        df["speed_px_frame"] = speed_col

    required = {"field_id", "cell_id", "frame", "x_rel_px", "y_rel_px", "speed_px_frame"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")
    df["speed_px_frame"] = pd.to_numeric(df["speed_px_frame"], errors="coerce")
    if "interpolated" not in df.columns:
        df["interpolated"] = False
    return df


def _compute_cell_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per unique (field_id, cell_id) with summary metrics.

    Speeds and MSD use only pairs where BOTH endpoints are real detections
    (``interpolated`` is False on both rows). Interpolated points are kept
    for total_path / end-to-end / confinement (geometric measures on the
    smoothed trajectory) but excluded from kinetic statistics to avoid the
    bias that cubic interpolation introduces (smoother → α ↓, speed ↓).
    """
    records = []
    for (fid, cid), g in df.groupby(["field_id", "cell_id"]):
        g = g.sort_values("frame").reset_index(drop=True)
        speeds_all = g["speed_px_frame"].dropna()
        if len(speeds_all) < 2:
            continue

        xs = g["x_rel_px"].values.astype(float)
        ys = g["y_rel_px"].values.astype(float)
        is_real = (~g["interpolated"].astype(bool)).values
        n = len(g)

        # Real-pair step mask (both endpoints real)
        if len(is_real) > 1:
            real_step = is_real[:-1] & is_real[1:]
        else:
            real_step = np.zeros(0, dtype=bool)

        speeds_full = g["speed_px_frame"].values.astype(float)
        # speed_px_frame at index i corresponds to the step (i-1 -> i);
        # ignore index 0 (NaN) and align with real_step.
        step_speeds = speeds_full[1:]
        speeds_real = step_speeds[real_step] if real_step.any() else speeds_all.values

        total_path = float(np.nansum(speeds_full))  # smoothed geometric path
        e2e = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
        confinement = e2e / total_path if total_path > 0 else np.nan

        # Speed summary
        if len(speeds_real) > 0:
            mean_speed = float(np.nanmean(speeds_real))
            max_speed  = float(np.nanmax(speeds_real))
            speed_std  = float(np.nanstd(speeds_real))
        else:
            mean_speed = float(speeds_all.mean())
            max_speed  = float(speeds_all.max())
            speed_std  = float(speeds_all.std())

        # Angular persistence (mean cosine of consecutive turning angles, real steps only)
        dx_seg = np.diff(xs)
        dy_seg = np.diff(ys)
        if real_step.any() and real_step.sum() > 1:
            dx_r, dy_r = dx_seg[real_step], dy_seg[real_step]
        else:
            dx_r, dy_r = dx_seg, dy_seg
        if len(dx_r) >= 2:
            dots  = dx_r[:-1] * dx_r[1:] + dy_r[:-1] * dy_r[1:]
            norms = np.hypot(dx_r[:-1], dy_r[:-1]) * np.hypot(dx_r[1:], dy_r[1:])
            valid = norms > 1e-9
            directionality = float(np.mean(np.clip(dots[valid] / norms[valid], -1, 1))) \
                             if valid.any() else np.nan
        else:
            directionality = np.nan

        # Gyration radius
        cm_x, cm_y = float(np.mean(xs)), float(np.mean(ys))
        gyration_radius = float(np.sqrt(np.mean((xs - cm_x)**2 + (ys - cm_y)**2)))

        # Time-averaged MSD: only real-pair lags
        max_lag = max(3, n // 4)
        lags, msd_vals = [], []
        for lag in range(1, max_lag + 1):
            real_pair = is_real[:-lag] & is_real[lag:]
            d2 = (xs[lag:] - xs[:-lag])**2 + (ys[lag:] - ys[:-lag])**2
            if real_pair.any():
                msd_vals.append(float(np.mean(d2[real_pair])))
            else:
                msd_vals.append(float(np.mean(d2)))
            lags.append(lag)

        # Log-log MSD fit; only accept alpha when R² >= ALPHA_R2_MIN
        alpha, alpha_r2 = np.nan, np.nan
        if len(lags) >= 3:
            try:
                log_lags = np.log(lags)
                log_msd  = np.log(np.clip(msd_vals, 1e-9, None))
                slope, intercept = np.polyfit(log_lags, log_msd, 1)
                resid  = log_msd - (slope * log_lags + intercept)
                ss_res = float(np.sum(resid**2))
                ss_tot = float(np.sum((log_msd - log_msd.mean())**2))
                alpha_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
                # Discard alpha when the log-log fit is unreliable
                alpha = float(slope) if (not np.isnan(alpha_r2) and alpha_r2 >= ALPHA_R2_MIN) \
                        else np.nan
            except Exception:
                pass

        records.append({
            "field_id":        fid,
            "cell_id":         cid,
            "n_frames":        n,
            "mean_speed":      mean_speed,
            "speed_std":       speed_std,
            "max_speed":       max_speed,
            "total_path":      total_path,
            "end_to_end":      e2e,
            "confinement":     confinement,
            "gyration_radius": gyration_radius,
            "directionality":  directionality,
            "alpha_msd":       alpha,
            "alpha_r2":        alpha_r2,
            "pct_interp":      float(g["interpolated"].astype(bool).mean() * 100),
        })
    return pd.DataFrame(records)


def _compute_ensemble_msd(df: pd.DataFrame, max_lag_frac: float = 0.25) -> pd.DataFrame:
    """Ensemble-averaged MSD: pool real-pair squared displacements per lag.

    Pairs where either endpoint is an interpolated point are excluded — they
    are linear/cubic estimates of unobserved positions and would bias the
    pooled MSD downward.
    """
    rows = []
    for (_, _), g in df.groupby(["field_id", "cell_id"]):
        g  = g.sort_values("frame").reset_index(drop=True)
        xs = g["x_rel_px"].values.astype(float)
        ys = g["y_rel_px"].values.astype(float)
        is_real = (~g["interpolated"].astype(bool)).values
        max_lag = max(3, int(len(g) * max_lag_frac))
        for lag in range(1, max_lag + 1):
            real_pair = is_real[:-lag] & is_real[lag:]
            d2 = (xs[lag:] - xs[:-lag])**2 + (ys[lag:] - ys[:-lag])**2
            if real_pair.any():
                for v in d2[real_pair]:
                    rows.append({"lag": lag, "msd": float(v)})
    if not rows:
        return pd.DataFrame(columns=["lag", "mean", "sem"])
    tmp = pd.DataFrame(rows)
    return tmp.groupby("lag")["msd"].agg(mean="mean", sem="sem").reset_index()


def _match_n(ma: pd.DataFrame, mb: pd.DataFrame, n: int, seed: int = 42):
    rng  = np.random.default_rng(seed)
    ia   = rng.choice(len(ma), size=min(n, len(ma)), replace=False)
    ib   = rng.choice(len(mb), size=min(n, len(mb)), replace=False)
    return ma.iloc[ia].reset_index(drop=True), mb.iloc[ib].reset_index(drop=True)


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. NaN inputs are preserved as NaN."""
    n = len(pvals)
    result = np.full(n, np.nan)
    valid = ~np.isnan(pvals)
    m = int(valid.sum())
    if m == 0:
        return result
    valid_idx = np.where(valid)[0]
    p_valid = pvals[valid_idx]
    order = np.argsort(p_valid)
    adjusted = p_valid[order] * m / (np.arange(1, m + 1))
    for i in range(m - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    adjusted = np.clip(adjusted, 0.0, 1.0)
    adj_reordered = np.empty(m)
    adj_reordered[order] = adjusted
    result[valid_idx] = adj_reordered
    return result


def _effect_size_r(U: float, n_a: int, n_b: int) -> float:
    """Rank-biserial correlation r for Mann-Whitney U.

    r = 1 - 2U/(n_a·n_b)  ∈ [-1, 1]
    Positive → group A tends to be larger.
    Conventions: |r| < 0.1 negligible, 0.1–0.3 small, 0.3–0.5 medium, > 0.5 large.
    """
    denom = n_a * n_b
    return float(1 - 2 * U / denom) if denom > 0 else np.nan


def _r_label(r: float) -> str:
    if np.isnan(r):
        return "—"
    ar = abs(r)
    if ar < 0.10:
        return "negligible"
    if ar < 0.30:
        return "small"
    if ar < 0.50:
        return "medium"
    return "large"


def _run_tests(ma: pd.DataFrame, mb: pd.DataFrame,
               name_a: str, name_b: str) -> pd.DataFrame:
    """Mann-Whitney U per metric with BH-FDR correction and rank-biserial r."""
    rows = []
    raw_pvals = []
    valid_idx = []

    for i, m in enumerate(METRIC_COLS):
        a = ma[m].dropna().values
        b = mb[m].dropna().values
        label = METRICS_LABELS[m]
        if len(a) < 3 or len(b) < 3:
            rows.append({
                "Metric": label,
                f"Median {name_a}": np.nan, f"Median {name_b}": np.nan,
                "p-value": np.nan, "q (FDR)": np.nan,
                "r (effect)": np.nan, "Effect size": "—",
                "Sig. q<0.05": "—",
            })
            continue
        U, pval = stats.mannwhitneyu(a, b, alternative="two-sided")
        r = _effect_size_r(float(U), len(a), len(b))
        rows.append({
            "Metric":             label,
            f"Median {name_a}":   round(float(np.median(a)), 4),
            f"Median {name_b}":   round(float(np.median(b)), 4),
            "p-value":            round(float(pval), 4),
            "q (FDR)":            np.nan,   # filled below
            "r (effect)":         round(r, 3),
            "Effect size":        _r_label(r),
            "Sig. q<0.05":        "—",
        })
        raw_pvals.append(float(pval))
        valid_idx.append(len(rows) - 1)

    # Apply BH correction only to rows that have a valid p-value
    if raw_pvals:
        q_vals = _benjamini_hochberg(np.array(raw_pvals))
        for j, idx in enumerate(valid_idx):
            q = float(q_vals[j])
            rows[idx]["q (FDR)"]   = round(q, 4)
            rows[idx]["Sig. q<0.05"] = "✓" if q < 0.05 else "✗"

    return pd.DataFrame(rows)


# ── FIGURES ───────────────────────────────────────────────────────────────────

def _make_violin_grid(ma: pd.DataFrame, mb: pd.DataFrame,
                      name_a: str, name_b: str) -> plt.Figure:
    n_cols = 3
    n_rows = (len(METRIC_COLS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.5 * n_rows))
    fig.patch.set_facecolor(DARK_BG)
    axes = np.array(axes).flatten()

    pairs = [(ma, name_a, COLOR_A), (mb, name_b, COLOR_B)]

    for i, metric in enumerate(METRIC_COLS):
        ax = axes[i]
        _dark_ax(ax)
        for j, (mdf, label, color) in enumerate(pairs):
            vals = mdf[metric].dropna().values
            if len(vals) < 2:
                continue
            parts = ax.violinplot([vals], positions=[j], showmedians=True, showextrema=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.5)
            for key in ("cmedians", "cmaxes", "cmins", "cbars"):
                if key in parts:
                    parts[key].set_color("#dddddd")
                    parts[key].set_linewidth(1.2)
            # Individual points (strip plot)
            rng = np.random.default_rng(i * 10 + j)
            jitter = rng.uniform(-0.07, 0.07, len(vals))
            ax.scatter(j + jitter, vals, color=color, alpha=0.65, s=16, zorder=3)

        ax.set_xticks([0, 1])
        ax.set_xticklabels([name_a, name_b], color="#cccccc", fontsize=10)
        ax.set_title(METRICS_LABELS[metric], fontsize=10, color="#dddddd", pad=5)
        ax.grid(axis="y", color="#1a1a3a", linewidth=0.8, zorder=0)

    for ax in axes[len(METRIC_COLS):]:
        ax.set_visible(False)

    fig.tight_layout(pad=2.0)
    return fig


def _make_msd_figure(msd_a: pd.DataFrame, msd_b: pd.DataFrame,
                     name_a: str, name_b: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(DARK_BG)
    _dark_ax(ax)

    for msd_df, name, color in [(msd_a, name_a, COLOR_A), (msd_b, name_b, COLOR_B)]:
        if msd_df.empty:
            continue
        lags = msd_df["lag"].values
        mean = msd_df["mean"].values
        sem  = msd_df["sem"].fillna(0).values
        ax.loglog(lags, mean, color=color, linewidth=2.2, label=name)
        ax.fill_between(lags,
                        np.clip(mean - sem, 1e-9, None),
                        mean + sem,
                        color=color, alpha=0.18)

    # Reference slopes
    lags_all = np.concatenate([msd_a["lag"].values, msd_b["lag"].values])
    x_ref = np.array([1, int(lags_all.max())]) if len(lags_all) else np.array([1, 10])
    means_all = np.concatenate([msd_a["mean"].values, msd_b["mean"].values])
    anchor = float(means_all.min()) * 0.4 if len(means_all) else 1.0
    ax.loglog(x_ref, anchor * x_ref**1, "--", color="#666688", linewidth=1.1, label="α=1 (diffusive)")
    ax.loglog(x_ref, anchor * x_ref**2, ":",  color="#666688", linewidth=1.1, label="α=2 (ballistic)")

    ax.set_xlabel("Lag (frames)", color="#cccccc", fontsize=11)
    ax.set_ylabel("MSD (px²)",    color="#cccccc", fontsize=11)
    ax.set_title("Ensemble MSD comparison (mean ± SEM)", color="#dddddd", fontsize=12)
    ax.legend(framealpha=0.15, facecolor="#111122", edgecolor="#333",
              labelcolor="#cccccc", fontsize=9)
    ax.grid(True, which="both", color="#1a1a3a", linewidth=0.7)
    fig.tight_layout()
    return fig


# ── PATTERN DETECTION ────────────────────────────────────────────────────────

def _extract_windows(df: pd.DataFrame, window_size: int, stride: int) -> pd.DataFrame:
    """Slide a window over each trajectory and extract movement features."""
    rows = []
    for (fid, cid), g in df.groupby(["field_id", "cell_id"]):
        g  = g.sort_values("frame").reset_index(drop=True)
        xs = g["x_rel_px"].values.astype(float)
        ys = g["y_rel_px"].values.astype(float)
        # Recompute step sizes from coordinates to be consistent
        steps = np.concatenate([[0.0], np.hypot(np.diff(xs), np.diff(ys))])
        n = len(g)

        for start in range(0, n - window_size + 1, stride):
            end  = start + window_size
            w_xs = xs[start:end]
            w_ys = ys[start:end]
            w_sp = steps[start:end]

            valid_sp   = w_sp[1:]           # first step undefined
            mean_speed = float(np.mean(valid_sp))
            speed_std  = float(np.std(valid_sp))

            dx_seg   = np.diff(w_xs)
            dy_seg   = np.diff(w_ys)
            path_len = float(np.sum(np.hypot(dx_seg, dy_seg)))
            e2e      = float(np.hypot(w_xs[-1] - w_xs[0], w_ys[-1] - w_ys[0]))
            straightness = e2e / path_len if path_len > 1e-9 else 0.0

            # Mean cosine of consecutive turning angles
            if len(dx_seg) >= 2:
                dots  = dx_seg[:-1] * dx_seg[1:] + dy_seg[:-1] * dy_seg[1:]
                norms = (np.hypot(dx_seg[:-1], dy_seg[:-1]) *
                         np.hypot(dx_seg[1:],  dy_seg[1:]))
                valid = norms > 1e-9
                ang_pers = float(np.mean(np.clip(dots[valid] / norms[valid], -1, 1))) \
                           if valid.any() else 0.0
            else:
                ang_pers = 0.0

            # Local MSD exponent (only meaningful for window >= 5)
            alpha_loc = np.nan
            if window_size >= 5:
                max_lag = max(2, window_size // 4)
                lags, msd_v = [], []
                for lag in range(1, max_lag + 1):
                    d2 = (w_xs[lag:] - w_xs[:-lag])**2 + (w_ys[lag:] - w_ys[:-lag])**2
                    msd_v.append(float(np.mean(d2)))
                    lags.append(lag)
                if len(lags) >= 2:
                    try:
                        alpha_loc = float(
                            np.polyfit(np.log(lags),
                                       np.log(np.clip(msd_v, 1e-9, None)), 1)[0])
                    except Exception:
                        pass

            rows.append({
                "field_id":        fid,
                "cell_id":         cid,
                "start_frame":     int(g["frame"].iloc[start]),
                "end_frame":       int(g["frame"].iloc[end - 1]),
                "mean_speed":      mean_speed,
                "speed_std":       speed_std,
                "straightness":    straightness,
                "ang_persistence": ang_pers,
                "alpha_local":     alpha_loc,
                "xs":              w_xs.tolist(),
                "ys":              w_ys.tolist(),
            })
    return pd.DataFrame(rows)


def _cluster_windows(df_w: pd.DataFrame, feature_cols: list, k: int, seed: int):
    """K-means on selected features; returns df with 'cluster' column."""
    from scipy.cluster.vq import kmeans2

    X      = df_w[feature_cols].copy()
    valid  = X.dropna()
    valid_idx = valid.index
    X_arr  = valid.values.astype(float)

    # Whiten manually (safer than scipy.whiten when a column is constant)
    stds = X_arr.std(axis=0)
    stds = np.where(stds < 1e-9, 1.0, stds)
    X_w  = X_arr / stds

    np.random.seed(seed)
    best_labels, best_inertia = None, np.inf
    for _ in range(15):
        try:
            centroids, labels = kmeans2(X_w, k, iter=50, minit="points")
            inertia = sum(
                float(np.sum((X_w[labels == j] - centroids[j]) ** 2))
                for j in range(k)
            )
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels  = labels.copy()
        except Exception:
            pass

    # Use positional indexing (.iloc) so that pandas Index types don't cause issues
    labels_out = np.full(len(df_w), np.nan)
    if best_labels is not None:
        pos = np.searchsorted(df_w.index, valid_idx)
        labels_out[pos] = best_labels.astype(float)
    result = df_w.copy()
    result["cluster"] = labels_out
    return result, stds


def _make_cluster_profiles(df_w: pd.DataFrame, feature_cols: list, k: int) -> plt.Figure:
    """One bar-chart panel per cluster showing normalised mean feature values."""
    # Normalise each feature to [0, 1] across all windows for visual comparison
    df_n = df_w[feature_cols + ["cluster"]].copy()
    for col in feature_cols:
        mn, mx = df_n[col].min(), df_n[col].max()
        df_n[col] = (df_n[col] - mn) / (mx - mn) if mx > mn else 0.5

    df_n["cluster"] = df_n["cluster"].dropna().astype(int)
    df_n = df_n.dropna(subset=["cluster"])
    cluster_means = df_n.groupby("cluster")[feature_cols].mean()

    fig, axes = plt.subplots(1, k, figsize=(3.5 * k, 4.2))
    fig.patch.set_facecolor(DARK_BG)
    if k == 1:
        axes = [axes]

    short_labels = [ALL_FEATURES.get(c, c) for c in feature_cols]

    for j in range(k):
        ax    = axes[j]
        color = CLUSTER_PALETTE[j % len(CLUSTER_PALETTE)]
        _dark_ax(ax)
        if j in cluster_means.index:
            vals = [cluster_means.loc[j, c] for c in feature_cols]
        else:
            vals = [0.0] * len(feature_cols)
        ax.barh(short_labels, vals, color=color, alpha=0.75, edgecolor="#111")
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="#444466", linewidth=0.8, linestyle="--")
        n_win = int((df_w["cluster"] == float(j)).sum())
        ax.set_title(f"Cluster {j}", color=color, fontsize=11, fontweight="bold")
        ax.set_xlabel(f"N = {n_win} windows", color="#aaa", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle("Cluster profiles  (features normalised to [0, 1])",
                 color="#dddddd", fontsize=11)
    fig.tight_layout(pad=1.8)
    return fig


def _make_frequency_figure(wins_a: pd.DataFrame, wins_b: pd.DataFrame,
                           k: int, name_a: str, name_b: str):
    """Stacked bar of cluster proportions + chi-square result."""
    from scipy.stats import chi2_contingency

    cl_a = wins_a["cluster"].dropna().astype(int)
    cl_b = wins_b["cluster"].dropna().astype(int)
    counts_a = cl_a.value_counts().reindex(range(k), fill_value=0)
    counts_b = cl_b.value_counts().reindex(range(k), fill_value=0)

    contingency = np.vstack([counts_a.values, counts_b.values])
    try:
        chi2, pval, _, _ = chi2_contingency(contingency)
    except Exception:
        chi2, pval = np.nan, np.nan

    prop_a = counts_a.values / counts_a.sum() if counts_a.sum() > 0 else np.zeros(k)
    prop_b = counts_b.values / counts_b.sum() if counts_b.sum() > 0 else np.zeros(k)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.patch.set_facecolor(DARK_BG)
    _dark_ax(ax)

    bot_a = bot_b = 0.0
    for j in range(k):
        color = CLUSTER_PALETTE[j % len(CLUSTER_PALETTE)]
        ax.bar([0], [prop_a[j]], bottom=[bot_a], color=color, alpha=0.82,
               width=0.45, edgecolor="#111", label=f"C{j}")
        ax.bar([1], [prop_b[j]], bottom=[bot_b], color=color, alpha=0.82,
               width=0.45, edgecolor="#111")
        # Value labels for clusters > 5 %
        if prop_a[j] > 0.05:
            ax.text(0, bot_a + prop_a[j] / 2, f"C{j}", ha="center",
                    va="center", color="#000", fontsize=8, fontweight="bold")
        if prop_b[j] > 0.05:
            ax.text(1, bot_b + prop_b[j] / 2, f"C{j}", ha="center",
                    va="center", color="#000", fontsize=8, fontweight="bold")
        bot_a += prop_a[j]
        bot_b += prop_b[j]

    ax.set_xticks([0, 1])
    ax.set_xticklabels([name_a, name_b], color="#cccccc", fontsize=11)
    ax.set_ylabel("Proportion of windows", color="#cccccc")
    ax.set_ylim(0, 1)
    ax.set_title("Cluster usage per condition", color="#dddddd", fontsize=12)
    ax.legend(title="Cluster", framealpha=0.2, facecolor="#111",
              edgecolor="#333", labelcolor="#ccc", fontsize=8, title_fontsize=8)

    sig_color = "#22c55e" if (not np.isnan(pval) and pval < 0.05) else "#ef4444"
    p_txt = f"p = {pval:.4f}" if not np.isnan(pval) else "p = N/A"
    chi_txt = f"χ² = {chi2:.2f}" if not np.isnan(chi2) else "χ² = N/A"
    ax.text(1.35, 0.95, chi_txt, transform=ax.transAxes,
            color="#aaaaaa", fontsize=10, va="top", ha="right")
    ax.text(1.35, 0.86, p_txt, transform=ax.transAxes,
            color=sig_color, fontsize=11, fontweight="bold", va="top", ha="right")

    fig.tight_layout()
    return fig, pval


def _make_segment_examples(wins_a: pd.DataFrame, wins_b: pd.DataFrame,
                           k: int, name_a: str, name_b: str,
                           n_examples: int = 6) -> plt.Figure:
    """Overlay normalised trajectory segments per cluster, coloured by condition."""
    fig, axes = plt.subplots(1, k, figsize=(3.5 * k, 4.0))
    fig.patch.set_facecolor(DARK_BG)
    if k == 1:
        axes = [axes]

    for j in range(k):
        ax    = axes[j]
        color = CLUSTER_PALETTE[j % len(CLUSTER_PALETTE)]
        _dark_ax(ax)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"Cluster {j}", color=color, fontsize=11, fontweight="bold")
        ax.axhline(0, color="#1a1a3a", linewidth=0.7)
        ax.axvline(0, color="#1a1a3a", linewidth=0.7)

        for wins, cond_color, ls in [
            (wins_a, COLOR_A, "-"),
            (wins_b, COLOR_B, "--"),
        ]:
            sub = wins[wins["cluster"] == float(j)]
            sample = sub.sample(min(n_examples, len(sub)), random_state=j)
            for _, row in sample.iterrows():
                rxs = np.array(row["xs"]) - row["xs"][0]
                rys = np.array(row["ys"]) - row["ys"][0]
                ax.plot(rxs, rys, color=cond_color, alpha=0.55, linewidth=1.2,
                        linestyle=ls)
                ax.plot(rxs[0],  rys[0],  "o", color=cond_color, markersize=4, alpha=0.8)
                ax.plot(rxs[-1], rys[-1], "s", color=cond_color, markersize=4, alpha=0.8)

        if j == 0:
            ax.plot([], [], "-",  color=COLOR_A, label=name_a, linewidth=1.5)
            ax.plot([], [], "--", color=COLOR_B, label=name_b, linewidth=1.5)
            ax.legend(framealpha=0.2, facecolor="#111", edgecolor="#333",
                      labelcolor="#ccc", fontsize=8)

    fig.suptitle("Example segments per cluster  (normalised to start at origin)",
                 color="#dddddd", fontsize=10)
    fig.tight_layout(pad=1.6)
    return fig


# ── CELL-LEVEL SUBPOPULATION ANALYSIS ────────────────────────────────────────

# Features used for cell-level clustering (must be columns in _compute_cell_metrics output)
CELL_CLUSTER_FEATURES = {
    "mean_speed":      "Mean speed",
    "speed_std":       "Speed variability",
    "confinement":     "Confinement",
    "gyration_radius": "Gyration radius",
    "directionality":  "Angular persistence",
    "alpha_msd":       "MSD exponent α",
}


def _pca2(X: np.ndarray) -> np.ndarray:
    """Return first 2 principal components of X (rows = samples, cols = features)."""
    X_c = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    return X_c @ Vt[:2].T   # (n, 2)


def _silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient (brute force, OK for n_cells < 5000)."""
    n = len(X)
    unique = np.unique(labels)
    if len(unique) < 2 or n < 4:
        return np.nan
    scores = []
    for i in range(n):
        same  = labels == labels[i]
        same[i] = False
        if same.sum() == 0:
            scores.append(0.0)
            continue
        a = float(np.mean(np.linalg.norm(X[same] - X[i], axis=1)))
        b_vals = []
        for cl in unique:
            if cl == labels[i]:
                continue
            other = X[labels == cl]
            if len(other) == 0:
                continue
            b_vals.append(float(np.mean(np.linalg.norm(other - X[i], axis=1))))
        b = min(b_vals) if b_vals else a
        scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(scores))


def _kmeans_cell(X: np.ndarray, k: int, seed: int, n_init: int = 15):
    """K-means with multiple restarts; returns (labels, inertia)."""
    from scipy.cluster.vq import kmeans2
    best_labels, best_inertia = None, np.inf
    rng = np.random.default_rng(seed)
    for _ in range(n_init):
        try:
            centroids, labels = kmeans2(X, k, iter=100, minit="points",
                                        seed=int(rng.integers(0, 99999)))
            inertia = float(sum(
                np.sum((X[labels == j] - centroids[j]) ** 2)
                for j in range(k)
            ))
            if inertia < best_inertia:
                best_inertia, best_labels = inertia, labels.copy()
        except Exception:
            pass
    return best_labels, best_inertia


def render_cell_subpop_section(ma_full: pd.DataFrame, mb_full: pd.DataFrame,
                               name_a: str, name_b: str):
    """Cell-level subpopulation analysis: PCA + K-means + chi-square."""
    st.markdown("---")
    st.markdown("""<div class="hbox" style="margin-top:8px;">
    <h2 style="margin:0;color:#ffd700;font-size:1.5rem;">
    🧩 Cell subpopulation analysis
    <span style="font-size:0.9rem;color:#555588;">— cell-level clustering</span></h2>
    <p style="margin:6px 0 0;color:#8888cc;font-size:0.85rem;">
    Each <em>cell</em> is a data point (not a window). K-means groups cells by movement
    profile; chi-square tests whether the mixture of subpopulations differs between
    conditions — even when population medians look identical.
    </p></div>""", unsafe_allow_html=True)

    # Feature selection
    st.markdown("#### Features for clustering")
    feat_cols_available = [f for f in CELL_CLUSTER_FEATURES
                           if f in ma_full.columns and f in mb_full.columns]
    sel_cols = st.columns(len(feat_cols_available))
    selected_feats = []
    for i, feat in enumerate(feat_cols_available):
        if sel_cols[i].checkbox(CELL_CLUSTER_FEATURES[feat], value=True,
                                key=f"cc_feat_{feat}"):
            selected_feats.append(feat)
    if len(selected_feats) < 2:
        st.warning("Select at least 2 features.")
        return

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        k_max = st.slider("Max K to evaluate", 2, 8, 6, key="cc_kmax")
    with c2:
        cc_seed = st.number_input("Seed", 0, 9999, 42, key="cc_seed")
    with c3:
        auto_k = st.checkbox("Auto K (max silhouette)", value=True, key="cc_autok")

    if not st.button("Run subpopulation analysis", type="primary",
                     use_container_width=True, key="cc_run"):
        return

    # Build combined feature matrix (drop rows with any NaN in selected features)
    combined = pd.concat([
        ma_full.assign(condition=name_a),
        mb_full.assign(condition=name_b),
    ], ignore_index=True)
    combined = combined.dropna(subset=selected_feats).reset_index(drop=True)
    if len(combined) < 4:
        st.error("Not enough valid cells after dropping NaN values.")
        return

    X_raw = combined[selected_feats].values.astype(float)
    stds  = X_raw.std(axis=0)
    stds  = np.where(stds < 1e-9, 1.0, stds)
    X_z   = (X_raw - X_raw.mean(axis=0)) / stds   # z-scored

    # ── Silhouette curve to choose K ─────────────────────────────────────────
    with st.spinner("Evaluating K…"):
        sil_scores = {}
        for k in range(2, k_max + 1):
            lbls, _ = _kmeans_cell(X_z, k, int(cc_seed))
            if lbls is not None:
                sil_scores[k] = _silhouette(X_z, lbls)

    best_k = max(sil_scores, key=sil_scores.get) if sil_scores else 2
    chosen_k = best_k if auto_k else None

    fig_sil, ax_sil = plt.subplots(figsize=(5, 2.8))
    fig_sil.patch.set_facecolor(DARK_BG); _dark_ax(ax_sil)
    ks = sorted(sil_scores); svs = [sil_scores[k] for k in ks]
    ax_sil.plot(ks, svs, "o-", color=COLOR_A, linewidth=2, markersize=6)
    if auto_k:
        ax_sil.axvline(best_k, color="#ffd700", linewidth=1.5, linestyle="--",
                       label=f"Best K = {best_k}")
        ax_sil.legend(fontsize=8, facecolor="#111", labelcolor="white")
    ax_sil.set_xlabel("K", fontsize=10); ax_sil.set_ylabel("Silhouette", fontsize=10)
    ax_sil.set_title("Silhouette score vs K (higher = better separation)", fontsize=9)
    ax_sil.set_xticks(ks)
    fig_sil.tight_layout()
    col_s, _ = st.columns([2, 1])
    with col_s:
        st.image(_fig_to_bytes(fig_sil), use_container_width=True)
    plt.close(fig_sil)

    if not auto_k:
        chosen_k = st.slider("Choose K", 2, k_max, best_k, key="cc_k_manual")

    # ── Final clustering with chosen K ────────────────────────────────────────
    with st.spinner(f"Clustering (K={chosen_k})…"):
        final_labels, _ = _kmeans_cell(X_z, chosen_k, int(cc_seed))

    if final_labels is None:
        st.error("Clustering failed — try a different seed or fewer features.")
        return

    combined["cluster"] = final_labels

    # ── PCA scatter ───────────────────────────────────────────────────────────
    st.markdown(f"### PCA projection (K = {chosen_k})")
    st.caption("Each point = one cell. Shape = condition; colour = cluster.")
    pca_xy = _pca2(X_z)
    fig_pca, ax_pca = plt.subplots(figsize=(6, 5))
    fig_pca.patch.set_facecolor(DARK_BG); _dark_ax(ax_pca)
    markers_cond = {name_a: "o", name_b: "^"}
    for cond, marker in markers_cond.items():
        for cl in range(chosen_k):
            mask = (combined["condition"] == cond) & (combined["cluster"] == cl)
            if not mask.any():
                continue
            ax_pca.scatter(pca_xy[mask, 0], pca_xy[mask, 1],
                           c=CLUSTER_PALETTE[cl % len(CLUSTER_PALETTE)],
                           marker=marker, alpha=0.7, s=36, edgecolors="none",
                           label=f"C{cl} / {cond}" if cl == 0 else None)
    # Legend: conditions
    ax_pca.scatter([], [], marker="o", c="#aaa", label=f"{name_a}  (circle)", s=30)
    ax_pca.scatter([], [], marker="^", c="#aaa", label=f"{name_b}  (triangle)", s=30)
    ax_pca.legend(fontsize=7, facecolor="#111", labelcolor="#ccc",
                  loc="upper right", framealpha=0.3)
    ax_pca.set_xlabel("PC1", fontsize=10); ax_pca.set_ylabel("PC2", fontsize=10)
    ax_pca.set_title("Cell-level PCA — clusters & conditions", fontsize=10)
    fig_pca.tight_layout()
    col_p, _ = st.columns([2, 1])
    with col_p:
        st.image(_fig_to_bytes(fig_pca), use_container_width=True)
    plt.close(fig_pca)

    # ── Cluster profiles ──────────────────────────────────────────────────────
    st.markdown("### Cluster profiles (normalised mean per feature)")
    fig_cp = _make_cluster_profiles(combined, selected_feats, chosen_k)
    st.image(_fig_to_bytes(fig_cp), use_container_width=True)
    plt.close(fig_cp)

    # ── Chi-square on CELL counts (independent units) ─────────────────────────
    st.markdown("### Subpopulation proportions per condition")
    st.caption("Each cell counts once. Chi-square tests whether the mixture of "
               "subpopulations differs between conditions.")

    counts_a = combined[combined["condition"] == name_a]["cluster"].value_counts()
    counts_b = combined[combined["condition"] == name_b]["cluster"].value_counts()
    counts_a = counts_a.reindex(range(chosen_k), fill_value=0)
    counts_b = counts_b.reindex(range(chosen_k), fill_value=0)

    contingency = np.vstack([counts_a.values, counts_b.values])
    try:
        from scipy.stats import chi2_contingency, fisher_exact
        if chosen_k == 2 and contingency.shape == (2, 2):
            # Use Fisher's exact for 2×2 tables (exact, not asymptotic)
            _, pval_chi = fisher_exact(contingency, alternative="two-sided")
            test_name = "Fisher's exact"
            chi2_val = np.nan
        else:
            chi2_val, pval_chi, _, _ = chi2_contingency(contingency)
            test_name = "Chi-square"
    except Exception:
        chi2_val, pval_chi, test_name = np.nan, np.nan, "—"

    fig_freq, pval_f = _make_frequency_figure(
        combined[combined["condition"] == name_a],
        combined[combined["condition"] == name_b],
        chosen_k, name_a, name_b,
    )
    col_f2, _ = st.columns([2, 1])
    with col_f2:
        st.image(_fig_to_bytes(fig_freq), use_container_width=True)
    plt.close(fig_freq)

    # Summary table: counts + proportions
    tbl = {"Cluster": [f"C{j}" for j in range(chosen_k)]}
    total_a, total_b = counts_a.sum(), counts_b.sum()
    tbl[f"{name_a} (n)"]  = counts_a.values.tolist()
    tbl[f"{name_a} (%)"]  = [f"{v/total_a*100:.1f}" for v in counts_a.values]
    tbl[f"{name_b} (n)"]  = counts_b.values.tolist()
    tbl[f"{name_b} (%)"]  = [f"{v/total_b*100:.1f}" for v in counts_b.values]
    st.dataframe(pd.DataFrame(tbl), hide_index=True, use_container_width=True)

    sig_color = "#22c55e" if (not np.isnan(pval_chi) and pval_chi < 0.05) else "#ef4444"
    chi_str = f"χ² = {chi2_val:.2f}, " if not np.isnan(chi2_val) else ""
    p_str   = f"p = {pval_chi:.4f}" if not np.isnan(pval_chi) else "p = N/A"
    st.markdown(
        f"**{test_name}**: {chi_str}{p_str}  "
        f"<span style='color:{sig_color};font-weight:700'>"
        f"{'Significant ✓' if (not np.isnan(pval_chi) and pval_chi < 0.05) else 'Not significant ✗'}"
        f"</span>",
        unsafe_allow_html=True,
    )

    # Download
    dl = combined[["condition", "field_id", "cell_id", "cluster"] + selected_feats]
    st.download_button("Download cell cluster assignments (CSV)",
                       data=dl.to_csv(index=False).encode("utf-8"),
                       file_name="cell_clusters.csv", mime="text/csv",
                       use_container_width=True, key="cc_dl")


def render_pattern_section(df_a: pd.DataFrame, df_b: pd.DataFrame,
                           name_a: str, name_b: str):
    st.markdown("---")
    st.markdown("""<div class="hbox" style="margin-top:8px;">
    <h2 style="margin:0;color:#ffd700;font-size:1.5rem;">
    🔍 Movement pattern detection
    <span style="font-size:0.9rem;color:#555588;">— unsupervised clustering</span></h2>
    <p style="margin:6px 0 0;color:#8888cc;font-size:0.85rem;">
    Trajectory windows are clustered by movement features. Comparing cluster frequencies
    between conditions reveals behavioural patterns that differ.
    </p></div>""", unsafe_allow_html=True)

    # ── PARAMETERS ───────────────────────────────────────────────────────────
    st.markdown("#### Parameters")
    c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
    with c1:
        window_size = st.slider("Window size (frames)", 3, 20, 7, key="pat_w")
    with c2:
        stride_mode = st.radio(
            "Window overlap",
            ["Non-overlapping", "Sliding (overlap)"],
            key="pat_stride",
            help="Non-overlapping windows are independent — preferred for chi-square. "
                 "Sliding produces more windows but correlated samples.",
        )
        stride = window_size if stride_mode == "Non-overlapping" else 1
    with c3:
        k = st.slider("Number of clusters (K)", 2, 8, 4, key="pat_k")
    with c4:
        pat_seed = st.number_input("Seed", 0, 9999, 42, key="pat_seed")

    # ── FEATURE SELECTION ─────────────────────────────────────────────────────
    st.markdown("#### Features for clustering")
    available = list(ALL_FEATURES.keys())
    if window_size < 5:
        available = [f for f in available if f != "alpha_local"]
        st.caption("Local MSD exponent (α) requires window ≥ 5 frames.")

    defaults = {f: True for f in available}
    cols = st.columns(len(available))
    selected = []
    for i, feat in enumerate(available):
        if cols[i].checkbox(ALL_FEATURES[feat], value=defaults[feat], key=f"pat_feat_{feat}"):
            selected.append(feat)

    if len(selected) == 0:
        st.warning("Select at least one feature.")
        return

    # ── RUN ───────────────────────────────────────────────────────────────────
    st.markdown("")
    if not st.button("Detect patterns", type="primary",
                     use_container_width=True, key="pat_run"):
        return

    with st.spinner("Extracting windows…"):
        wins_a = _extract_windows(df_a, window_size, stride)
        wins_b = _extract_windows(df_b, window_size, stride)

    n_wa, n_wb = len(wins_a), len(wins_b)
    st.info(f"Windows extracted — {name_a}: {n_wa} · {name_b}: {n_wb}  "
            f"(window={window_size} fr, stride={stride} fr)")

    if n_wa < k or n_wb < k:
        st.error("Not enough windows for the chosen K. Reduce K or window size.")
        return

    # Cluster on the combined pool so centroids are shared between conditions
    with st.spinner("Clustering…"):
        df_all = pd.concat(
            [wins_a.assign(condition=name_a), wins_b.assign(condition=name_b)],
            ignore_index=True,
        )
        df_all_c, _ = _cluster_windows(df_all, selected, k, int(pat_seed))
        wins_a_c = df_all_c[df_all_c["condition"] == name_a].copy()
        wins_b_c = df_all_c[df_all_c["condition"] == name_b].copy()

    # ── CLUSTER PROFILES ──────────────────────────────────────────────────────
    st.markdown("### Cluster profiles")
    st.caption("Each panel shows the normalised mean of the selected features for that cluster.")
    fig_p = _make_cluster_profiles(df_all_c, selected, k)
    st.image(_fig_to_bytes(fig_p), use_container_width=True)
    plt.close(fig_p)

    # ── FREQUENCY COMPARISON ──────────────────────────────────────────────────
    st.markdown("### Cluster usage per condition")
    fig_f, pval = _make_frequency_figure(wins_a_c, wins_b_c, k, name_a, name_b)
    col_f, _ = st.columns([2, 1])
    with col_f:
        st.image(_fig_to_bytes(fig_f), use_container_width=True)
    plt.close(fig_f)

    if not np.isnan(pval):
        if pval < 0.05:
            st.success(
                f"Significant difference in cluster usage between conditions "
                f"(χ² test, p = {pval:.4f})."
            )
        else:
            st.info(
                f"No significant difference in cluster usage (χ² test, p = {pval:.4f}). "
                "Try adjusting K or the window size."
            )

    # ── SEGMENT EXAMPLES ──────────────────────────────────────────────────────
    st.markdown("### Example trajectory segments per cluster")
    st.caption(
        "Up to 6 random segments per condition, normalised to start at the origin. "
        "Circle = start, square = end."
    )
    fig_s = _make_segment_examples(wins_a_c, wins_b_c, k, name_a, name_b)
    st.image(_fig_to_bytes(fig_s), use_container_width=True)
    plt.close(fig_s)

    # ── DOWNLOAD ──────────────────────────────────────────────────────────────
    st.markdown("---")
    dl_cols = ["condition", "field_id", "cell_id",
               "start_frame", "end_frame", "cluster"] + selected
    st.download_button(
        "Download cluster assignments (CSV)",
        data=df_all_c[dl_cols].to_csv(index=False).encode("utf-8"),
        file_name="pattern_clusters.csv",
        mime="text/csv",
        use_container_width=True,
        key="pat_dl",
    )


# ── MAIN RENDER ───────────────────────────────────────────────────────────────

def render_stats_page():
    st.markdown("""<div class="hbox">
    <h1 style="margin:0;color:#00d4ff;font-size:2rem;">
    📊 Statistical Analysis
    <span style="font-size:1rem;color:#555588;">— condition comparison</span></h1>
    <p style="margin:6px 0 0;color:#8888cc;font-size:0.9rem;">
    Upload two tracker CSV files to compare movement patterns between conditions.
    Equal-N sampling ensures statistically valid comparisons (Mann-Whitney U test).
    </p></div>""", unsafe_allow_html=True)

    # ── DATA LOADING ──────────────────────────────────────────────────────────
    test_csvs = sorted(TEST_FILES_DIR.glob("*.csv"))
    use_test  = st.checkbox(
        "Use built-in test data",
        value=False,
        help="Loads sample CSVs from the test_files/ folder included with the app.",
    )

    col_a, col_b = st.columns(2)
    df_a = df_b = None

    if use_test:
        if len(test_csvs) < 2:
            st.error("Need at least 2 CSV files in test_files/.")
            return
        test_names = [f.name for f in test_csvs]
        with col_a:
            st.markdown("#### Condition 1")
            sel_a  = st.selectbox("File", test_names, index=0, key="tsel_a")
            name_a = st.text_input(
                "Label",
                value=sel_a.replace("_cerevisae_track_data.csv", "").replace("_", " "),
                key="lbl_a",
            )
            try:
                df_a = _load_csv(TEST_FILES_DIR / sel_a)
                n_cells_a = df_a.groupby(["field_id", "cell_id"]).ngroups
                st.success(f"{len(df_a)} rows · {n_cells_a} cells")
            except Exception as e:
                st.error(str(e))

        with col_b:
            st.markdown("#### Condition 2")
            default_b = min(1, len(test_names) - 1)
            sel_b  = st.selectbox("File", test_names, index=default_b, key="tsel_b")
            name_b = st.text_input(
                "Label",
                value=sel_b.replace("_cerevisae_track_data.csv", "").replace("_", " "),
                key="lbl_b",
            )
            try:
                df_b = _load_csv(TEST_FILES_DIR / sel_b)
                n_cells_b = df_b.groupby(["field_id", "cell_id"]).ngroups
                st.success(f"{len(df_b)} rows · {n_cells_b} cells")
            except Exception as e:
                st.error(str(e))

    else:
        with col_a:
            st.markdown("#### Condition 1")
            name_a = st.text_input("Label", value="Condition 1", key="lbl_a")
            up_a   = st.file_uploader("CSV file", type=["csv"], key="up_a")
            if up_a:
                try:
                    df_a = _load_csv(up_a)
                    n_cells_a = df_a.groupby(["field_id", "cell_id"]).ngroups
                    st.success(f"{len(df_a)} rows · {n_cells_a} cells")
                except Exception as e:
                    st.error(str(e))

        with col_b:
            st.markdown("#### Condition 2")
            name_b = st.text_input("Label", value="Condition 2", key="lbl_b")
            up_b   = st.file_uploader("CSV file", type=["csv"], key="up_b")
            if up_b:
                try:
                    df_b = _load_csv(up_b)
                    n_cells_b = df_b.groupby(["field_id", "cell_id"]).ngroups
                    st.success(f"{len(df_b)} rows · {n_cells_b} cells")
                except Exception as e:
                    st.error(str(e))

    if df_a is None or df_b is None:
        st.info("Load data for both conditions to continue.")
        return

    # ── PER-CELL METRICS (full set, before sampling) ──────────────────────────
    with st.spinner("Computing per-cell metrics…"):
        ma_full = _compute_cell_metrics(df_a)
        mb_full = _compute_cell_metrics(df_b)

    n_a   = len(ma_full)
    n_b   = len(mb_full)
    n_min = min(n_a, n_b)

    if n_min < 3:
        st.error(f"Need at least 3 valid cells per condition (found {n_a} and {n_b}).")
        return

    # ── N MATCHING ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Sampling & N matching")
    st.caption(
        "Mann-Whitney U does not require equal N — the default is to use every cell "
        "available in each condition (maximum power). Disable **Use all cells** to "
        "randomly subsample both conditions to the same N (e.g. for paired-N controls)."
    )

    use_all = st.checkbox(
        "Use all cells (recommended — no random subsampling)",
        value=True,
        help="When ON, the test uses every cell from both conditions (Mann-Whitney "
             "handles unequal sample sizes correctly). When OFF, both conditions are "
             "randomly subsampled to the slider value, which reduces statistical power.",
    )

    c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
    with c1:
        n_sample = st.slider(
            "Cells per condition (N)",
            min_value=3,
            max_value=n_min,
            value=n_min,
            disabled=use_all,
        )
    with c2:
        seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42,
                               help="Reproducibility seed for the random subsample.",
                               disabled=use_all)
    with c3:
        st.metric(f"{name_a}", f"{n_a} cells")
    with c4:
        st.metric(f"{name_b}", f"{n_b} cells")

    if use_all:
        ma, mb = ma_full, mb_full
        effective_n = f"{n_a} vs {n_b}"
    else:
        ma, mb = _match_n(ma_full, mb_full, n_sample, int(seed))
        effective_n = f"{n_sample} vs {n_sample} (random subsample)"

    # ── RUN ───────────────────────────────────────────────────────────────────
    st.markdown("---")
    run = st.button("Run statistical analysis", type="primary", use_container_width=True)

    if run:
        # ── SUMMARY STATISTICS ────────────────────────────────────────────────
        st.markdown("### Summary statistics")
        summary = []
        for m in METRIC_COLS:
            a_vals = ma[m].dropna()
            b_vals = mb[m].dropna()
            summary.append({
                "Metric":             METRICS_LABELS[m],
                f"{name_a} — median": round(float(a_vals.median()), 4),
                f"{name_a} — IQR":    f"{round(float(a_vals.quantile(0.25)), 3)} – {round(float(a_vals.quantile(0.75)), 3)}",
                f"{name_b} — median": round(float(b_vals.median()), 4),
                f"{name_b} — IQR":    f"{round(float(b_vals.quantile(0.25)), 3)} – {round(float(b_vals.quantile(0.75)), 3)}",
            })
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

        # ── STATISTICAL TESTS ─────────────────────────────────────────────────
        st.markdown(f"### Mann-Whitney U tests (two-sided, N = {effective_n})")
        st.caption(
            "**p-value**: raw Mann-Whitney p.  "
            "**q (FDR)**: Benjamini-Hochberg adjusted p — use this to control false "
            f"discoveries across {len(METRIC_COLS)} simultaneous tests.  "
            "**r (effect)**: rank-biserial correlation: |r| < 0.1 negligible, "
            "0.1–0.3 small, 0.3–0.5 medium, > 0.5 large.  "
            f"α is excluded when MSD log-log R² < {ALPHA_R2_MIN} (unreliable fit)."
        )
        test_df = _run_tests(ma, mb, name_a, name_b)

        def _color_sig(val):
            if val == "✓":
                return "color: #22c55e; font-weight: bold"
            if val == "✗":
                return "color: #ef4444"
            return ""

        st.dataframe(
            test_df.style.map(_color_sig, subset=["Sig. q<0.05"]),
            use_container_width=True,
            hide_index=True,
        )

        # ── DISTRIBUTION PLOTS ────────────────────────────────────────────────
        st.markdown("### Distribution comparison")
        with st.spinner("Rendering violin plots…"):
            fig_v = _make_violin_grid(ma, mb, name_a, name_b)
        st.image(_fig_to_bytes(fig_v), use_container_width=True)
        plt.close(fig_v)

        # ── ENSEMBLE MSD ──────────────────────────────────────────────────────
        st.markdown("### Ensemble MSD")
        with st.spinner("Computing ensemble MSD…"):
            sampled_a = ma[["field_id", "cell_id"]]
            sampled_b = mb[["field_id", "cell_id"]]
            df_a_s = df_a.merge(sampled_a, on=["field_id", "cell_id"])
            df_b_s = df_b.merge(sampled_b, on=["field_id", "cell_id"])
            msd_a = _compute_ensemble_msd(df_a_s)
            msd_b = _compute_ensemble_msd(df_b_s)
        fig_m = _make_msd_figure(msd_a, msd_b, name_a, name_b)
        col_msd, _ = st.columns([2, 1])
        with col_msd:
            st.image(_fig_to_bytes(fig_m), use_container_width=True)
        plt.close(fig_m)

        # ── DOWNLOAD ──────────────────────────────────────────────────────────
        st.markdown("---")
        all_metrics = pd.concat(
            [ma.assign(condition=name_a), mb.assign(condition=name_b)],
            ignore_index=True,
        )
        st.download_button(
            "Download per-cell metrics (CSV)",
            data=all_metrics.to_csv(index=False).encode("utf-8"),
            file_name="per_cell_metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── CELL-LEVEL SUBPOPULATION (uses full ma_full / mb_full) ───────────────
    render_cell_subpop_section(ma_full, mb_full, name_a, name_b)

    # ── PATTERN DETECTION (window-based clustering) ───────────────────────────
    render_pattern_section(df_a, df_b, name_a, name_b)
