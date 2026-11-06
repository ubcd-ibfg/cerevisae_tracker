"""
visualization.py - Matplotlib figure generation and styling for Particle Tracker.

Functions for creating publication-quality visualizations:
- Trajectory plots (absolute and relative coordinates)
- Signal intensity traces
- Mean squared displacement analysis
- Animated GIFs
- Field segmentation overviews
- Drift detection visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy import stats
from PIL import Image, ImageDraw
import io
import zipfile
from pathlib import Path

from config import DARK_BG, COLOR_SPINE, COLOR_TICK, COLOR_LABEL, COLOR_TITLE, FIG_DPI
from field_analyzer import apply_drift_correction


def fig_to_bytes(fig):
    """Convert a matplotlib figure to PNG bytes.

    Args:
        fig: A matplotlib figure object to convert.

    Returns:
        bytes: PNG image data as bytes suitable for display in Streamlit.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


def _dark(ax):
    """Apply dark theme styling to a matplotlib axis.

    Configures axis appearance with dark background, light text, and custom colors
    matching the application's dark theme aesthetic.

    Args:
        ax: A matplotlib axis object to style.
    """
    ax.set_facecolor(DARK_BG)
    for sp in ax.spines.values():
        sp.set_color(COLOR_SPINE)
    ax.tick_params(colors=COLOR_TICK)
    ax.xaxis.label.set_color(COLOR_LABEL)
    ax.yaxis.label.set_color(COLOR_LABEL)
    ax.title.set_color(COLOR_TITLE)


def make_traj_fig(result, sd, stack=None, cell_label=""):
    """Three-panel trajectory visualization.

    Panel 1: trajectory path overlaid on the max-projection of *stack* (when provided)
             or plotted in relative coordinates centred on (0, 0) (when stack is None).
    Panel 2: MSD curve with optional linear fit.
    Panel 3: step-size histogram.

    Args:
        result: Tracking result dict containing 'trajectory'.
        sd: Statistics dict with msd_values, alpha_msd, motion_type,
            diffusion_coefficient, and step_sizes.
        stack: Optional 3-D array (n_frames, H, W).  When supplied the first
               panel shows an absolute-coordinate overlay on the max projection;
               when None it shows a relative-coordinate plot centred on (0, 0).
        cell_label: Label shown in the relative-mode panel title.

    Returns:
        matplotlib.figure.Figure with three subplots.
    """
    traj = result["trajectory"]
    xy = np.array([(t[1], t[2]) for t in traj])
    fr = np.array([t[0] for t in traj])
    ip = np.array([t[4] for t in traj])

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes:
        _dark(ax)

    # ── Panel 1: trajectory path ──────────────────────────────────────────────
    ax = axes[0]
    if stack is not None:
        ax.imshow(stack.max(axis=0), cmap="inferno",
                  origin="upper", aspect="equal")
    pts = xy.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    norm = plt.Normalize(fr.min(), fr.max())
    lc = LineCollection(segs, cmap="cool", norm=norm, linewidth=2, alpha=0.9)
    lc.set_array(fr[:-1])
    ax.add_collection(lc)
    real = ~ip
    ax.scatter(xy[real, 0], xy[real, 1], c=fr[real], cmap="cool", norm=norm,
               s=18, zorder=5, edgecolors="none")
    if ip.any():
        ax.scatter(xy[ip, 0], xy[ip, 1], marker="x", c="yellow", s=35,
                   zorder=6, linewidths=1.2, label="Interpolated")
        ax.legend(fontsize=7, facecolor="#111",
                  labelcolor="white", loc="upper right")
    ax.scatter(*xy[0],  s=90, c="lime", zorder=7,
               edgecolors="white", linewidths=0.8)
    ax.scatter(*xy[-1], s=90, c="red",  zorder=7, edgecolors="white",
               linewidths=0.8, marker="*")
    if stack is None:
        ax.axhline(0, color="#444", linewidth=0.8, linestyle=":")
        ax.axvline(0, color="#444", linewidth=0.8, linestyle=":")
        ax.scatter(0, 0, marker="+", s=200, c="white", zorder=8, linewidths=2)
        ax.set_aspect("equal")
        ax.autoscale()
        ax.invert_yaxis()  # match image coords: row 0 at top, y increases downward
        ax.set_title(f"Relative trajectory — {cell_label}")
        ax.set_xlabel("dx (px)")
        ax.set_ylabel("dy (px)")
    else:
        ax.set_title("Trajectory (o=start, *=end)")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
    cb = plt.colorbar(lc, ax=ax, label="Frame")
    cb.ax.yaxis.label.set_color("#ccc")
    cb.ax.tick_params(colors="#aaa")

    # ── Panel 2: MSD ──────────────────────────────────────────────────────────
    ax2 = axes[1]
    lags = [m[0] for m in sd["msd_values"]]
    msds = [m[1] for m in sd["msd_values"]]
    ax2.plot(lags, msds, "o-", color="#00d4ff", markersize=4, linewidth=1.6)
    if not np.isnan(sd["diffusion_coefficient"]):
        la = np.array(lags[:8])
        sl = sd["diffusion_coefficient"] * 4
        ic = np.mean(msds[:8]) - sl * la.mean()
        ax2.plot(la, sl * la + ic, "--", color="#ff6b6b", linewidth=1.3,
                 label=f"D={sd['diffusion_coefficient']:.3f}")
        ax2.legend(fontsize=8, facecolor="#111", labelcolor="white")
    a_s = f"a={sd['alpha_msd']:.2f}" if not np.isnan(sd["alpha_msd"]) else ""
    ax2.set_title(f"MSD  {a_s}  {sd['motion_type']}")
    ax2.set_xlabel("Lag (frames)")
    ax2.set_ylabel("MSD (px²)")

    # ── Panel 3: step-size distribution ──────────────────────────────────────
    ax3 = axes[2]
    steps = np.array(sd["step_sizes"])
    if len(steps):
        ax3.hist(steps, bins=max(5, len(steps) // 5), color="#7c4dff",
                 edgecolor=DARK_BG, alpha=0.85, density=True)
        ax3.axvline(sd["mean_speed_px_frame"], color="#ff9800", linestyle="--",
                    linewidth=1.5, label=f"Mean:{sd['mean_speed_px_frame']:.2f}")
        ax3.legend(fontsize=8, facecolor="#111", labelcolor="white")
    ax3.set_title("Speed distribution")
    ax3.set_xlabel("px/frame")
    ax3.set_ylabel("Density")

    plt.tight_layout(pad=1.8)
    return fig


def make_signal_fig(sig, n_full):
    """Create an intensity trace visualization figure.

    Plots the raw and smoothed intensity traces with threshold, analyzed region,
    and signal cutoff information.

    Args:
        sig: Dictionary containing intensity_trace, intensity_smooth, end_frame,
             threshold_abs, ref_intensity, and auto_cut flag.
        n_full: Total number of frames in the full stack.

    Returns:
        matplotlib.figure.Figure: Figure object displaying the signal analysis.
    """
    raw = np.array(sig["intensity_trace"])
    sm = np.array(sig["intensity_smooth"])
    end = sig["end_frame"]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor(DARK_BG)
    _dark(ax)
    ax.plot(np.arange(len(raw)), raw, color="#00d4ff",
            alpha=0.25, linewidth=0.8)
    ax.plot(np.arange(len(sm)),  sm,  color="#00d4ff",
            linewidth=1.8, label="Smoothed intensity")
    ax.axvspan(0, end, alpha=0.07, color="#22c55e", label="Analyzed region")
    if end < n_full-1:
        ax.axvspan(end, n_full-1, alpha=0.07,
                   color="#ef4444", label="Discarded")
    ax.axhline(sig["threshold_abs"], color="#f59e0b", linestyle="--", linewidth=1.2,
               label=f"Threshold ({sig['threshold_abs']:.0f})")
    ax.axhline(sig["ref_intensity"], color="#444",
               linestyle=":", linewidth=1.0)
    if sig["auto_cut"]:
        ax.axvline(end, color="#ef4444", linewidth=1.8,
                   label=f"Cut frame {end}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Max intensity")
    ax.set_title("Intensity trace")
    ax.legend(fontsize=8, facecolor="#111",
              labelcolor="white", loc="upper right")
    ax.set_xlim(0, n_full-1)
    plt.tight_layout()
    return fig


def make_msd_log(sd):
    """Create a log-log plot of mean squared displacement (MSD).

    Visualizes MSD with power-law fit and reference lines for different regimes
    (subdiffusive, diffusive, superdiffusive, ballistic).

    Args:
        sd: Dictionary containing msd_values (list of [lag, msd] pairs),
            alpha_msd (power law exponent), motion_type, and other statistics.

    Returns:
        matplotlib.figure.Figure: Log-log plot figure with MSD data and fits.
    """
    lm = np.array([m[0] for m in sd["msd_values"]], dtype=float)
    mm = np.array([m[1] for m in sd["msd_values"]], dtype=float)
    mask = mm > 0
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.patch.set_facecolor(DARK_BG)
    _dark(ax)
    ax.loglog(lm[mask], mm[mask], "o-", color="#00d4ff",
              markersize=5, linewidth=1.8, label="MSD")
    if not np.isnan(sd["alpha_msd"]) and mask.sum() >= 3:
        la = lm[mask][:8]
        ms = mm[mask][:8]
        ls, li, *_ = stats.linregress(np.log(la), np.log(ms))
        ax.loglog(la, np.exp(li)*la**ls, "--", color="#ff6b6b", linewidth=1.4,
                  label=f"a={ls:.2f} ({sd['motion_type']})")
    if mask.sum() >= 2:
        rx = np.array([lm[mask][0], lm[mask][-1]])
        sc = mm[mask][0]
        for e, c in [(1, "#555"), (2, "#444")]:
            ax.loglog(rx, sc*(rx/rx[0])**e, ":", color=c, linewidth=1)
    ax.legend(fontsize=8, facecolor="#111", labelcolor="white")
    ax.set_title("MSD log-log")
    ax.set_xlabel("Lag (frames)")
    ax.set_ylabel("MSD (px2)")
    plt.tight_layout()
    return fig


def make_gif(stack, result, tail, fps):
    """Create an animated GIF of the trajectory overlaid on the image stack.

    Renders each frame with trajectory tail, detection markers, and frame counter.
    Interpolated points are marked differently from detected points.

    Args:
        stack: 3D array of image frames (n_frames, height, width).
        result: Dictionary containing 'trajectory' with detected positions.
        tail: Number of historical frames to show in trajectory tail.
        fps: Frames per second for GIF animation speed.

    Returns:
        bytes: GIF image data as bytes.
    """
    td = {t[0]: t for t in result["trajectory"]}
    vmin, vmax = stack.min(), stack.max()
    sn = ((stack - vmin) / max(vmax-vmin, 1) * 255).astype(np.uint8)
    cmap = plt.get_cmap("inferno")
    pf = []
    for i in range(stack.shape[0]):
        rgb = (cmap(sn[i]/255.0)[:, :, :3]*255).astype(np.uint8)
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        pts = [(td[j][1], td[j][2])
               for j in range(max(0, i-tail), i+1) if j in td]
        for k in range(len(pts)-1):
            a = int(60+195*(k/len(pts)))
            draw.line([pts[k], pts[k+1]], fill=(0, a, 255), width=2)
        if i in td:
            tx, ty = td[i][1], td[i][2]
            col = (255, 220, 0) if td[i][4] else (0, 255, 100)
            r = 5
            draw.ellipse([(tx-r, ty-r), (tx+r, ty+r)], outline=col, width=2)
            draw.ellipse([(tx-2, ty-2), (tx+2, ty+2)], fill=col)
        draw.text((4, 4), f"Frame {i:04d}", fill=(200, 200, 200))
        if i in td and td[i][4]:
            draw.text((4, 17), "* interp", fill=(255, 220, 0))
        pf.append(img)
    buf = io.BytesIO()
    dur = max(50, 1000//fps)
    pf[0].save(buf, format="GIF", save_all=True,
               append_images=pf[1:], loop=0, duration=dur, optimize=False)
    buf.seek(0)
    return buf.read()


def make_gif_roi(stack, cell, raw_det, result, drift_info, tail, fps):
    """Create an animated GIF of a cell's region of interest (ROI) with absolute trajectory.

    Renders a cropped view centered on the cell with trajectory overlay in absolute coordinates,
    accounting for drift correction. Marks the cell centroid and shows interpolated points.

    Args:
        stack: 3D array of image frames (n_frames, height, width).
        cell: Dictionary with 'bbox' [r0, c0, r1, c1] and 'centroid' (cy, cx).
        raw_det: List of raw detections [frame, x, y, intensity, abs_x, abs_y] or None.
        result: Dictionary containing 'trajectory' with relative positions.
        drift_info: Dictionary with 'dx' and 'dy' arrays for drift per frame, or None.
        tail: Number of historical frames to show in trajectory tail.
        fps: Frames per second for GIF animation speed.

    Returns:
        bytes or None: GIF image data as bytes, or None if no valid frames exist.
    """
    r0b, c0b, r1b, c1b = cell["bbox"]
    pad = 20
    H, W = stack.shape[1], stack.shape[2]
    td = {t[0]: t for t in result["trajectory"]}
    abs_map = {d[0]: (d[4], d[5]) for d in raw_det if d}
    vmin, vmax = stack.min(), stack.max()
    cmap = plt.get_cmap("inferno")
    pf = []
    cy0, cx0 = cell["centroid"]
    for i in range(stack.shape[0]):
        dy_off = drift_info["dy"][i] if drift_info else 0.0
        dx_off = drift_info["dx"][i] if drift_info else 0.0
        r0 = max(0, int(r0b - dy_off) - pad)
        r1 = min(H, int(r1b - dy_off) + pad)
        c0 = max(0, int(c0b - dx_off) - pad)
        c1 = min(W, int(c1b - dx_off) + pad)
        if r1 <= r0 or c1 <= c0:
            continue
        roi = stack[i, r0:r1, c0:c1]
        sn = ((roi - vmin) / max(vmax-vmin, 1) * 255).astype(np.uint8)
        rgb = (cmap(sn/255.0)[:, :, :3]*255).astype(np.uint8)
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        pts = []
        for j in range(max(0, i-tail), i+1):
            if j in abs_map:
                pts.append((abs_map[j][0]-c0, abs_map[j][1]-r0))
        for k in range(len(pts)-1):
            a = int(60+195*(k/max(len(pts), 1)))
            draw.line([pts[k], pts[k+1]], fill=(0, a, 255), width=2)
        if i in abs_map:
            ax_ = abs_map[i][0]-c0
            ay_ = abs_map[i][1]-r0
            col = (255, 220, 0) if (i in td and td[i][4]) else (0, 255, 100)
            draw.ellipse([(ax_-5, ay_-5), (ax_+5, ay_+5)],
                         outline=col, width=2)
            draw.ellipse([(ax_-2, ay_-2), (ax_+2, ay_+2)], fill=col)
        ccx = int(cx0-dx_off)-c0
        ccy = int(cy0-dy_off)-r0
        draw.line([(ccx-8, ccy), (ccx+8, ccy)], fill=(200, 80, 80), width=1)
        draw.line([(ccx, ccy-8), (ccx, ccy+8)], fill=(200, 80, 80), width=1)
        draw.text((2, 2), f"F{i:04d}", fill=(200, 200, 200))
        pf.append(img)
    if not pf:
        return None
    buf = io.BytesIO()
    dur = max(50, 1000//fps)
    pf[0].save(buf, format="GIF", save_all=True,
               append_images=pf[1:], loop=0, duration=dur, optimize=False)
    buf.seek(0)
    return buf.read()


def make_field_overview(ref_frame, cells, labeled):
    """Create a two-panel field overview visualization.

    Left panel shows reference frame; right panel overlays segmented cell regions
    with color labels and centroids marked.

    Args:
        ref_frame: 2D array of the reference frame (e.g., median projection).
        cells: List of cell dictionaries with 'label', 'area', and 'centroid' (cy, cx).
        labeled: 2D array where each pixel value indicates cell label.

    Returns:
        matplotlib.figure.Figure: Figure with segmentation overview.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes:
        _dark(ax)
    axes[0].imshow(ref_frame, cmap="inferno", origin="upper")
    axes[0].set_title("Reference frame (median)")
    axes[1].imshow(ref_frame, cmap="gray", origin="upper", alpha=0.55)
    cmap_seg = plt.get_cmap("tab20")
    for i, c in enumerate(cells):
        mask = labeled == c["label"]
        color = cmap_seg(i % 20)
        overlay = np.zeros((*ref_frame.shape, 4))
        overlay[mask] = [*color[:3], 0.45]
        axes[1].imshow(overlay, origin="upper")
        cy, cx = c["centroid"]
        axes[1].scatter(cx, cy, marker="+", s=120,
                        c="white", linewidths=2, zorder=5)
        axes[1].text(cx+4, cy-4, f"C{c['label']}", color="white", fontsize=7,
                     fontweight="bold", bbox=dict(facecolor="#000", alpha=0.5, pad=1))
    axes[1].set_title(f"Segmented cells ({len(cells)})")
    plt.tight_layout(pad=1.5)
    return fig


def make_drift_fig(drift_info, field_name=""):
    """Create a three-panel drift visualization figure.

    Plots dx/dy per frame, drift magnitude, and cumulative drift trajectory
    to show spatial drift in the image stack.

    Args:
        drift_info: Dictionary with 'dx', 'dy', 'magnitude' arrays; 'max_drift_px',
                    'is_trending', 'slope_x', 'slope_y'.
        field_name: Optional string name of the field for display in title.

    Returns:
        matplotlib.figure.Figure: Figure with three drift analysis subplots.
    """
    dy = np.array(drift_info["dy"])
    dx = np.array(drift_info["dx"])
    mag = np.array(drift_info["magnitude"])
    frames = np.arange(len(dy))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes:
        _dark(ax)
    axes[0].plot(frames, dx, color="#00d4ff", linewidth=1.8, label="dx")
    axes[0].plot(frames, dy, color="#ff6b6b", linewidth=1.8, label="dy")
    axes[0].axhline(0, color="#555", linewidth=0.8, linestyle=":")
    axes[0].legend(fontsize=8, facecolor="#111", labelcolor="white")
    axes[0].set_title(f"Drift per frame — {field_name}")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Displacement (px)")
    axes[1].plot(frames, mag, color="#f59e0b", linewidth=1.8)
    axes[1].axhline(drift_info["max_drift_px"], color="#ef4444", linestyle="--",
                    linewidth=1.2, label=f"Max: {drift_info['max_drift_px']:.2f} px")
    axes[1].legend(fontsize=8, facecolor="#111", labelcolor="white")
    axes[1].set_title("Drift magnitude")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("px")
    cdx = dx   # already cumulative (each value = total shift from frame 0)
    cdy = dy
    axes[2].plot(cdx, cdy, color="#7c4dff", linewidth=1.8, alpha=0.85)
    axes[2].scatter(0, 0, c="lime", s=80, zorder=5)
    axes[2].scatter(cdx[-1], cdy[-1], c="red", s=80, marker="*", zorder=5)
    axes[2].set_title("Cumulative drift")
    axes[2].set_aspect("equal")
    axes[2].set_xlabel("dx cumul. (px)")
    axes[2].set_ylabel("dy cumul. (px)")
    plt.tight_layout(pad=1.5)
    return fig
