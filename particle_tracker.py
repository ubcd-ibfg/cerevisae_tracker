#!/usr/bin/env python3
"""
particle_tracker.py
====================
Tracking del punto más intenso en una serie de imágenes TIFF.

Uso:
    python particle_tracker.py <ruta_tiff_o_carpeta> [opciones]

Ejemplos:
    python particle_tracker.py stack.tif
    python particle_tracker.py carpeta_de_frames/
    python particle_tracker.py stack.tif --max-gap 0.30 --threshold-pct 90

Salidas (en --output-dir):
    trajectory.png  – trayectoria + MSD + histograma velocidades
    trajectory.csv  – coordenadas frame a frame
    annotated.gif   – GIF con marcadores superpuestos
    report.txt      – reporte completo de estadísticas
"""

from PIL import Image, ImageDraw
import tifffile
from scipy import ndimage, interpolate, stats
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import os
import sys
import argparse
import warnings
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from config import (
    ALPHA_BALLISTIC,
    ALPHA_SUBDIFFUSIVE,
    ALPHA_SUPERDIFFUSIVE,
    MSD_FIT_LEN,
    MSD_MAX_LAG,
)

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE IMÁGENES
# ══════════════════════════════════════════════════════════════════════════════

def load_tiff(path: Path) -> np.ndarray:
    """
    Carga un TIFF stack o carpeta de TIFFs individuales.
    Retorna array (n_frames, H, W) float32.
    Maneja automáticamente frames de distinto tamaño recortando al mínimo común.
    """
    path = Path(path)

    if path.is_dir():
        tiff_files = sorted(
            list(path.glob("*.tif")) + list(path.glob("*.tiff")),
            key=lambda p: p.name,
        )
        if not tiff_files:
            raise FileNotFoundError(
                f"No se encontraron archivos TIFF en: {path}")

        frames = []
        for f in tiff_files:
            img = tifffile.imread(str(f))
            if img.ndim == 2:
                frames.append(img.astype(np.float32))
            elif img.ndim == 3:
                # (H, W, C) si el tercer eje es pequeño; (Z/T, H, W) si es grande
                if img.shape[2] <= 4:
                    frames.append(img[..., 0].astype(np.float32))
                else:
                    for i in range(img.shape[0]):
                        frames.append(img[i].astype(np.float32))
            elif img.ndim == 4:
                for i in range(img.shape[0]):
                    frames.append(img[i, 0].astype(np.float32))
            else:
                frames.append(img.reshape(
                    img.shape[0], img.shape[1]).astype(np.float32))

        # Manejar frames con distinto tamaño → recortar al mínimo común
        shapes = [fr.shape for fr in frames]
        if len(set(shapes)) > 1:
            min_h = min(s[0] for s in shapes)
            min_w = min(s[1] for s in shapes)
            frames = [fr[:min_h, :min_w] for fr in frames]
            print(
                f"  Formas heterogeneas detectadas → recortadas a {min_h}x{min_w} px")

        stack = np.stack(frames, axis=0)

    else:
        img = tifffile.imread(str(path))
        if img.ndim == 2:
            stack = img[np.newaxis].astype(np.float32)
        elif img.ndim == 3:
            stack = img.astype(np.float32)
        elif img.ndim == 4:
            stack = img[:, 0].astype(np.float32)
        else:
            raise ValueError(f"Forma TIFF no soportada: {img.shape}")

    print(
        f"  Cargados {stack.shape[0]} frames  |  tamanio: {stack.shape[1]}x{stack.shape[2]} px")
    return stack


# ══════════════════════════════════════════════════════════════════════════════
# DETECCION AUTOMATICA DE PERDIDA DE SENAL
# ══════════════════════════════════════════════════════════════════════════════

def detect_signal_end(
    stack: np.ndarray,
    threshold_ratio: float = 0.15,
    window: int = 7,
    gaussian_sigma: float = 1.5,
    smooth_window: int = 5,
) -> dict:
    """
    Detecta el ultimo frame en que la senal es significativa.

    Estrategia
    ----------
    1. Calcula la intensidad pico suavizada por frame.
    2. Normaliza respecto al percentil 95 del 20% inicial (senal inicial robusta).
    3. Busca la primera racha de `window` frames consecutivos por debajo de
       threshold_ratio * ref → ese es el frame de corte.

    Retorna dict
    ------------
    end_frame        : indice del ultimo frame valido (inclusive)
    intensity_trace  : intensidad pico cruda por frame
    intensity_smooth : traza suavizada
    intensity_norm   : traza normalizada [0-1]
    threshold_abs    : umbral en unidades originales
    ref_intensity    : intensidad de referencia
    auto_cut         : True si se detecto caida de senal antes del final
    """
    n = stack.shape[0]

    # Traza de intensidad maxima por frame
    raw_trace = np.empty(n, dtype=np.float64)
    for i in range(n):
        sm = ndimage.gaussian_filter(
            stack[i].astype(np.float64), sigma=gaussian_sigma)
        raw_trace[i] = sm.max()

    # Promedio movil
    k = max(1, smooth_window)
    kernel = np.ones(k) / k
    smooth_trace = np.convolve(raw_trace, kernel, mode="same")
    half = k // 2
    if half > 0:
        smooth_trace[:half] = raw_trace[:half]
        smooth_trace[-half:] = raw_trace[-half:]

    # Intensidad de referencia: percentil 95 del 20% inicial
    n_ref = max(1, n // 5)
    ref_intensity = float(np.percentile(smooth_trace[:n_ref], 95))
    if ref_intensity <= 0:
        ref_intensity = float(smooth_trace.max()) or 1.0

    threshold_abs = threshold_ratio * ref_intensity
    intensity_norm = smooth_trace / ref_intensity

    # Buscar primera racha de `window` frames bajo umbral
    below = smooth_trace < threshold_abs
    end_frame = n - 1

    consecutive = 0
    for i in range(n):
        if below[i]:
            consecutive += 1
            if consecutive >= window:
                end_frame = max(0, i - window)
                break
        else:
            consecutive = 0

    return {
        "end_frame": int(end_frame),
        "intensity_trace": raw_trace.tolist(),
        "intensity_smooth": smooth_trace.tolist(),
        "intensity_norm": intensity_norm.tolist(),
        "threshold_abs": float(threshold_abs),
        "ref_intensity": float(ref_intensity),
        "auto_cut": end_frame < n - 1,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DETECCION DE PICO
# ══════════════════════════════════════════════════════════════════════════════

def detect_peak(
    frame: np.ndarray,
    threshold_pct: float = 95.0,
    roi_radius: int = 5,
    gaussian_sigma: float = 1.5,
):
    """
    Detecta el punto mas intenso en un frame mediante centroide sub-pixel.
    Retorna (x, y, intensity) o None.
    """
    smoothed = ndimage.gaussian_filter(
        frame.astype(np.float64), sigma=gaussian_sigma)
    peak_idx = np.unravel_index(np.argmax(smoothed), smoothed.shape)
    peak_y, peak_x = int(peak_idx[0]), int(peak_idx[1])
    peak_intensity = float(frame[peak_y, peak_x])

    threshold = float(np.percentile(frame, threshold_pct))
    if peak_intensity < threshold:
        return None

    y0 = max(0, peak_y - roi_radius)
    y1 = min(frame.shape[0], peak_y + roi_radius + 1)
    x0 = max(0, peak_x - roi_radius)
    x1 = min(frame.shape[1], peak_x + roi_radius + 1)

    roi = smoothed[y0:y1, x0:x1]
    roi_norm = roi - roi.min()
    total = roi_norm.sum()

    if total == 0:
        return (float(peak_x), float(peak_y), peak_intensity)

    y_grid = np.arange(y0, y1, dtype=np.float64)
    x_grid = np.arange(x0, x1, dtype=np.float64)
    cx = float(np.sum(roi_norm * x_grid[np.newaxis, :]) / total)
    cy = float(np.sum(roi_norm * y_grid[:, np.newaxis]) / total)

    return (cx, cy, peak_intensity)


# ══════════════════════════════════════════════════════════════════════════════
# TRACKING PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def track_particle(
    stack: np.ndarray,
    threshold_pct: float = 95.0,
    roi_radius: int = 5,
    gaussian_sigma: float = 1.5,
) -> list:
    """
    Detecta el pico en cada frame.
    Retorna lista de (frame_idx, x, y, intensity) o None por frame.
    """
    n_frames = stack.shape[0]
    raw = []
    for i in range(n_frames):
        det = detect_peak(stack[i], threshold_pct=threshold_pct,
                          roi_radius=roi_radius, gaussian_sigma=gaussian_sigma)
        raw.append((i, det[0], det[1], det[2]) if det is not None else None)
    return raw


# ══════════════════════════════════════════════════════════════════════════════
# INTERPOLACION Y DESCARTE
# ══════════════════════════════════════════════════════════════════════════════

def process_trajectory(raw_detections: list, max_gap_ratio: float = 0.20) -> dict:
    """
    Decide si la trayectoria es completa, interpolable o descartable.
    Retorna dict con status, trajectory, missing_ratio, gap_info.
    """
    n_frames = len(raw_detections)
    detected = [(i, *r[1:])
                for i, r in enumerate(raw_detections) if r is not None]
    n_detected = len(detected)
    n_missing = n_frames - n_detected
    missing_ratio = n_missing / n_frames

    # Encontrar gaps
    det_set = set(d[0] for d in detected)
    gaps = []
    gap_start = None
    for i in range(n_frames):
        if i not in det_set:
            if gap_start is None:
                gap_start = i
        else:
            if gap_start is not None:
                gaps.append((gap_start, i - 1, i - gap_start))
                gap_start = None
    if gap_start is not None:
        gaps.append((gap_start, n_frames - 1, n_frames - gap_start))

    if missing_ratio > max_gap_ratio:
        return {"status": "discarded", "trajectory": [], "missing_ratio": missing_ratio,
                "gap_info": gaps, "n_frames": n_frames, "n_detected": n_detected}

    if n_missing == 0:
        trajectory = [(d[0], d[1], d[2], d[3], False) for d in detected]
        return {"status": "complete", "trajectory": trajectory, "missing_ratio": 0.0,
                "gap_info": [], "n_frames": n_frames, "n_detected": n_detected}

    if n_detected < 2:
        return {"status": "discarded", "trajectory": [], "missing_ratio": missing_ratio,
                "gap_info": gaps, "n_frames": n_frames, "n_detected": n_detected}

    fi = np.array([d[0] for d in detected], dtype=float)
    xs = np.array([d[1] for d in detected], dtype=float)
    ys = np.array([d[2] for d in detected], dtype=float)
    ints = np.array([d[3] for d in detected], dtype=float)

    kind = "cubic" if n_detected >= 4 else "linear"
    fx = interpolate.interp1d(
        fi, xs,   kind=kind, bounds_error=False, fill_value="extrapolate")
    fy = interpolate.interp1d(
        fi, ys,   kind=kind, bounds_error=False, fill_value="extrapolate")
    fint = interpolate.interp1d(
        fi, ints, kind="linear", bounds_error=False, fill_value=np.nan)

    det_dict = {d[0]: d for d in detected}
    trajectory = []
    for idx in range(n_frames):
        if idx in det_dict:
            d = det_dict[idx]
            trajectory.append((idx, d[1], d[2], d[3], False))
        else:
            trajectory.append(
                (idx, float(fx(idx)), float(fy(idx)), float(fint(idx)), True))

    return {"status": "interpolated", "trajectory": trajectory, "missing_ratio": missing_ratio,
            "gap_info": gaps, "n_frames": n_frames, "n_detected": n_detected, "interp_kind": kind}


# ══════════════════════════════════════════════════════════════════════════════
# ESTADISTICAS Y CLASIFICACION DE MOVIMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def compute_statistics(trajectory: list) -> dict:
    """Compute trajectory statistics with interpolation-bias correction.

    Speeds, MSD and directionality are computed using ONLY pairs/steps where
    both endpoints are real detections (t[4] is False). Interpolated points
    are kept in the trajectory for visualization and for geometric measures
    (total_path, end_to_end, gyration), but excluded from kinetic statistics
    because cubic interpolation creates artificially smooth motion that would
    bias α towards subdiffusive and shrink the mean speed.
    """
    traj_xy = np.array([(t[1], t[2]) for t in trajectory])
    intensities = np.array([t[3] for t in trajectory])
    is_real = np.array([not t[4] for t in trajectory], dtype=bool)

    dx = np.diff(traj_xy[:, 0])
    dy = np.diff(traj_xy[:, 1])
    step_sizes = np.sqrt(dx**2 + dy**2)

    # Steps where BOTH endpoints are real (not interpolated)
    if len(is_real) > 1:
        real_step = is_real[:-1] & is_real[1:]
    else:
        real_step = np.zeros(0, dtype=bool)
    step_sizes_real = step_sizes[real_step] if real_step.any() else step_sizes

    total_path = float(step_sizes.sum())  # full smoothed path (geometric)
    end_to_end = float(np.linalg.norm(traj_xy[-1] - traj_xy[0]))
    confinement = end_to_end / total_path if total_path > 0 else 0.0
    mean_speed = float(step_sizes_real.mean()) if len(step_sizes_real) > 0 else 0.0
    max_speed = float(step_sizes_real.max()) if len(step_sizes_real) > 0 else 0.0

    n = len(traj_xy)
    max_lag = max(2, min(n // 2, MSD_MAX_LAG))
    msd_values = []
    msd_n_pairs = []
    for lag in range(1, max_lag + 1):
        real_pair = is_real[:-lag] & is_real[lag:]
        disp = traj_xy[lag:] - traj_xy[:-lag]
        if real_pair.any():
            msd_lag = float(np.mean(np.sum(disp[real_pair] ** 2, axis=1)))
            n_pairs = int(real_pair.sum())
        else:
            msd_lag = np.nan
            n_pairs = 0
        msd_values.append((lag, msd_lag))
        msd_n_pairs.append(n_pairs)

    fit_len = min(len(msd_values), MSD_FIT_LEN)
    _valid = np.array(msd_n_pairs[:fit_len]) > 0
    lags_fit = np.array([m[0] for m in msd_values[:fit_len]], dtype=float)[_valid]
    msds_fit = np.array([m[1] for m in msd_values[:fit_len]], dtype=float)[_valid]

    if len(lags_fit) >= 3:
        slope, _, r_lin, _, _ = stats.linregress(lags_fit, msds_fit)
        D = slope / 4.0
        msd_r2_linear = r_lin**2
    else:
        D = np.nan
        msd_r2_linear = np.nan

    if len(lags_fit) >= 3 and np.all(msds_fit > 0):
        alpha, _, r_log, _, _ = stats.linregress(
            np.log(lags_fit), np.log(msds_fit))
        msd_r2_log = r_log**2
    else:
        alpha = np.nan
        msd_r2_log = np.nan

    if np.isnan(alpha):
        motion_type = "Indeterminate"
    elif alpha < ALPHA_SUBDIFFUSIVE:
        motion_type = "Subdiffusive / confined"
    elif alpha < ALPHA_SUPERDIFFUSIVE:
        motion_type = "Diffusive (Brownian)"
    elif alpha < ALPHA_BALLISTIC:
        motion_type = "Superdiffusive"
    else:
        motion_type = "Ballistic / directed"

    # Directionality from real-step angles only (interpolation produces
    # artificially coherent directions and inflates the index).
    if real_step.sum() > 1:
        dx_dir, dy_dir = dx[real_step], dy[real_step]
    elif len(dx) > 1:
        dx_dir, dy_dir = dx, dy
    else:
        dx_dir = dy_dir = np.array([])
    if len(dx_dir) > 1:
        angles = np.arctan2(dy_dir, dx_dir)
        mean_dir_x = float(np.mean(np.cos(angles)))
        mean_dir_y = float(np.mean(np.sin(angles)))
        directionality = float(np.sqrt(mean_dir_x**2 + mean_dir_y**2))
        mean_angle_deg = float(np.degrees(np.arctan2(mean_dir_y, mean_dir_x)))
    else:
        directionality = np.nan
        mean_angle_deg = np.nan

    x_min, x_max = float(traj_xy[:, 0].min()), float(traj_xy[:, 0].max())
    y_min, y_max = float(traj_xy[:, 1].min()), float(traj_xy[:, 1].max())
    cm = traj_xy.mean(axis=0)
    gyration_radius = float(np.sqrt(np.mean(np.sum((traj_xy - cm) ** 2, axis=1))))

    # Intensity stats from REAL detections only (interpolated intensities are
    # linear estimates that bias the CV — used in QC T6)
    real_intensities = intensities[is_real]
    if len(real_intensities) > 0:
        mean_intensity = float(np.nanmean(real_intensities))
        std_intensity = float(np.nanstd(real_intensities))
    else:
        mean_intensity = float(np.nanmean(intensities))
        std_intensity = float(np.nanstd(intensities))

    return {
        "n_frames": n,
        "n_real": int(is_real.sum()),
        "total_path_px": total_path,
        "end_to_end_px": end_to_end,
        "confinement_ratio": confinement,
        "mean_speed_px_frame": mean_speed,
        "max_speed_px_frame": max_speed,
        "diffusion_coefficient": D,
        "msd_r2_linear": msd_r2_linear,
        "alpha_msd": alpha,
        "msd_r2_log": msd_r2_log,
        "motion_type": motion_type,
        "directionality_index": directionality,
        "mean_direction_deg": mean_angle_deg,
        "bounding_box_px": (x_min, y_min, x_max, y_max),
        "gyration_radius_px": gyration_radius,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "msd_values": msd_values,
        "msd_n_pairs": msd_n_pairs,
        "step_sizes": step_sizes_real.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# GRAFICA DE TRAYECTORIA + MSD
# ══════════════════════════════════════════════════════════════════════════════

def plot_trajectory(result, stats_d, output_path, stack):
    traj = result["trajectory"]
    traj_xy = np.array([(t[1], t[2]) for t in traj])
    frames = np.array([t[0] for t in traj])
    interp = np.array([t[4] for t in traj])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0e0e1a")
    for ax in axes:
        ax.set_facecolor("#0e0e1a")
        for sp in ax.spines.values():
            sp.set_color("#444")
        ax.tick_params(colors="#ccc")
        ax.xaxis.label.set_color("#ccc")
        ax.yaxis.label.set_color("#ccc")
        ax.title.set_color("#eee")

    ax = axes[0]
    ax.imshow(stack.max(axis=0), cmap="inferno",
              origin="upper", aspect="equal")
    pts = traj_xy.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    norm = plt.Normalize(frames.min(), frames.max())
    lc = LineCollection(segs, cmap="cool", norm=norm,
                        linewidth=1.8, alpha=0.85)
    lc.set_array(frames[:-1])
    ax.add_collection(lc)
    real = ~interp
    ax.scatter(traj_xy[real, 0], traj_xy[real, 1], c=frames[real], cmap="cool",
               norm=norm, s=18, zorder=5, edgecolors="none")
    if interp.any():
        ax.scatter(traj_xy[interp, 0], traj_xy[interp, 1], marker="x", c="yellow",
                   s=30, zorder=6, linewidths=0.8, label="Interpolado")
        ax.legend(fontsize=8, facecolor="#1a1a2e",
                  labelcolor="white", loc="upper right")
    ax.scatter(*traj_xy[0], s=80, c="lime", zorder=7,
               edgecolors="white", linewidths=0.8)
    ax.scatter(*traj_xy[-1], s=80, c="red", zorder=7, edgecolors="white",
               linewidths=0.8, marker="*")
    cb = plt.colorbar(lc, ax=ax, label="Frame")
    cb.ax.yaxis.label.set_color("#ccc")
    cb.ax.tick_params(colors="#ccc")
    ax.set_title("Trayectoria (proyeccion maxima)")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")

    ax2 = axes[1]
    msd_vals = stats_d["msd_values"]
    lags_m = [m[0] for m in msd_vals]
    msds_m = [m[1] for m in msd_vals]
    ax2.plot(lags_m, msds_m, "o-", color="#00d4ff",
             markersize=4, linewidth=1.5)
    if not np.isnan(stats_d["diffusion_coefficient"]):
        fit_len = min(len(lags_m), 8)
        la = np.array(lags_m[:fit_len])
        slope = stats_d["diffusion_coefficient"] * 4
        ic = np.mean(msds_m[:fit_len]) - slope * la.mean()
        ax2.plot(la, slope * la + ic, "--", color="#ff6b6b", linewidth=1.2,
                 label=f"D = {stats_d['diffusion_coefficient']:.2f} px2/frame")
        ax2.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
    alpha_str = f"alpha={stats_d['alpha_msd']:.2f}" if not np.isnan(
        stats_d["alpha_msd"]) else ""
    ax2.set_title(f"MSD  {alpha_str}  ({stats_d['motion_type']})")
    ax2.set_xlabel("Lag (frames)")
    ax2.set_ylabel("MSD (px2)")

    ax3 = axes[2]
    steps = np.array(stats_d["step_sizes"])
    if len(steps) > 0:
        ax3.hist(steps, bins=max(5, len(steps)//5), color="#7c4dff",
                 edgecolor="#1a1a2e", alpha=0.85, density=True)
        ax3.axvline(stats_d["mean_speed_px_frame"], color="#ff9800", linestyle="--",
                    linewidth=1.5, label=f"Media: {stats_d['mean_speed_px_frame']:.2f} px/frame")
        ax3.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
    ax3.set_title("Distribucion de velocidades")
    ax3.set_xlabel("Velocidad (px/frame)")
    ax3.set_ylabel("Densidad")

    plt.tight_layout(pad=2.0)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Grafica guardada: {output_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# GIF ANOTADO
# ══════════════════════════════════════════════════════════════════════════════

def make_annotated_gif(stack, result, output_path, tail_length=15, fps=10):
    traj_dict = {t[0]: t for t in result["trajectory"]}
    vmin, vmax = stack.min(), stack.max()
    stack_n = ((stack - vmin) / max(vmax - vmin, 1) * 255).astype(np.uint8)
    cmap = plt.get_cmap("inferno")

    pil_frames = []
    for i in range(stack.shape[0]):
        frame_rgb = (cmap(stack_n[i] / 255.0)[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(img)

        tail_pts = [(traj_dict[j][1], traj_dict[j][2])
                    for j in range(max(0, i - tail_length), i + 1) if j in traj_dict]
        if len(tail_pts) > 1:
            for k in range(len(tail_pts) - 1):
                a = int(60 + 195 * (k / len(tail_pts)))
                draw.line([tail_pts[k], tail_pts[k + 1]],
                          fill=(0, a, 255), width=2)

        if i in traj_dict:
            tx, ty = traj_dict[i][1], traj_dict[i][2]
            color = (255, 220, 0) if traj_dict[i][4] else (0, 255, 100)
            r = 5
            draw.ellipse([(tx-r, ty-r), (tx+r, ty+r)], outline=color, width=2)
            draw.ellipse([(tx-2, ty-2), (tx+2, ty+2)], fill=color)

        draw.text((5, 5), f"Frame {i:04d}", fill=(200, 200, 200))
        if i in traj_dict and traj_dict[i][4]:
            draw.text((5, 18), "* interpolado", fill=(255, 220, 0))
        pil_frames.append(img)

    if pil_frames:
        dur = max(50, 1000 // fps)
        pil_frames[0].save(str(output_path), save_all=True,
                           append_images=pil_frames[1:], loop=0, duration=dur, optimize=False)
        print(f"  GIF guardado: {output_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# CSV y REPORTE
# ══════════════════════════════════════════════════════════════════════════════

def save_csv(result, output_path):
    rows = [{"frame": t[0], "x_px": round(t[1], 4), "y_px": round(t[2], 4),
             "intensity": round(t[3], 2) if not np.isnan(t[3]) else np.nan,
             "interpolated": t[4]} for t in result["trajectory"]]
    pd.DataFrame(rows).to_csv(str(output_path), index=False)
    print(f"  CSV guardado: {output_path.name}")


def save_report(result, stats_d, output_path, params):
    L = []
    L.append("=" * 62)
    L.append("  REPORTE DE TRACKING DE PARTICULA")
    L.append("=" * 62)
    L.append("\n-- PARAMETROS --")
    for k, v in params.items():
        L.append(f"  {k:<30}: {v}")
    L.append("\n-- DETECCION --")
    L.append(f"  Estado: {result['status'].upper()}")
    L.append(f"  Frames totales   : {result['n_frames']}")
    L.append(f"  Detectados       : {result['n_detected']}")
    L.append(
        f"  Faltantes        : {result['n_frames'] - result['n_detected']}")
    L.append(f"  Fraccion faltante: {result['missing_ratio']*100:.1f}%")
    if result.get("gap_info"):
        L.append("\n-- GAPS --")
        for g in result["gap_info"]:
            L.append(f"  Frames {g[0]}-{g[1]}  ({g[2]} consecutivos)")
    if result["status"] != "discarded":
        s = stats_d
        L.append("\n-- TRAYECTORIA --")
        L.append(f"  Longitud total (px)     : {s['total_path_px']:.2f}")
        L.append(f"  Desplazamiento ext (px) : {s['end_to_end_px']:.2f}")
        L.append(f"  Ratio confinamiento     : {s['confinement_ratio']:.3f}")
        L.append(f"  Radio de giro (px)      : {s['gyration_radius_px']:.2f}")
        L.append(f"\n-- VELOCIDAD --")
        L.append(f"  Media (px/frame)        : {s['mean_speed_px_frame']:.4f}")
        L.append(f"  Maxima (px/frame)       : {s['max_speed_px_frame']:.4f}")
        L.append(f"\n-- MSD --")
        L.append(f"  Exponente alpha         : {s['alpha_msd']:.4f}")
        L.append(
            f"  Coef. difusion D        : {s['diffusion_coefficient']:.6f} px2/frame")
        L.append(f"  Tipo de movimiento      : {s['motion_type']}")
        L.append(
            f"  Indice direccionalidad  : {s['directionality_index']:.3f}")
    L.append("\n" + "=" * 62)
    output_path.write_text("\n".join(L), encoding="utf-8")
    print(f"  Reporte guardado: {output_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN (CLI)
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Tracking del punto mas intenso en imagenes TIFF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Ejemplos:
          python particle_tracker.py stack.tif
          python particle_tracker.py frames/  --max-gap 0.40 --output-dir resultados/
          python particle_tracker.py stack.tif --threshold-pct 90 --no-auto-cut
        """),
    )
    parser.add_argument(
        "input", help="Archivo TIFF (stack) o carpeta de TIFFs.")
    parser.add_argument("--output-dir", default="tracker_output")
    parser.add_argument("--max-gap", type=float, default=0.20, metavar="FRAC",
                        help="Fraccion maxima de frames faltantes (default 0.20 = 20%%).")
    parser.add_argument("--threshold-pct", type=float, default=95.0, metavar="PCT",
                        help="Percentil minimo de intensidad (default 95).")
    parser.add_argument("--roi-radius", type=int, default=5)
    parser.add_argument("--gaussian-sigma", type=float, default=1.5)
    parser.add_argument("--tail-length", type=int, default=15)
    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--no-auto-cut", action="store_true",
                        help="Desactivar corte automatico por perdida de senal.")
    parser.add_argument("--signal-threshold", type=float, default=0.15,
                        help="Fraccion de la senal inicial bajo la cual se considera perdida (default 0.15).")
    parser.add_argument("--signal-window", type=int, default=7,
                        help="Frames consecutivos bajo umbral para confirmar perdida (default 7).")

    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] No existe: {input_path}")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*58)
    print("  PARTICLE TRACKER")
    print("="*58)

    print("\n[1/5] Cargando imagenes...")
    stack_full = load_tiff(input_path)
    n_full = stack_full.shape[0]

    print(f"\n[2/5] Analizando senal ({n_full} frames)...")
    if not args.no_auto_cut:
        sig = detect_signal_end(stack_full, threshold_ratio=args.signal_threshold,
                                window=args.signal_window, gaussian_sigma=args.gaussian_sigma)
        end_f = sig["end_frame"]
        stack = stack_full[:end_f + 1]
        if sig["auto_cut"]:
            print(
                f"  Senal detectada hasta frame {end_f} (de {n_full}). Procesando {end_f+1} frames.")
        else:
            print(f"  Senal activa en todos los frames ({n_full}).")
    else:
        stack = stack_full
        sig = None
        print("  Corte automatico desactivado.")

    n_frames = stack.shape[0]

    print(f"\n[3/5] Detectando pico en {n_frames} frames...")
    raw = track_particle(stack, threshold_pct=args.threshold_pct,
                         roi_radius=args.roi_radius, gaussian_sigma=args.gaussian_sigma)
    n_det = sum(1 for r in raw if r is not None)
    print(f"  Detecciones: {n_det}/{n_frames}  ({100*n_det/n_frames:.1f}%)")

    print("\n[4/5] Evaluando trayectoria...")
    result = process_trajectory(raw, max_gap_ratio=args.max_gap)
    print(f"  Estado: {result['status'].upper()}")
    if result["status"] == "discarded":
        print(
            f"  Particula DESCARTADA ({result['missing_ratio']*100:.1f}% faltantes > {args.max_gap*100:.0f}%)")
        sys.exit(0)

    st = compute_statistics(result["trajectory"])

    print("\n[5/5] Generando salidas...")
    params = {"Frames cargados": n_full, "Frames analizados": n_frames,
              "threshold_pct": args.threshold_pct, "max_gap": args.max_gap}
    plot_trajectory(result, st, out_dir / "trajectory.png", stack)
    save_csv(result, out_dir / "trajectory.csv")
    save_report(result, st, out_dir / "report.txt", params)
    if not args.no_gif:
        make_annotated_gif(stack, result, out_dir / "annotated.gif",
                           tail_length=args.tail_length, fps=args.gif_fps)

    print(f"\n  Resultados en: {out_dir}/")
    print("="*58 + "\n")


if __name__ == "__main__":
    main()
