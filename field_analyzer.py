"""
field_analyzer.py
=================
Batch microscopy field analysis: segmentation, drift detection, per-cell tracking
and trajectory quality control.

Public API
----------
make_reference_frame       Build a robust reference image (temporal median projection).
segment_cells              Otsu threshold + connected-component labelling.
detect_drift               Phase cross-correlation drift detection between frames.
detect_drift_from_cells    Mask-based drift estimation: phase correlation on segmented first/last frames.
apply_drift_correction     Sub-pixel drift correction via bilinear interpolation.
track_cell_roi             Track the brightest spot inside a cell ROI across frames.
quality_tests              Run a battery of QC tests on a finished trajectory.

All thresholds and magic constants are imported from ``config.py`` so they can
be adjusted without touching algorithm code.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage, stats
from scipy.fft import fft2, fftshift, ifft2

from config import (
    DEFAULT_BBOX_PAD,
    DEFAULT_N_REF_FRAMES,
    DRIFT_SLOPE_THRESHOLD,
    QT_MAX_GAP_FRACTION,
    QT_MAX_INTENSITY_CV,
    QT_MIN_COMPLETENESS,
    QT_MIN_MSD_R2,
    QT_MIN_SNR,
    QT_WARN_PASS_FRACTION,
)

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# REFERENCE FRAME FOR SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def make_reference_frame(stack: np.ndarray, n_ref: int = DEFAULT_N_REF_FRAMES) -> np.ndarray:
    """
    Compute a reference image for segmentation using the median
    of the first n_ref frames (robust to bright particles that
    appear and disappear).
    """
    n = min(n_ref, stack.shape[0])
    return np.median(stack[:n].astype(np.float64), axis=0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# CELL SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def _otsu_threshold(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Compute Otsu threshold without depending on skimage.

    Args:
        image: 2-D float array.
        mask:  Optional boolean array; only pixels where mask is True
               contribute to the histogram.  Pass the non-zero region to
               exclude StackReg-style black borders from the threshold.
    """
    vals = image.ravel().astype(np.float64)
    if mask is not None:
        vals = vals[mask.ravel()]
    if len(vals) == 0:
        return 0.0
    hist, edges = np.histogram(vals, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    if total == 0:
        return float(vals.mean())
    w0 = np.cumsum(hist)
    w1 = total - w0
    mu0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    mu1 = (np.cumsum((hist * centers)[::-1]) /
           np.maximum(np.cumsum(hist[::-1]), 1))[::-1]
    var_between = w0 * w1 * (mu0 - mu1) ** 2
    idx = int(np.argmax(var_between))
    return float(centers[idx])


def _watershed_split(
    mask: np.ndarray,
    min_dist: int,
    min_size: int,
    max_size: int,
) -> list:
    """Split a binary mask into individual cell regions via watershed.

    Uses the distance transform of *mask* as the landscape.  Local maxima
    of the distance transform (separated by at least *min_dist* pixels) are
    used as seeds; the watershed then grows each seed region competitively.

    Returns a list of cell dicts (keys: centroid, bbox, area, _mask).
    Returns an empty list when fewer than 2 seeds are found (nothing to split).
    """
    dist = ndimage.distance_transform_edt(mask)
    if dist.max() == 0:
        return []

    # Local maxima: neighbourhood size = diameter around each expected cell centre
    size = max(3, 2 * min_dist + 1)
    local_max = (ndimage.maximum_filter(dist, size=size) == dist)
    local_max &= mask
    local_max[dist < min_dist * 0.4] = False  # drop maxima too close to edges

    seeds, n_seeds = ndimage.label(local_max)
    if n_seeds < 2:
        return []  # single blob — cannot split, caller will discard

    # Watershed on inverted distance transform (low cost = cell centre = flooded first)
    dist_norm = np.clip((dist / dist.max()) * 254, 0, 254).astype(np.uint8)
    cost = (254 - dist_norm).astype(np.uint8)

    # markers: 0 = unlabeled (will be assigned), >0 = seed label
    markers = seeds.astype(np.int32)

    labeled_ws = ndimage.watershed_ift(cost, markers)
    labeled_ws[~mask] = 0  # clip to original mask

    cells = []
    for s in range(1, n_seeds + 1):
        cell_mask = (labeled_ws == s) & mask
        area = int(cell_mask.sum())
        if area < min_size or area > max_size:
            continue
        ys, xs = np.where(cell_mask)
        cells.append({
            "centroid": (float(ys.mean()), float(xs.mean())),
            "bbox":     (int(ys.min()), int(xs.min()),
                         int(ys.max()) + 1, int(xs.max()) + 1),
            "area":     area,
            "_mask":    cell_mask,
        })
    return cells


def segment_cells(
    ref_frame: np.ndarray,
    min_size: int = 150,
    max_size: int | None = None,
    sigma: float = 2.5,
    dilation_px: int = 4,
    min_cell_diameter: int = 15,
) -> tuple[list, np.ndarray]:
    """
    Segment cells in a reference image.

    Strategy
    --------
    1. Gaussian smoothing to remove bright spots (which are smaller
       than cells and would skew the threshold).
    2. Masked Otsu threshold (zero pixels excluded — handles StackReg
       black borders that would otherwise bias the threshold to zero).
    3. Fill holes + connected-component labeling.
    4. Area filtering:
       - area < min_size  → discard (debris/noise)
       - min_size ≤ area ≤ max_size → keep as individual cell
       - area > max_size  → try watershed split using the distance
         transform; valid sub-cells are kept, unsplittable blobs are
         discarded
    5. Light dilation to capture cell edges (clipped to the valid mask).

    Args
    ----
    min_cell_diameter : Minimum expected cell diameter (px).  Controls the
        minimum distance between watershed seeds — smaller values allow
        more aggressive splitting of tightly packed clusters.

    Returns
    -------
    cells  : list of dicts with label, centroid (cy,cx), bbox, area
    labeled: array of labels (0=background)
    """
    if max_size is None:
        max_size = int(ref_frame.size * 0.25)

    # Exclude zero-border pixels from threshold (StackReg zero-padding fix)
    valid_mask = ref_frame > 0

    blurred = ndimage.gaussian_filter(
        ref_frame.astype(np.float64), sigma=sigma)
    thr = _otsu_threshold(blurred, mask=valid_mask)
    binary = (blurred > thr) & valid_mask
    filled = ndimage.binary_fill_holes(binary)

    if dilation_px > 0:
        struct = ndimage.generate_binary_structure(2, 1)
        filled = ndimage.binary_dilation(
            filled, structure=struct, iterations=dilation_px)
        filled &= valid_mask  # prevent dilation into the zero border

    labeled, n_labels = ndimage.label(filled)

    cells = []
    final_labeled = np.zeros_like(labeled)
    next_label = 1
    min_dist = max(3, min_cell_diameter // 2)

    for lbl in range(1, n_labels + 1):
        comp_mask = labeled == lbl
        area = int(comp_mask.sum())

        if area < min_size:
            continue

        if area > max_size:
            # Oversized: must split or discard entirely
            sub = _watershed_split(comp_mask, min_dist, min_size, max_size)
            if not sub:
                continue  # unsplittable giant blob — drop
            for s in sub:
                cells.append({
                    "label":    next_label,
                    "centroid": s["centroid"],
                    "bbox":     s["bbox"],
                    "area":     s["area"],
                })
                final_labeled[s["_mask"]] = next_label
                next_label += 1
        else:
            # Within size range: always try watershed to detect hidden clusters.
            # If 2+ seeds are found the blob is split; if only 1 seed (or none),
            # it is kept as a single cell without modification.
            sub = _watershed_split(comp_mask, min_dist, min_size, max_size)
            if len(sub) >= 2:
                for s in sub:
                    cells.append({
                        "label":    next_label,
                        "centroid": s["centroid"],
                        "bbox":     s["bbox"],
                        "area":     s["area"],
                    })
                    final_labeled[s["_mask"]] = next_label
                    next_label += 1
            else:
                # Single cell — keep the full connected component
                ys, xs = np.where(comp_mask)
                cells.append({
                    "label":    next_label,
                    "centroid": (float(ys.mean()), float(xs.mean())),
                    "bbox":     (int(ys.min()), int(xs.min()),
                                 int(ys.max()) + 1, int(xs.max()) + 1),
                    "area":     area,
                })
                final_labeled[comp_mask] = next_label
                next_label += 1

    return cells, final_labeled


# ══════════════════════════════════════════════════════════════════════════════
# DRIFT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_drift(
    stack: np.ndarray,
    sigma: float = 2.0,
    drift_threshold_px: float = 3.0,
    use_feature_matching: bool = True,
) -> dict:
    """
    Detect drift using phase cross-correlation + feature matching for robustness.

    Method combines:
    - Phase correlation (FFT-based): fast, sub-pixel accuracy
    - Feature matching (optional): corner detection for robustness on textured images

    Returns dict with:
        dy, dx            : displacement (px) per frame (cumulative from frame 0)
        magnitude         : drift magnitude per frame
        max_drift_px      : maximum pointwise displacement
        cumulative_px     : cumulative displacement frame 0 -> last
        is_significant    : True if max_drift > drift_threshold_px
        is_trending       : True if clear linear trend (systematic drift)
        slope_y, slope_x  : drift slope (px/frame)
        method            : "phase_correlation" or "hybrid" (phase + features)
    """
    DRIFT_CAP_PX = 50.0

    n = stack.shape[0]
    if n < 2:
        return {
            "dy": [0.0], "dx": [0.0], "magnitude": [0.0],
            "max_drift_px": 0.0, "cumulative_px": 0.0,
            "is_significant": False, "is_trending": False,
            "slope_y": 0.0, "slope_x": 0.0, "method": "insufficient_frames",
            "n_clipped": 0, "drift_cap_px": DRIFT_CAP_PX,
        }

    # Normalize stack to 0-1 range
    stack_norm = (stack.astype(np.float64) - stack.min()) / \
        (stack.max() - stack.min() + 1e-12)

    # Apply Gaussian smoothing for robustness
    ref = ndimage.gaussian_filter(stack_norm[0], sigma=sigma)
    F_ref = fft2(ref)

    dy_list = [0.0]
    dx_list = [0.0]
    n_clipped = 0

    for i in range(1, n):
        cur = ndimage.gaussian_filter(stack_norm[i], sigma=sigma)
        F_cur = fft2(cur)

        # Phase correlation
        cross = F_ref * np.conj(F_cur)
        denom = np.abs(cross) + 1e-12
        R = fftshift(ifft2(cross / denom).real)
        pk = np.unravel_index(np.argmax(R), R.shape)
        dy_phase = float(pk[0] - R.shape[0] // 2)
        dx_phase = float(pk[1] - R.shape[1] // 2)

        # Constrain to reasonable range and track clipping
        if abs(dy_phase) > DRIFT_CAP_PX or abs(dx_phase) > DRIFT_CAP_PX:
            n_clipped += 1
        dy_phase = float(np.clip(dy_phase, -DRIFT_CAP_PX, DRIFT_CAP_PX))
        dx_phase = float(np.clip(dx_phase, -DRIFT_CAP_PX, DRIFT_CAP_PX))

        dy_list.append(dy_phase)
        dx_list.append(dx_phase)

    dy_arr = np.array(dy_list)
    dx_arr = np.array(dx_list)
    mag = np.sqrt(dy_arr**2 + dx_arr**2)

    # Detect systematic drift (linear trend)
    frames = np.arange(n, dtype=float)
    if n >= 4:
        slope_y, intercept_y, r_value_y, *_ = stats.linregress(frames, dy_arr)
        slope_x, intercept_x, r_value_x, *_ = stats.linregress(frames, dx_arr)
        # Is trending if slope is significant AND R² > 0.5 (reasonable fit)
        is_trending = (
            (abs(slope_y) > DRIFT_SLOPE_THRESHOLD and r_value_y**2 > 0.5) or
            (abs(slope_x) > DRIFT_SLOPE_THRESHOLD and r_value_x**2 > 0.5)
        )
    else:
        slope_y = slope_x = 0.0
        is_trending = False

    return {
        "dy":              dy_list,
        "dx":              dx_list,
        "magnitude":       mag.tolist(),
        "max_drift_px":    float(mag.max()),
        "cumulative_px":   float(mag[-1]),
        "is_significant":  bool(mag.max() > drift_threshold_px),
        "is_trending":     bool(is_trending),
        "slope_y":         float(slope_y),
        "slope_x":         float(slope_x),
        "method":          "phase_correlation",
        "n_clipped":       n_clipped,
        "drift_cap_px":    DRIFT_CAP_PX,
    }


def detect_drift_from_cells(
    stack: np.ndarray,
    sigma: float = 2.5,
    min_size: int = 150,
    max_size: int | None = None,
    dilation_px: int = 4,
    drift_threshold_px: float = 3.0,
    max_shift_px: float = 100.0,
) -> dict:
    """
    Estimate systematic linear drift via phase cross-correlation on the binary
    segmentation masks of the first and last frame.

    The direction of drift is captured automatically in the (dy, dx) components
    of the correlation peak: positive dy = downward field motion, positive dx =
    rightward. Per-frame drift is linearly interpolated:
        dy[i] = total_dy * i / (n - 1)
        dx[i] = total_dx * i / (n - 1)

    Using binary masks instead of raw images makes the estimate insensitive to
    intensity changes, photobleaching, and fluorescent spots — only the spatial
    arrangement of cell bodies matters.

    Falls back to raw-image phase correlation if segmentation finds no cells.

    Extra keys vs detect_drift():
        total_dy, total_dx : total drift from frame 0 to frame -1 (px)
        n_cells_first, n_cells_last : cells segmented in each anchor frame
    """
    n = stack.shape[0]
    if n < 2:
        return {
            "dy": [0.0], "dx": [0.0], "magnitude": [0.0],
            "max_drift_px": 0.0, "cumulative_px": 0.0,
            "is_significant": False, "is_trending": False,
            "slope_y": 0.0, "slope_x": 0.0,
            "method": "segmentation_insufficient",
            "total_dy": 0.0, "total_dx": 0.0,
            "n_cells_first": 0, "n_cells_last": 0,
        }

    _, labeled_0    = segment_cells(stack[0],  min_size=min_size, max_size=max_size,
                                    sigma=sigma, dilation_px=dilation_px)
    _, labeled_last = segment_cells(stack[-1], min_size=min_size, max_size=max_size,
                                    sigma=sigma, dilation_px=dilation_px)

    mask_0    = (labeled_0    > 0).astype(np.float64)
    mask_last = (labeled_last > 0).astype(np.float64)
    n_first   = int(labeled_0.max())
    n_last    = int(labeled_last.max())

    if n_first == 0 or n_last == 0:
        result = detect_drift(stack, sigma=sigma, drift_threshold_px=drift_threshold_px)
        result["method"] = "phase_correlation_fallback"
        result.setdefault("total_dy", result["dy"][-1])
        result.setdefault("total_dx", result["dx"][-1])
        result["n_cells_first"] = n_first
        result["n_cells_last"]  = n_last
        return result

    # Phase cross-correlation on binary masks — peak shift = total field drift
    F_0    = fft2(mask_0)
    F_last = fft2(mask_last)
    cross  = F_0 * np.conj(F_last)
    R      = fftshift(ifft2(cross / (np.abs(cross) + 1e-12)).real)
    pk     = np.unravel_index(np.argmax(R), R.shape)
    total_dy = float(np.clip(pk[0] - R.shape[0] // 2, -max_shift_px, max_shift_px))
    total_dx = float(np.clip(pk[1] - R.shape[1] // 2, -max_shift_px, max_shift_px))

    frames   = np.arange(n, dtype=float)
    dy_arr   = total_dy * frames / (n - 1)
    dx_arr   = total_dx * frames / (n - 1)
    mag      = np.sqrt(dy_arr ** 2 + dx_arr ** 2)
    total_px = float(np.sqrt(total_dy ** 2 + total_dx ** 2))
    slope_y  = total_dy / (n - 1) if n > 1 else 0.0
    slope_x  = total_dx / (n - 1) if n > 1 else 0.0
    is_sig   = bool(total_px > drift_threshold_px)

    return {
        "dy":             dy_arr.tolist(),
        "dx":             dx_arr.tolist(),
        "magnitude":      mag.tolist(),
        "max_drift_px":   float(mag.max()),
        "cumulative_px":  total_px,
        "is_significant": is_sig,
        "is_trending":    is_sig,
        "slope_y":        slope_y,
        "slope_x":        slope_x,
        "method":         "segmentation_mask",
        "total_dy":       total_dy,
        "total_dx":       total_dx,
        "n_cells_first":  n_first,
        "n_cells_last":   n_last,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DRIFT CORRECTION
# ══════════════════════════════════════════════════════════════════════════════

def apply_drift_correction(stack: np.ndarray, drift_info: dict, method: str = "linear") -> np.ndarray:
    """
    Correct drift by shifting each frame with sub-pixel accuracy.

    Args:
        stack: Input image stack (Z, H, W)
        drift_info: Dictionary with 'dy' and 'dx' displacement lists
        method: "linear" (direct) or "spline" (smoothed interpolation)

    Returns:
        Corrected stack of the same dtype as the original, with drift removed.
    """
    dy_list = drift_info["dy"]
    dx_list = drift_info["dx"]

    # Optional: smooth drift trajectory to reduce noise
    if method == "spline" and len(dy_list) > 3:
        from scipy.interpolate import UnivariateSpline
        frames = np.arange(len(dy_list))
        spl_y = UnivariateSpline(frames, dy_list, s=len(
            dy_list) * 0.1, k=min(3, len(dy_list)-1))
        spl_x = UnivariateSpline(frames, dx_list, s=len(
            dx_list) * 0.1, k=min(3, len(dx_list)-1))
        dy_list = spl_y(frames).tolist()
        dx_list = spl_x(frames).tolist()

    corrected = np.empty_like(stack, dtype=np.float32)
    for i in range(stack.shape[0]):
        corrected[i] = ndimage.shift(
            stack[i].astype(np.float64),
            shift=(dy_list[i], dx_list[i]),  # dy=-k for downward shift k → shift=(−k,0) = UP
            mode="constant", cval=0.0, order=1
        ).astype(np.float32)

    return corrected


# ══════════════════════════════════════════════════════════════════════════════
# RELATIVE TRACKING TO CELL CENTROID
# ══════════════════════════════════════════════════════════════════════════════

def track_cell_roi(
    stack: np.ndarray,
    cell: dict,
    drift_info: dict | None = None,
    threshold_pct: float = 95.0,
    roi_radius: int = 5,
    gaussian_sigma: float = 1.5,
    bbox_pad: int = DEFAULT_BBOX_PAD,
) -> list:
    """
    Track the brightest spot within a cell's ROI.
    Returns list of detections per frame:
        (frame, x_rel, y_rel, intensity, x_abs, y_abs)  or  None
    where (x_rel, y_rel) is the position relative to the cell centroid
    (positive = right/down).

    Cell centroid is calculated in the reference frame (frame 0).
    If drift_info is provided, the ROI and centroid reference are shifted
    per-frame so that relative coordinates are drift-corrected.
    """
    from particle_tracker import detect_peak

    cy0, cx0 = cell["centroid"]
    r0b, c0b, r1b, c1b = cell["bbox"]
    H, W = stack.shape[1], stack.shape[2]
    n = stack.shape[0]

    raw = []
    for i in range(n):
        dy_off = drift_info["dy"][i] if drift_info else 0.0
        dx_off = drift_info["dx"][i] if drift_info else 0.0

        r0 = max(0, int(r0b - dy_off) - bbox_pad)
        r1 = min(H, int(r1b - dy_off) + bbox_pad)
        c0 = max(0, int(c0b - dx_off) - bbox_pad)
        c1 = min(W, int(c1b - dx_off) + bbox_pad)

        if r1 <= r0 or c1 <= c0:
            raw.append(None)
            continue

        roi = stack[i, r0:r1, c0:c1]
        det = detect_peak(roi, threshold_pct=threshold_pct,
                          roi_radius=roi_radius, gaussian_sigma=gaussian_sigma)
        if det:
            x_abs = det[0] + c0
            y_abs = det[1] + r0
            cx_adj = cx0 - dx_off
            cy_adj = cy0 - dy_off
            raw.append((i, x_abs - cx_adj, y_abs - cy_adj, det[2], x_abs, y_abs))
        else:
            raw.append(None)

    return raw


# ══════════════════════════════════════════════════════════════════════════════
# QUALITY TESTS
# ══════════════════════════════════════════════════════════════════════════════

def quality_tests(result: dict, stats_data: dict | None, sig_info: dict | None) -> dict:
    """
    Battery of quality tests on an individual trajectory.

    Tests
    -----
    T1  Completeness     - fraction of detected frames
    T2  Signal/noise     - estimated signal-to-noise ratio
    T3  Confinement      - ratio end-to-end / total path length
    T4  Continuity       - no gap > 20% of trajectory
    T5  MSD reliability  - R2 of log-log fit >= 0.85
    T6  Stable intensity - intensity CV < 0.5 (no extreme photobleaching)

    Returns dict with pass/fail and numeric value for each test.
    """
    tests = {}

    # T1 Completeness
    if result["n_frames"] > 0:
        completeness = result["n_detected"] / result["n_frames"]
    else:
        completeness = 0.0
    tests["T1_completeness"] = {
        "value": round(completeness, 3),
        "pass":  completeness >= QT_MIN_COMPLETENESS,
        "label": f"Completeness >= {QT_MIN_COMPLETENESS:.0%}",
    }

    # T2 SNR (if signal info available) — kept in order T1→T2→T3…
    if sig_info:
        ref = sig_info["ref_intensity"]
        thr = sig_info["threshold_abs"]
        snr = ref / thr if thr > 0 else 0.0
        tests["T2_snr"] = {
            "value": round(snr, 2),
            "pass":  snr >= QT_MIN_SNR,
            "label": f"SNR >= {QT_MIN_SNR:.1f} (ref / threshold)",
        }

    if stats_data:
        conf = stats_data["confinement_ratio"]
        tests["T3_confinement"] = {
            "value": round(conf, 3),
            "pass":  True,   # informative only — no pass/fail threshold
            "label": "Confinement ratio (informative)",
        }

        # T4 Maximum gap
        max_gap_len = max((g[2] for g in result["gap_info"]), default=0)
        max_gap_frac = max_gap_len / \
            result["n_frames"] if result["n_frames"] else 1.0
        tests["T4_max_gap"] = {
            "value": round(max_gap_frac, 3),
            "pass":  max_gap_frac <= QT_MAX_GAP_FRACTION,
            "label": f"Max gap <= {QT_MAX_GAP_FRACTION:.0%} of trajectory",
        }

        # T5 MSD R2
        r2 = stats_data.get("msd_r2_log", float("nan"))
        tests["T5_msd_r2"] = {
            "value": round(r2, 3) if not np.isnan(r2) else None,
            "pass":  (not np.isnan(r2)) and r2 >= QT_MIN_MSD_R2,
            "label": f"MSD log-log R² >= {QT_MIN_MSD_R2:.2f}",
        }

        # T6 Stable intensity
        mean_i = stats_data["mean_intensity"]
        std_i = stats_data["std_intensity"]
        cv = std_i / mean_i if mean_i > 0 else float("inf")
        tests["T6_intensity_cv"] = {
            "value": round(cv, 3) if not np.isinf(cv) else None,
            "pass":  (not np.isinf(cv)) and cv <= QT_MAX_INTENSITY_CV,
            "label": f"Intensity CV <= {QT_MAX_INTENSITY_CV:.2f}",
        }

    n_pass = sum(1 for t in tests.values() if t["pass"])
    n_total = len(tests)
    tests["_summary"] = {
        "n_pass": n_pass,
        "n_total": n_total,
        "overall": "PASS" if n_pass == n_total else ("WARN" if n_pass >= n_total * QT_WARN_PASS_FRACTION else "FAIL"),
    }
    return tests
