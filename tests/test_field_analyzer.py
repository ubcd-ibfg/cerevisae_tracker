"""Tests for field_analyzer: segmentation, drift detection and quality tests."""
from __future__ import annotations

import numpy as np
import pytest

import field_analyzer as fa


# ── Reference frame & segmentation ────────────────────────────────────────────

def test_make_reference_frame_uses_median_of_first_n():
    """First n_ref frames are bright, later ones black → median is bright."""
    n_ref = 5
    bright = np.full((n_ref, 16, 16), 100.0, dtype=np.float32)
    dark = np.zeros((10, 16, 16), dtype=np.float32)
    stack = np.concatenate([bright, dark], axis=0)
    ref = fa.make_reference_frame(stack, n_ref=n_ref)
    assert ref.shape == (16, 16)
    assert ref.mean() == pytest.approx(100.0)


def test_segment_cells_finds_two_blobs():
    """Two well-separated Gaussian blobs → exactly two segmented cells."""
    img = np.zeros((64, 64), dtype=np.float32)
    yy, xx = np.mgrid[:64, :64]
    img += 200.0 * np.exp(-((yy - 16) ** 2 + (xx - 16) ** 2) / (2 * 4.0 ** 2))
    img += 200.0 * np.exp(-((yy - 48) ** 2 + (xx - 48) ** 2) / (2 * 4.0 ** 2))
    cells, labeled = fa.segment_cells(img, min_size=20, sigma=1.5, dilation_px=2)
    assert len(cells) == 2
    centroids = sorted([c["centroid"] for c in cells])
    assert centroids[0][0] < centroids[1][0]


def test_segment_cells_respects_max_size():
    """A very large blob is excluded when max_size is tight."""
    img = np.zeros((64, 64), dtype=np.float32)
    yy, xx = np.mgrid[:64, :64]
    img += 200.0 * np.exp(-((yy - 32) ** 2 + (xx - 32) ** 2) / (2 * 8.0 ** 2))
    cells, _ = fa.segment_cells(img, min_size=10, max_size=50,
                                sigma=1.5, dilation_px=0)
    assert cells == []


def test_segment_cells_splits_touching_blobs_via_watershed():
    """When neighbouring cells touch and the standard pipeline produces a
    single merged blob above ``max_size``, the watershed fallback must split
    it into the individual cells.

    Build three Gaussians whose tails overlap so that an Otsu + dilation
    pipeline puts them in one connected component that exceeds max_size=600.
    Without the fallback, segment_cells used to return 0 cells.
    """
    img = np.full((64, 96), 30.0, dtype=np.float32)
    yy, xx = np.mgrid[:64, :96]
    centres = [(32, 24), (32, 48), (32, 72)]
    for cy, cx in centres:
        img += 220.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 5.0 ** 2))

    cells, labeled = fa.segment_cells(img, min_size=50, max_size=600,
                                      sigma=2.0, dilation_px=2)
    assert len(cells) == 3, f"expected 3 cells, got {len(cells)}"
    # Each cell centroid must match (within a few pixels) one of the input
    # Gaussian centres.
    cell_centroids = sorted((round(c["centroid"][1]), round(c["centroid"][0]))
                            for c in cells)
    expected = sorted((cx, cy) for cy, cx in centres)
    for (got_x, got_y), (exp_x, exp_y) in zip(cell_centroids, expected):
        assert abs(got_x - exp_x) <= 3 and abs(got_y - exp_y) <= 3, \
            f"centroid {got_x, got_y} far from expected {exp_x, exp_y}"


def test_segment_cells_robust_to_stackreg_zero_padding():
    """Reference frame with a wide zero-padded border (StackReg-style) must
    still segment cells correctly.

    Without masking, the zero-padding creates a histogram spike at 0 that
    drags Otsu's threshold towards zero. Otsu then declares everything that
    isn't exactly 0 as foreground, which collapses the whole inner region
    into one connected component (rejected by max_size) — yielding zero
    segmented cells. This used to break Mode B for any field corrected with
    StackReg.
    """
    # 128×128 frame, central 80×80 valid region with two Gaussian blobs and
    # a low non-zero background; the outer border is hard zeros.
    img = np.zeros((128, 128), dtype=np.float32)
    inner = slice(24, 104)
    img[inner, inner] = 30.0  # background inside the StackReg-valid region
    yy, xx = np.mgrid[:128, :128]
    img += 250.0 * np.exp(-((yy - 50) ** 2 + (xx - 50) ** 2) / (2 * 5.0 ** 2))
    img += 250.0 * np.exp(-((yy - 80) ** 2 + (xx - 80) ** 2) / (2 * 5.0 ** 2))
    # Re-zero the outer border (Gaussians spread a tiny bit, but pixels
    # outside the inner region should be exactly 0 like StackReg output).
    border = np.ones((128, 128), dtype=bool)
    border[inner, inner] = False
    img[border] = 0.0

    cells, labeled = fa.segment_cells(img, min_size=30, sigma=2.0, dilation_px=2)
    assert len(cells) == 2, f"expected 2 cells, got {len(cells)}"

    # None of the segmented pixels should land on the zero-padded border.
    foreground = labeled > 0
    assert not (foreground & border).any(), "segmentation leaked into the zero border"


# ── Drift detection ───────────────────────────────────────────────────────────

def test_detect_drift_zero_for_static_stack():
    """Identical frames → no drift detected."""
    rng = np.random.default_rng(0)
    frame = rng.standard_normal((48, 48)).astype(np.float32) * 5 + 100
    stack = np.stack([frame] * 8, axis=0)
    info = fa.detect_drift(stack, sigma=1.0, drift_threshold_px=1.0)
    assert info["max_drift_px"] < 1.0
    assert info["is_significant"] is False
    assert info["n_clipped"] == 0


def test_detect_drift_recovers_synthetic_shift():
    """Hand-built shift sequence: each frame shifted (1, 0) more than the
    previous → cumulative drift grows linearly in y.

    Sign convention: ``detect_drift`` uses ``F_ref * conj(F_cur)`` so a
    frame shifted DOWN by ``k`` pixels reports ``dy = -k``. The magnitude
    must equal the true shift; the negative sign is documented and consumed
    consistently by ``apply_drift_correction``.
    """
    base = np.zeros((64, 64), dtype=np.float32)
    yy, xx = np.mgrid[:64, :64]
    base += 200.0 * np.exp(-((yy - 32) ** 2 + (xx - 32) ** 2) / (2 * 3.0 ** 2))
    frames = []
    for k in range(6):
        f = np.roll(base, shift=k, axis=0)  # shift down by k pixels
        frames.append(f)
    stack = np.stack(frames, axis=0)
    info = fa.detect_drift(stack, sigma=1.0, drift_threshold_px=1.0)
    # Magnitude is correct; sign follows the F_ref * conj(F_cur) convention.
    assert info["dy"][-1] == pytest.approx(-5.0, abs=1.0)
    assert abs(info["dx"][-1]) < 1.5
    assert info["is_significant"] is True


def test_detect_then_correct_recovers_original():
    """End-to-end pipeline contract: detect → correct must put the blob
    back at its original location.

    Reproduce: blob at (32, 32); frame 1 shifted down by 5 → blob at (37, 32).
    After ``apply_drift_correction`` the blob must land within ~1 px of the
    original (32, 32). This used to be broken: the sign of the shift inside
    ``apply_drift_correction`` was inverted, doubling the drift to (42, 32).
    """
    base = np.zeros((64, 64), dtype=np.float32)
    yy, xx = np.mgrid[:64, :64]
    base += 200.0 * np.exp(-((yy - 32) ** 2 + (xx - 32) ** 2) / 18.0)
    shifted = np.roll(base, shift=5, axis=0)
    stack = np.stack([base, shifted], axis=0)
    info = fa.detect_drift(stack, sigma=1.0, drift_threshold_px=1.0)
    corrected = fa.apply_drift_correction(stack, info)
    peak_corrected = np.unravel_index(np.argmax(corrected[1]), corrected[1].shape)
    assert abs(peak_corrected[0] - 32) <= 1
    assert abs(peak_corrected[1] - 32) <= 1


def test_detect_drift_reports_clipped_count():
    """When the synthetic drift exceeds the ±50 px cap, n_clipped > 0."""
    base = np.zeros((128, 128), dtype=np.float32)
    yy, xx = np.mgrid[:128, :128]
    base += 200.0 * np.exp(-((yy - 64) ** 2 + (xx - 64) ** 2) / (2 * 3.0 ** 2))
    # 60 px shift between frame 0 and frame 1 (above the 50 px cap).
    frames = [base, np.roll(base, shift=60, axis=0)]
    stack = np.stack(frames, axis=0)
    info = fa.detect_drift(stack, sigma=1.0, drift_threshold_px=1.0)
    assert info["n_clipped"] >= 1
    assert info["drift_cap_px"] == 50.0


# ── apply_drift_correction ────────────────────────────────────────────────────

def test_apply_drift_correction_returns_float32():
    """Docstring contract: output dtype is float32 regardless of input."""
    stack = np.full((4, 16, 16), 100, dtype=np.uint16)
    drift = {"dy": [0.0, 0.5, 1.0, 1.5], "dx": [0.0, 0.0, 0.0, 0.0]}
    out = fa.apply_drift_correction(stack, drift)
    assert out.dtype == np.float32
    assert out.shape == stack.shape


def test_apply_drift_correction_undoes_known_shift():
    """A frame shifted DOWN by 3 and RIGHT by 2 should land back near the
    original when given the matching ``detect_drift``-style drift_info,
    where positive shift produces *negative* dy/dx (F_ref * conj(F_cur))."""
    base = np.zeros((32, 32), dtype=np.float32)
    yy, xx = np.mgrid[:32, :32]
    base += 100.0 * np.exp(-((yy - 16) ** 2 + (xx - 16) ** 2) / 4.0)
    shifted = np.roll(np.roll(base, 3, axis=0), 2, axis=1)
    stack = np.stack([base, shifted], axis=0)
    # detect_drift convention: a frame shifted DOWN by 3 yields dy = -3.
    drift = {"dy": [0.0, -3.0], "dx": [0.0, -2.0]}
    corrected = fa.apply_drift_correction(stack, drift)
    diff = np.abs(corrected[1, 5:-5, 5:-5] - base[5:-5, 5:-5]).mean()
    assert diff < 1.0  # within sub-pixel interpolation noise


# ── Quality tests ─────────────────────────────────────────────────────────────

def _make_result(n_frames, n_detected, gaps=()):
    return {
        "n_frames": n_frames,
        "n_detected": n_detected,
        "gap_info": list(gaps),
        "missing_ratio": (n_frames - n_detected) / max(n_frames, 1),
    }


def test_quality_tests_pass_when_everything_clean():
    result = _make_result(n_frames=100, n_detected=100)
    sd = {
        "confinement_ratio": 0.5,
        "mean_intensity": 100.0,
        "std_intensity": 10.0,        # CV = 0.1, well below 0.6
        "msd_r2_log": 0.99,
    }
    sig = {"ref_intensity": 1000.0, "threshold_abs": 100.0}  # SNR = 10
    qt = fa.quality_tests(result, sd, sig)
    assert qt["_summary"]["overall"] == "PASS"
    # Order check: T1 first, then T2 SNR, then T3..T6
    keys = [k for k in qt if k != "_summary"]
    assert keys == ["T1_completeness", "T2_snr", "T3_confinement",
                    "T4_max_gap", "T5_msd_r2", "T6_intensity_cv"]


def test_quality_tests_fail_on_low_completeness():
    result = _make_result(n_frames=100, n_detected=30)  # 30 %
    sd = {
        "confinement_ratio": 0.5,
        "mean_intensity": 100.0,
        "std_intensity": 10.0,
        "msd_r2_log": 0.99,
    }
    sig = {"ref_intensity": 1000.0, "threshold_abs": 100.0}
    qt = fa.quality_tests(result, sd, sig)
    assert qt["T1_completeness"]["pass"] is False
    assert qt["_summary"]["overall"] in ("WARN", "FAIL")


def test_quality_tests_handles_infinite_cv():
    """mean_intensity == 0 → CV = inf → 'value' should be None, not 'inf'."""
    result = _make_result(n_frames=10, n_detected=10)
    sd = {
        "confinement_ratio": 0.5,
        "mean_intensity": 0.0,
        "std_intensity": 5.0,
        "msd_r2_log": 0.95,
    }
    qt = fa.quality_tests(result, sd, sig_info=None)
    assert qt["T6_intensity_cv"]["value"] is None
    assert qt["T6_intensity_cv"]["pass"] is False
