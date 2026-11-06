"""Tests for stats_page: CSV loading, BH-FDR, and per-cell metrics."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

import stats_page as sp


# ── _benjamini_hochberg ───────────────────────────────────────────────────────

def test_bh_fdr_known_values():
    """Reference values computed by hand:

    With pvals = [0.01, 0.02, 0.03, 0.04, 0.5] (m=5):
      raw rank/m * p : [0.05, 0.05, 0.05, 0.05, 0.5]
      monotonic from tail: [0.05, 0.05, 0.05, 0.05, 0.5]
    """
    adj = sp._benjamini_hochberg(np.array([0.01, 0.02, 0.03, 0.04, 0.5]))
    np.testing.assert_allclose(adj, [0.05, 0.05, 0.05, 0.05, 0.5], rtol=1e-9)


def test_bh_fdr_preserves_nan():
    adj = sp._benjamini_hochberg(np.array([0.01, np.nan, 0.5]))
    assert not np.isnan(adj[0])
    assert np.isnan(adj[1])
    assert not np.isnan(adj[2])


def test_bh_fdr_all_nan_returns_all_nan():
    adj = sp._benjamini_hochberg(np.array([np.nan, np.nan]))
    assert np.all(np.isnan(adj))


def test_bh_fdr_clamps_to_one():
    """Tiny m with large p shouldn't produce values > 1."""
    adj = sp._benjamini_hochberg(np.array([0.9, 0.95]))
    assert adj.max() <= 1.0


# ── _load_csv: Mode A vs Mode B ───────────────────────────────────────────────

MODE_A_CSV = """cell_id,frame,x_px,y_px,intensity,interpolated
A,0,1.0,2.0,100,False
A,1,1.5,2.5,101,False
A,2,2.0,3.0,102,False
"""

MODE_B_CSV = """field_id,cell_id,frame,x_rel_px,y_rel_px,intensity,interpolated,speed_px_frame
F1,C1,0,0.0,0.0,100,False,
F1,C1,1,1.0,0.0,101,False,1.0
F1,C1,2,2.0,0.0,102,False,1.0
"""


def test_load_csv_mode_a_normalises_schema():
    df = sp._load_csv(io.StringIO(MODE_A_CSV))
    # Mode A doesn't have field_id → synthesised as "single".
    assert "field_id" in df.columns
    assert (df["field_id"] == "single").all()
    # Coordinates renamed.
    assert "x_rel_px" in df.columns and "y_rel_px" in df.columns
    # Speed computed from successive positions: hypot(0.5, 0.5) = 0.707…
    speeds = df["speed_px_frame"].tolist()
    assert np.isnan(speeds[0])  # first frame has no previous
    assert speeds[1] == pytest.approx(np.sqrt(0.5), rel=1e-3)


def test_load_csv_mode_b_passes_through():
    df = sp._load_csv(io.StringIO(MODE_B_CSV))
    assert set(["field_id", "cell_id", "x_rel_px", "y_rel_px",
                "speed_px_frame", "interpolated"]).issubset(df.columns)
    assert df["field_id"].iloc[0] == "F1"


def test_load_csv_rejects_csv_missing_required_columns():
    bad = "foo,bar\n1,2\n"
    with pytest.raises(ValueError, match="Missing columns"):
        sp._load_csv(io.StringIO(bad))


# ── _compute_cell_metrics ─────────────────────────────────────────────────────

def _make_ballistic_df(n=40, cells=("C1", "C2"), interp_every=None):
    rows = []
    for cid in cells:
        for f in range(n):
            interpolated = bool(interp_every and f % interp_every == 0 and f != 0)
            rows.append({
                "field_id":       "F1",
                "cell_id":        cid,
                "frame":          f,
                "x_rel_px":       float(f),
                "y_rel_px":       0.0,
                "speed_px_frame": np.nan if f == 0 else 1.0,
                "interpolated":   interpolated,
                "intensity":      100.0,
            })
    return pd.DataFrame(rows)


def test_compute_cell_metrics_ballistic_alpha_close_to_2():
    df = _make_ballistic_df()
    metrics = sp._compute_cell_metrics(df)
    assert len(metrics) == 2
    for _, row in metrics.iterrows():
        assert row["alpha_msd"] == pytest.approx(2.0, abs=0.1)
        # End-to-end equals total path for a straight line.
        assert row["end_to_end"] == pytest.approx(row["total_path"], rel=1e-6)


def test_compute_cell_metrics_excludes_interpolated_pairs():
    """Alpha should still be ~2 even with some interpolated points, because
    the ballistic structure of the *real* pairs is unchanged."""
    df = _make_ballistic_df(interp_every=3)
    metrics = sp._compute_cell_metrics(df)
    for _, row in metrics.iterrows():
        assert row["alpha_msd"] == pytest.approx(2.0, abs=0.2)


def test_compute_cell_metrics_skips_cells_with_too_few_points():
    df = pd.DataFrame([{
        "field_id": "F1", "cell_id": "C1", "frame": 0,
        "x_rel_px": 0.0, "y_rel_px": 0.0,
        "speed_px_frame": np.nan, "interpolated": False, "intensity": 100,
    }])
    metrics = sp._compute_cell_metrics(df)
    assert metrics.empty
