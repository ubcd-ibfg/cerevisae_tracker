# Particle Tracker — Multi-Mode Cell Trajectory Analysis

A Streamlit application for tracking fluorescent particles/organelles in time-lapse microscopy images.  Supports three modes: single-cell tracking (Mode A), full-field batch analysis (Mode B), and statistical comparison of conditions (Mode C).

**Version:** 0.4.0 | **Language:** Python 3.8+

---

## Features

### Mode A — Single-Cell Analysis
- Upload one TIFF per cell (multi-frame stack)
- Automatic signal-end detection to discard photobleached frames
- Gaussian-smoothed peak detection with configurable intensity threshold
- Gap interpolation for missing detections
- 3-panel figure: trajectory overlay · MSD curve · step-size histogram
- Log-log MSD fit → α exponent + motion-type classification
- Animated GIF export (annotated overlay)
- Per-cell CSV, text report, and ZIP bundle

### Mode B — Full-Field Batch Analysis
- Upload one TIFF per field (multiple cells per frame)
- Otsu segmentation on the temporal median projection
- Phase cross-correlation drift detection per field
- Optional drift correction (bilinear interpolation)
- Manual cell inclusion/exclusion before tracking
- Relative-coordinate tracking (position relative to cell centroid)
- Automated quality-control battery (6 tests: completeness, SNR, confinement, max gap, MSD R², intensity CV)
- Per-field and all-fields ZIP export

### Mode C — Statistical Analysis
- Upload CSV exports from Mode A or B (two conditions)
- Per-cell metrics: mean speed, max speed, total path length, end-to-end displacement, confinement ratio, MSD alpha, interpolation fraction
- Optional equal-N subsampling for balanced comparison
- Mann-Whitney U tests (two-sided) with FDR-corrected p-values
- Violin + strip-plot grid for all metrics
- Ensemble MSD with mean ± SEM and reference slope overlays
- Unsupervised K-means pattern detection on trajectory windows:
  - Selectable feature set (mean speed, speed variability, straightness, angular persistence, local alpha)
  - Interactive cluster selection and profile visualization
  - Chi-square test for cluster frequency differences between conditions

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/particle-tracker.git
cd particle-tracker

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install streamlit numpy scipy pandas pillow matplotlib tifffile

# 4. Launch
streamlit run app_tracker.py
```

The app opens at `http://localhost:8501`.

### Dependencies

| Package | Purpose |
|---------|---------|
| streamlit | Web UI |
| numpy | Array operations |
| scipy | FFT drift detection, stats tests, K-means |
| pandas | DataFrames, CSV export |
| pillow (PIL) | GIF generation |
| matplotlib | All figures |
| tifffile | TIFF read/write (raw-crop export) |

> **Note:** `scikit-image` is *not* required — Otsu thresholding is implemented internally.

---

## Usage

### Running

```bash
streamlit run app_tracker.py
```

Switch between modes with the sidebar radio selector.

---

## Mode A — Single-Cell Analysis

### Workflow

1. Select **Single cell** from the sidebar.
2. Upload one or more TIFF stacks (one cell per file).
3. Configure parameters (see below).
4. Click **Analyze all cells**.

### Parameters

| Parameter | Description |
|-----------|-------------|
| Auto-cut | Enable automatic signal-end detection |
| Signal threshold % | Intensity percentile below which the signal is considered lost |
| Consecutive frames | Number of sub-threshold frames that trigger the cutoff |
| Trace smoothing | Gaussian window (frames) applied before threshold detection |
| Min intensity percentile | Detection threshold for the peak finder |
| ROI radius (px) | Search radius around each candidate peak |
| Gaussian sigma | Blur applied before peak detection |
| Max missing (%) | Maximum fraction of frames without a detection before discarding |
| Generate GIF | Enable/disable animated GIF output |
| Tail length | Number of frames shown as the trajectory tail in the GIF |
| FPS | GIF playback speed |

### Outputs

Per cell:
- **Trajectory figure** — 3-panel: path overlay on max-projection · MSD curve · step-size histogram
- **Signal trace** — raw and smoothed intensity with threshold and cutoff marker
- **MSD log-log plot** — power-law fit with motion-type annotation
- **Animated GIF** — frame-by-frame overlay
- **CSV** — `cell_id, frame, x_px, y_px, intensity, interpolated`
- **Text report** — parameters, status, motion statistics
- **ZIP** — all of the above for one cell, or all cells combined

---

## Mode B — Full-Field Batch Analysis

### Workflow

#### Phase 1 — Scan
1. Select **Full field (batch)**.
2. Upload one or more TIFF stacks (one field per file, multiple cells per frame).
3. Set segmentation and drift parameters.
4. Click **Scan fields (Phase 1)**.

#### Phase 2 — Review
1. Inspect the segmentation overlay; uncheck cells to exclude them.
2. Review the drift report for each field.
3. Choose a drift-handling strategy per field:
   - **Auto-correct** — apply sub-pixel drift correction
   - **Continue without correction**
   - **Skip field**
4. Click **Confirm and track all cells**.

#### Phase 3 — Results
Tracking, quality tests, and report generation run automatically.

### Parameters

| Parameter | Description |
|-----------|-------------|
| Min cell size (px²) | Minimum connected-component area to accept as a cell |
| Max cell size (px²) | Maximum area (rejects fusions and artefacts) |
| Segmentation smoothing | Gaussian sigma for the reference-frame blur |
| Edge dilation (px) | Morphological dilation to capture cell boundaries |
| Drift threshold (px) | Minimum drift magnitude to flag as significant |
| (Detection params) | Same as Mode A |

### Quality Tests

| Test | Metric | Default threshold |
|------|--------|-------------------|
| T1 Completeness | Fraction of frames with a detection | ≥ 70 % |
| T2 SNR | Reference intensity / threshold | ≥ 3.0 |
| T3 Confinement | End-to-end / total path (informative) | — |
| T4 Max gap | Largest gap / trajectory length | ≤ 20 % |
| T5 MSD R² | Log-log fit coefficient of determination | ≥ 0.80 |
| T6 Intensity CV | Coefficient of variation of intensity | ≤ 0.60 |

All thresholds are defined in `config.py`.

### Outputs

Per field:
- **Segmentation overview** — reference frame · labeled cell masks
- **Drift figure** — dx/dy per frame · magnitude · cumulative displacement
- **Per-cell trajectory figure** (relative coordinates)
- **Quality-test report** — PASS / WARN / FAIL per test
- **CSV** — `field_id, cell_id, frame, x_rel_px, y_rel_px, intensity, interpolated`
- **Text report**
- **Raw TIFF crops** — unprocessed cell crops (original bit-depth) for external QC
- **ZIP** — per-field or all-fields bundle

---

## Mode C — Statistical Analysis

### Workflow

1. Select **Statistical analysis**.
2. Upload one CSV for **Condition 1** and one for **Condition 2** (exports from Mode A or B).
3. Optionally adjust the N-matching slider (default: minimum N between conditions).
4. Click **Run statistical analysis** to compute metrics and tests.
5. Optionally click **Detect patterns** to run K-means clustering independently.

### Statistical Tests

- **Mann-Whitney U** (two-sided, non-parametric) — appropriate for independent samples with unknown distributions.
- p-values reported raw and after Benjamini-Hochberg FDR correction.

### Pattern Detection

Trajectory windows are featurized and clustered with K-means.

| Feature | Description |
|---------|-------------|
| mean_speed | Average step size in the window |
| speed_std | Variability of step sizes |
| straightness | End-to-end / total path for the window |
| ang_persistence | Mean cosine of consecutive turn angles |
| alpha_local | MSD power-law exponent fitted within the window |

### Outputs

- Violin + strip plots for all metrics
- Ensemble MSD log-log with ± SEM
- Cluster profile heatmap
- Cluster frequency bar chart with chi-square p-value
- Example trajectory segments per cluster

---

## Motion Classification

MSD power-law: MSD(τ) ∝ τ^α

| α range | Classification |
|---------|---------------|
| α < 0.7 | Subdiffusive / confined |
| 0.7 ≤ α ≤ 1.3 | Diffusive (Brownian) |
| 1.3 < α ≤ 1.8 | Superdiffusive |
| α > 1.8 | Ballistic / directed |

Thresholds are defined in `config.py` (`ALPHA_SUBDIFFUSIVE`, `ALPHA_SUPERDIFFUSIVE`, `ALPHA_BALLISTIC`).

---

## Project Structure

```
particle-tracker/
├── app_tracker.py       # Streamlit application (Modes A, B, C dispatcher)
├── particle_tracker.py  # Single-cell tracking: peak detection, MSD, statistics
├── field_analyzer.py    # Field analysis: segmentation, drift, quality tests
├── stats_page.py        # Mode C: statistical analysis and pattern detection
├── config.py            # All constants, thresholds, colours, session-state keys
└── README.md
```

### Module Responsibilities

| Module | Public API |
|--------|-----------|
| `particle_tracker.py` | `detect_peak`, `analyze_trajectory`, `compute_statistics` |
| `field_analyzer.py` | `make_reference_frame`, `segment_cells`, `detect_drift`, `apply_drift_correction`, `track_cell_roi`, `quality_tests` |
| `stats_page.py` | `render_stats_page`, `render_pattern_section` |
| `config.py` | constants only — import and use, do not modify at runtime |

---

## Customization

All numeric thresholds, colour hex codes, figure sizes, and session-state key strings are centralized in `config.py`. To change a default (e.g., the minimum completeness threshold or the accent colour), edit only that file.

---

## Tips

**Weak signal:**  raise the intensity percentile threshold; reduce Gaussian sigma for sharper peaks.

**Crowded fields:**  tighten min/max cell size to exclude debris and cell fusions; increase segmentation smoothing for smoother boundaries.

**High drift:**  always apply drift correction before tracking; verify the drift figure to confirm the correction worked.

**Statistical analysis:**  use equal-N matching when conditions have very different cell counts to avoid power imbalances; fix the random seed for reproducibility.

---

## References

- Saxton, M.J. & Jacobson, K. (1997). *Single-particle tracking: applications to membrane dynamics.* Annu Rev Biophys Biomol Struct 26, 373–399.
- Feder, T.J. et al. (1996). *Constrained diffusion or immobile fraction on cell surfaces.* Biophys J 70, 2767–2773.
- Qian, H. et al. (1991). *Single particle tracking: analysis of diffusion and flow in two-dimensional systems.* Biophys J 60, 910–921.

---

## License

[Specify your license here]

---

## Citation

If you use this software in your research, please cite:

```
Particle Tracker — Multi-Mode Cell Trajectory Analysis (v0.4.0)
[Authors], [Institution], [Year]
```

---

*Last updated: 2026-04-29*
