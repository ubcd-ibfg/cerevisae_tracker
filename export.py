"""
export.py - ZIP archive creation and data export helpers for Particle Tracker.

Functions for generating downloadable ZIP files with trajectories, GIFs, reports,
and raw TIFF stacks for single cell, batch field, and statistical analyses.
"""

import io
import zipfile
import numpy as np
import pandas as pd
import tifffile
from PIL import Image

from field_analyzer import apply_drift_correction
from visualization import make_gif_roi


def build_selected_cell_rois_zip(selected_keys, results, fields, tail, fps):
    """Build a ZIP archive containing ROI GIFs for selected tracked cells.

    Args:
        selected_keys: List of strings with values like "field/cell".
        results: Batch tracking results dictionary.
        fields: Batch field scan dictionary with original stacks and drift info.
        tail: Number of frames to show in GIF tail.
        fps: Frames per second for GIF exports.

    Returns:
        bytes or None: ZIP archive bytes, or None if no ROI data was created.
    """
    if not selected_keys:
        return None
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for key in selected_keys:
            if "/" not in key:
                continue
            field_name, clabel = key.split("/", 1)
            fr = results.get(field_name)
            if not fr or fr.get("skipped"):
                continue
            cd = fr["cells"].get(clabel)
            if not cd or "error" in cd:
                continue
            fd = fields.get(field_name)
            if not fd:
                continue
            stack = fd["stack"]
            drift_used = fd["drift_info"]
            if fd.get("decision") == "correct":
                stack = apply_drift_correction(stack, drift_used)
                drift_used = None
            archive_name = f"{field_name}/{field_name}_{clabel}_roi.gif"
            if cd.get("gif"):
                zf.writestr(archive_name, cd["gif"])
                continue
            gif = make_gif_roi(
                stack, cd["cell_meta"], cd["raw_det"], cd["result"], drift_used, tail, fps)
            if gif:
                zf.writestr(archive_name, gif)
                continue
            bbox = cd["cell_meta"]["bbox"]
            pad = 20
            H, W = stack.shape[1], stack.shape[2]
            r0 = max(0, bbox[0] - pad)
            r1 = min(H, bbox[2] + pad)
            c0 = max(0, bbox[1] - pad)
            c1 = min(W, bbox[3] + pad)
            roi = stack[0, r0:r1, c0:c1]
            norm = ((roi - roi.min()) / max(roi.ptp(), 1e-8)
                    * 255).astype(np.uint8)
            img = Image.fromarray(norm)
            png_buf = io.BytesIO()
            img.save(png_buf, format="PNG")
            png_buf.seek(0)
            zf.writestr(
                f"{field_name}/{field_name}_{clabel}_frame0000.png", png_buf.read())
    buf.seek(0)
    return buf.read() if buf.getbuffer().nbytes > 0 else None


def build_raw_tif_zip(selected_keys, results, fields, pad=20):
    """ZIP of raw (unprocessed) TIFF stacks cropped around each selected cell.

    Crops use the original stack (before any drift correction) to preserve
    original bit-depth and pixel values. The crop window is fixed to the
    cell bbox from the reference frame, so any field drift will be visible
    in the exported TIFF — useful for manual QC.

    Each TIFF is a multi-frame stack (T, H, W) compatible with Fiji/ImageJ.
    """
    if not selected_keys:
        return None
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for key in selected_keys:
            if "/" not in key:
                continue
            field_name, clabel = key.split("/", 1)
            fr = results.get(field_name)
            if not fr or fr.get("skipped"):
                continue
            cd = fr["cells"].get(clabel)
            if not cd or "error" in cd:
                continue
            fd = fields.get(field_name)
            if not fd:
                continue

            # Always use the original unprocessed stack
            stack = fd["stack"]
            H, W = stack.shape[1], stack.shape[2]

            r0b, c0b, r1b, c1b = cd["cell_meta"]["bbox"]
            r0 = max(0, r0b - pad)
            r1 = min(H, r1b + pad)
            c0 = max(0, c0b - pad)
            c1 = min(W, c1b + pad)

            # (T, crop_H, crop_W), original dtype
            crop = stack[:, r0:r1, c0:c1]

            tif_buf = io.BytesIO()
            tifffile.imwrite(
                tif_buf, crop,
                photometric="minisblack",
                metadata={"axes": "TYX",
                          "field": field_name,
                          "cell": clabel,
                          "bbox_r0": r0, "bbox_c0": c0},
            )
            tif_buf.seek(0)
            zf.writestr(
                f"{field_name}/{field_name}_{clabel}_raw.tif", tif_buf.read())

    buf.seek(0)
    data = buf.read()
    return data if len(data) > 0 else None


# ── ZIP export helpers ───────────────────────────────────────────────────────

def _traj_row_a(cell_id: str, t: tuple) -> dict:
    """Create a trajectory row for single-cell mode.

    Args:
        cell_id: Cell identifier string.
        t: Tuple of (frame, x, y, intensity, is_interpolated).

    Returns:
        Dictionary row for CSV export.
    """
    return {"cell_id": cell_id, "frame": t[0],
            "x_px": round(t[1], 4), "y_px": round(t[2], 4),
            "intensity": round(t[3], 2) if not np.isnan(t[3]) else np.nan,
            "interpolated": t[4]}


def _traj_row_b(field_id: str, cell_id: str, t: tuple) -> dict:
    """Create a trajectory row for batch (field) mode.

    Args:
        field_id: Field identifier string.
        cell_id: Cell identifier string.
        t: Tuple of (frame, x_rel, y_rel, intensity, is_interpolated).

    Returns:
        Dictionary row for CSV export.
    """
    return {"field_id": field_id, "cell_id": cell_id, "frame": t[0],
            "x_rel_px": round(t[1], 4), "y_rel_px": round(t[2], 4),
            "intensity": round(t[3], 2) if not np.isnan(t[3]) else np.nan,
            "interpolated": t[4]}


def build_single_cell_zip(cname: str, data: dict, trows: list) -> bytes:
    """ZIP with trajectory PNG, CSV, annotated GIF, and text report for one cell.

    Args:
        cname: Cell name string.
        data: Dictionary with 'tf_b' (PNG bytes), 'gif' (GIF bytes), 'report' (str).
        trows: List of trajectory row dictionaries for CSV.

    Returns:
        bytes: ZIP archive data.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{cname}_trajectory.png", data["tf_b"])
        zf.writestr(f"{cname}_trajectory.csv",
                    pd.DataFrame(trows).to_csv(index=False).encode("utf-8"))
        if data["gif"]:
            zf.writestr(f"{cname}_annotated.gif", data["gif"])
        zf.writestr(f"{cname}_report.txt", data["report"].encode("utf-8"))
    buf.seek(0)
    return buf.read()


def build_all_cells_zip(cells: dict) -> bytes:
    """ZIP with trajectories, GIFs and reports for all cells (Mode A).

    Args:
        cells: Dictionary of cell analysis results keyed by cell name.

    Returns:
        bytes: ZIP archive data.
    """
    buf = io.BytesIO()
    all_csv = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for nm, d in cells.items():
            if "error" in d:
                continue
            if d["tf_b"]:
                zf.writestr(f"{nm}/{nm}_trajectory.png", d["tf_b"])
            if d["gif"]:
                zf.writestr(f"{nm}/{nm}_annotated.gif",  d["gif"])
            zf.writestr(f"{nm}/{nm}_report.txt", d["report"].encode("utf-8"))
            for t in d["result"]["trajectory"]:
                all_csv.append(_traj_row_a(nm, t))
        if all_csv:
            zf.writestr("all_cells.csv",
                        pd.DataFrame(all_csv).to_csv(index=False).encode("utf-8"))
    buf.seek(0)
    return buf.read()


def build_field_zip(fname: str, fr: dict) -> bytes:
    """ZIP with overview, per-cell PNGs/GIFs/reports, and trajectory CSV for one field.

    Args:
        fname: Field name string.
        fr: Field result dictionary with 'ov_bytes' (overview PNG) and 'cells' dict.

    Returns:
        bytes: ZIP archive data.
    """
    buf = io.BytesIO()
    fc = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if fr["ov_bytes"]:
            zf.writestr(f"{fname}_overview.png", fr["ov_bytes"])
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
                fc.append(_traj_row_b(fname, cl2, t))
        if fc:
            zf.writestr(f"{fname}_trajectories.csv",
                        pd.DataFrame(fc).to_csv(index=False).encode("utf-8"))
    buf.seek(0)
    return buf.read()


def build_all_fields_zip(results: dict) -> bytes:
    """ZIP with all fields: overviews, cell PNGs/GIFs/reports, and combined CSV (Mode B).

    Args:
        results: Dictionary of all field results.

    Returns:
        bytes: ZIP archive data.
    """
    buf = io.BytesIO()
    ac = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn2, fr2 in results.items():
            if fr2.get("skipped"):
                continue
            if fr2["ov_bytes"]:
                zf.writestr(f"{fn2}/{fn2}_overview.png", fr2["ov_bytes"])
            for cl2, cd2 in fr2["cells"].items():
                if "error" in cd2:
                    continue
                if cd2["tf_b"]:
                    zf.writestr(f"{fn2}/{fn2}_{cl2}_traj.png", cd2["tf_b"])
                if cd2["gif"]:
                    zf.writestr(f"{fn2}/{fn2}_{cl2}_roi.gif",  cd2["gif"])
                zf.writestr(f"{fn2}/{fn2}_{cl2}_report.txt",
                            cd2["report"].encode("utf-8"))
                for t in cd2["result"]["trajectory"]:
                    ac.append(_traj_row_b(fn2, cl2, t))
        if ac:
            zf.writestr("all_fields_cells.csv",
                        pd.DataFrame(ac).to_csv(index=False).encode("utf-8"))
    buf.seek(0)
    return buf.read()
