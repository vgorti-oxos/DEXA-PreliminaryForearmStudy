from __future__ import annotations

import argparse
import base64
import json
import math
import tempfile
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import tifffile
import tkinter as tk
from tkinter import messagebox


PIXEL_SPACING_MM = 0.1321
CROP_PX = 100
WRIST_SIDE_RIGHT = True
MU_B_L = 2.492
MU_B_H = 1.141
CALIBRATION_FACTOR_50_70 = 2.4329
EPS = 1e-6

ROI_LINES_MM = {
    "styloid": 0.0,
    "ud_start": 10.0,
    "mid_start": 25.0,
    "one_third_start": 74.0,
    "roi_end": 94.0,
}

ROI_DEFAULT_WIDTH_PX = 600.0
ROI_MIN_WIDTH_PX = 350.0
ROI_MAX_WIDTH_PX = 900.0

REGION_BANDS_MM = {
    "UD": (10.0, 25.0),
    "MID": (25.0, 74.0),
    "ONE_THIRD": (74.0, 94.0),
}

# Values transcribed from the reference table PNG files in reference scripts/.
NORM_TABLES: dict[str, dict[str, dict[str, tuple[float, float]]]] = {
    "UD": {
        "F": {
            "25-29": (0.382, 0.049),
            "30-34": (0.381, 0.045),
            "35-39": (0.380, 0.049),
            "40-44": (0.374, 0.053),
            "45-49": (0.374, 0.046),
            "50-54": (0.353, 0.057),
            "55-59": (0.323, 0.057),
            "60-64": (0.296, 0.062),
            "65-69": (0.273, 0.060),
            "70-74": (0.264, 0.058),
            "75-79": (0.249, 0.060),
            "80-84": (0.236, 0.067),
        },
        "M": {
            "25-29": (0.492, 0.052),
            "30-34": (0.504, 0.072),
            "35-39": (0.482, 0.054),
            "40-44": (0.471, 0.053),
            "45-49": (0.461, 0.066),
            "50-54": (0.474, 0.063),
            "55-59": (0.456, 0.064),
            "60-64": (0.444, 0.066),
            "65-69": (0.429, 0.070),
            "70-74": (0.412, 0.074),
            "75-79": (0.395, 0.095),
            "80-84": (0.372, 0.022),
        },
    },
    "ONE_THIRD": {
        "F": {
            "25-29": (0.475, 0.039),
            "30-34": (0.482, 0.039),
            "35-39": (0.479, 0.047),
            "40-44": (0.474, 0.048),
            "45-49": (0.467, 0.047),
            "50-54": (0.460, 0.050),
            "55-59": (0.429, 0.055),
            "60-64": (0.401, 0.061),
            "65-69": (0.373, 0.065),
            "70-74": (0.359, 0.063),
            "75-79": (0.339, 0.070),
            "80-84": (0.309, 0.072),
        },
        "M": {
            "25-29": (0.580, 0.047),
            "30-34": (0.587, 0.049),
            "35-39": (0.583, 0.042),
            "40-44": (0.583, 0.048),
            "45-49": (0.566, 0.051),
            "50-54": (0.571, 0.051),
            "55-59": (0.559, 0.056),
            "60-64": (0.545, 0.061),
            "65-69": (0.529, 0.068),
            "70-74": (0.510, 0.072),
            "75-79": (0.485, 0.092),
            "80-84": (0.464, 0.072),
        },
    },
}

# Counts for the 25-29 peak-reference bin, transcribed from the same tables.
# T-scores use pooled women+men peak BMD so they are not sex-stratified.
NORM_PEAK_COUNTS: dict[str, dict[str, int]] = {
    "UD": {"F": 86, "M": 52},
    "ONE_THIRD": {"F": 86, "M": 52},
}


@dataclass
class PatientInput:
    patient: str
    low_path: Path
    high_path: Path
    gender: str
    age: int


@dataclass
class RoiPlacement:
    styloid_xy: tuple[float, float]
    axis_xy: tuple[float, float]
    unit_down: tuple[float, float]
    unit_right: tuple[float, float]
    width_px: float


def read_tiff(path: Path) -> np.ndarray:
    img = tifffile.imread(path)
    arr = np.asarray(img)
    if arr.ndim == 3:
        if arr.shape[-1] >= 3:
            arr = arr[..., :3].mean(axis=-1)
        else:
            arr = arr.mean(axis=-1)
    return arr.astype(np.float32)


def normalize_u8(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    vals = img[np.isfinite(img)]
    if vals.size == 0:
        return np.zeros(img.shape, dtype=np.uint8)
    lo, hi = np.percentile(vals, [p_low, p_high])
    if hi <= lo:
        lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        return np.zeros(img.shape, dtype=np.uint8)
    out = np.clip((img - lo) / (hi - lo), 0, 1)
    return (out * 255).astype(np.uint8)


def attenuation(bg: np.ndarray, img: np.ndarray) -> np.ndarray:
    bg = np.maximum(bg.astype(np.float32), EPS)
    img = np.maximum(img.astype(np.float32), EPS)
    att = np.log(bg / img)
    att[~np.isfinite(att)] = 0
    return att


def crop_edges(arr: np.ndarray, crop_px: int) -> np.ndarray:
    if crop_px <= 0:
        return arr.copy()
    h, w = arr.shape[:2]
    if h <= 2 * crop_px + 10 or w <= 2 * crop_px + 10:
        return arr.copy()
    return arr[crop_px : h - crop_px, crop_px : w - crop_px].copy()


def largest_components(mask: np.ndarray, keep: int = 1, min_area: int = 100) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return np.zeros(mask.shape, dtype=np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(areas)[::-1]
    out = np.zeros(mask.shape, dtype=np.uint8)
    kept = 0
    for idx in order:
        if areas[idx] < min_area:
            continue
        out[labels == idx + 1] = 255
        kept += 1
        if kept >= keep:
            break
    return out


def make_forearm_mask(att: np.ndarray) -> np.ndarray:
    img = normalize_u8(cv2.GaussianBlur(att, (0, 0), 3), 2, 98)
    _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(otsu) > otsu.size * 0.65:
        otsu = cv2.bitwise_not(otsu)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    mask = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = largest_components(mask, keep=1, min_area=5000)
    return mask


def rotate_image(arr: np.ndarray, angle_deg: float, fill_value: float = 0) -> np.ndarray:
    h, w = arr.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        arr,
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(fill_value),
    )


def rotate_mask(mask: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = mask.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    out = cv2.warpAffine(mask, mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return (out > 0).astype(np.uint8) * 255


def orientation_angle(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if len(xs) < 100:
        return 90.0
    pts = np.column_stack([xs, ys]).astype(np.float32)
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    vx, vy = vecs[:, np.argmax(vals)]
    if vx < 0:
        vx, vy = -vx, -vy
    angle_from_x = math.degrees(math.atan2(vy, vx))
    return 90.0 - angle_from_x


def standardize_orientation(
    low: np.ndarray, high: np.ndarray, bg_low: np.ndarray, bg_high: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if WRIST_SIDE_RIGHT:
        low_r = cv2.rotate(low, cv2.ROTATE_90_COUNTERCLOCKWISE)
        high_r = cv2.rotate(high, cv2.ROTATE_90_COUNTERCLOCKWISE)
        bg_low_r = cv2.rotate(bg_low, cv2.ROTATE_90_COUNTERCLOCKWISE)
        bg_high_r = cv2.rotate(bg_high, cv2.ROTATE_90_COUNTERCLOCKWISE)
        orientation = "rotate_90_counterclockwise"
    else:
        low_r = cv2.rotate(low, cv2.ROTATE_90_CLOCKWISE)
        high_r = cv2.rotate(high, cv2.ROTATE_90_CLOCKWISE)
        bg_low_r = cv2.rotate(bg_low, cv2.ROTATE_90_CLOCKWISE)
        bg_high_r = cv2.rotate(bg_high, cv2.ROTATE_90_CLOCKWISE)
        orientation = "rotate_90_clockwise"

    meta = {
        "orientation": orientation,
        "rotation_degrees": 90.0 if WRIST_SIDE_RIGHT else -90.0,
        "flipped_180": False,
        "note": "Fixed acquisition-orientation rotation only; no contour/PCA straightening applied.",
    }
    return low_r, high_r, bg_low_r, bg_high_r, meta


def register_high_to_low(low_att: np.ndarray, high_att: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    fixed = normalize_u8(cv2.GaussianBlur(low_att, (0, 0), 2), 2, 98).astype(np.float32) / 255.0
    moving = normalize_u8(cv2.GaussianBlur(high_att, (0, 0), 2), 2, 98).astype(np.float32) / 255.0
    fixed_edges = cv2.Canny((fixed * 255).astype(np.uint8), 30, 90).astype(np.float32) / 255.0
    moving_edges = cv2.Canny((moving * 255).astype(np.uint8), 30, 90).astype(np.float32) / 255.0
    fixed_mix = 0.65 * fixed + 0.35 * fixed_edges
    moving_mix = 0.65 * moving + 0.35 * moving_edges

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-6)
    cc = float("nan")
    try:
        cc, warp = cv2.findTransformECC(
            fixed_mix,
            moving_mix,
            warp,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            None,
            5,
        )
    except cv2.error:
        try:
            cc, warp = cv2.findTransformECC(
                fixed_mix,
                moving_mix,
                warp,
                cv2.MOTION_AFFINE,
                criteria,
                None,
                5,
            )
        except cv2.error:
            warp = np.eye(2, 3, dtype=np.float32)
            cc = float("nan")

    h, w = low_att.shape
    return warp, warp, cc


def apply_warp(arr: np.ndarray, warp: np.ndarray, fill_value: float | None = None) -> np.ndarray:
    h, w = arr.shape[:2]
    if fill_value is None:
        fill_value = float(np.median(arr))
    return cv2.warpAffine(
        arr,
        warp,
        (w, h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(fill_value),
    )


def segment_bones(low_att: np.ndarray, high_att: np.ndarray, forearm_mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    combined = 0.7 * low_att + 0.3 * high_att
    combined = cv2.GaussianBlur(combined, (0, 0), 1.2)
    inside = combined[forearm_mask > 0]
    meta: dict[str, Any] = {}
    if inside.size < 100:
        return np.zeros(low_att.shape, dtype=np.uint8), {"error": "forearm mask too small"}

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalize_u8(combined, 1, 99))
    enhanced[forearm_mask == 0] = 0

    masked_vals = enhanced[forearm_mask > 0]
    q80 = float(np.percentile(masked_vals, 80))
    q88 = float(np.percentile(masked_vals, 88))
    q92 = float(np.percentile(masked_vals, 92))
    otsu_val, _ = cv2.threshold(masked_vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(float(otsu_val), q80)
    threshold = min(threshold, q92)
    meta.update({"q80": q80, "q88": q88, "q92": q92, "otsu": float(otsu_val), "threshold": threshold})

    candidate = np.zeros_like(enhanced, dtype=np.uint8)
    candidate[(enhanced >= threshold) & (forearm_mask > 0)] = 255
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 21)), iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    h, w = candidate.shape
    components: list[tuple[float, int]] = []
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        cw = int(stats[lab, cv2.CC_STAT_WIDTH])
        ch = int(stats[lab, cv2.CC_STAT_HEIGHT])
        if area < 400:
            continue
        elongation = ch / max(cw, 1)
        distal_penalty = 0.65 if y < h * 0.18 and ch < h * 0.28 else 1.0
        score = area * max(elongation, 0.5) * distal_penalty
        components.append((score, lab))

    components.sort(reverse=True)
    bone = np.zeros_like(candidate)
    for _, lab in components[:4]:
        bone[labels == lab] = 255

    # Keep long central components after a light reconstruction; this preserves shafts but removes speckles.
    bone = cv2.morphologyEx(bone, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 17)), iterations=1)
    bone = cv2.morphologyEx(bone, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 5)), iterations=1)
    bone = largest_components(bone, keep=4, min_area=500)
    meta["component_count_kept"] = min(len(components), 4)
    meta["bone_area_px"] = int(np.count_nonzero(bone))
    return bone, meta


def normalize_float_inside(img: np.ndarray, mask: np.ndarray, p_low: float = 5.0, p_high: float = 98.0) -> np.ndarray:
    vals = img[mask > 0]
    if vals.size < 10:
        return np.zeros(img.shape, dtype=np.float32)
    lo, hi = np.percentile(vals, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((img.astype(np.float32) - float(lo)) / float(hi - lo), 0.0, 1.0)
    out[mask == 0] = 0
    return out.astype(np.float32)


def fill_selected_components(mask: np.ndarray, roi_mask: np.ndarray, keep: int = 5) -> tuple[np.ndarray, list[dict[str, float]]]:
    mask = (mask > 0).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    components: list[tuple[float, int, dict[str, float]]] = []
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        cw = int(stats[lab, cv2.CC_STAT_WIDTH])
        ch = int(stats[lab, cv2.CC_STAT_HEIGHT])
        if area < 700 or ch < 60:
            continue
        elongation = ch / max(cw, 1)
        width_penalty = 1.0 if 15 <= cw <= 280 else 0.45
        score = area * (1.0 + min(elongation, 8.0)) * width_penalty
        meta = {"area": area, "x": x, "y": y, "width": cw, "height": ch, "elongation": float(elongation)}
        components.append((score, lab, meta))

    components.sort(reverse=True, key=lambda item: item[0])
    out = np.zeros_like(mask)
    kept_meta: list[dict[str, float]] = []
    for _score, lab, meta in components[:keep]:
        comp = (labels == lab).astype(np.uint8) * 255
        comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 45)), iterations=1)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, 255, -1)
        kept_meta.append(meta)
    out = cv2.bitwise_and(out, roi_mask)
    return out, kept_meta


def roi_canonical_maps(roi: RoiPlacement) -> tuple[np.ndarray, np.ndarray]:
    width = max(int(round(roi.width_px)), 1)
    height = max(int(round(ROI_LINES_MM["roi_end"] / PIXEL_SPACING_MM)), 1)
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    sty = np.asarray(roi.styloid_xy, dtype=np.float32)
    down = np.asarray(roi.unit_down, dtype=np.float32)
    right = np.asarray(roi.unit_right, dtype=np.float32)
    xmap = sty[0] + right[0] * (u - width / 2.0) + down[0] * v
    ymap = sty[1] + right[1] * (u - width / 2.0) + down[1] * v
    return xmap.astype(np.float32), ymap.astype(np.float32)


def remap_to_roi(arr: np.ndarray, roi: RoiPlacement, interpolation: int, border_value: float = 0) -> np.ndarray:
    xmap, ymap = roi_canonical_maps(roi)
    return cv2.remap(
        arr,
        xmap,
        ymap,
        interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(border_value),
    )


def canonical_mask_to_image(mask_c: np.ndarray, shape: tuple[int, int], roi: RoiPlacement) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    pts = np.stack([xx, yy], axis=-1).astype(np.float32)
    sty = np.asarray(roi.styloid_xy, dtype=np.float32)
    down = np.asarray(roi.unit_down, dtype=np.float32)
    right = np.asarray(roi.unit_right, dtype=np.float32)
    rel = pts - sty
    u = rel @ right + mask_c.shape[1] / 2.0
    v = rel @ down
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    valid = (ui >= 0) & (ui < mask_c.shape[1]) & (vi >= 0) & (vi < mask_c.shape[0])
    out = np.zeros(shape, dtype=np.uint8)
    out[valid] = mask_c[vi[valid], ui[valid]]
    return out


def rowwise_bone_candidate(score: np.ndarray, search_mask: np.ndarray) -> np.ndarray:
    mask = np.zeros(score.shape, dtype=np.uint8)
    width = score.shape[1]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 1))
    for y in range(score.shape[0]):
        valid = search_mask[y] > 0
        if np.count_nonzero(valid) < max(25, width * 0.08):
            continue
        vals = score[y, valid]
        if vals.size < 20:
            continue
        # Distal rows need a slightly lower threshold because metaphyseal/cancellous bone is less peaky.
        y_mm = y * PIXEL_SPACING_MM
        percentile = 50 if y_mm < 35 else 60
        threshold = max(float(np.percentile(vals, percentile)), 0.24)
        row = ((score[y] >= threshold) & valid).astype(np.uint8)[None, :] * 255
        row = cv2.morphologyEx(row, cv2.MORPH_CLOSE, horizontal_kernel, iterations=1)
        row = cv2.morphologyEx(row, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)), iterations=1)
        mask[y] = row[0]
    return mask


def keep_roi_components(mask: np.ndarray, search_mask: np.ndarray, keep: int = 6) -> tuple[np.ndarray, list[dict[str, float]]]:
    mask = cv2.bitwise_and((mask > 0).astype(np.uint8) * 255, search_mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    components: list[tuple[float, int, dict[str, float]]] = []
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        cw = int(stats[lab, cv2.CC_STAT_WIDTH])
        ch = int(stats[lab, cv2.CC_STAT_HEIGHT])
        y_mm = y * PIXEL_SPACING_MM
        if area < 500:
            continue
        if ch < 45 and y_mm > 28:
            continue
        elongation = ch / max(cw, 1)
        distal_bonus = 1.45 if y_mm < 30 else 1.0
        width_penalty = 1.0 if 12 <= cw <= 360 else 0.6
        score = area * (1.0 + min(elongation, 7.0)) * distal_bonus * width_penalty
        meta = {"area": area, "x": x, "y": y, "width": cw, "height": ch, "elongation": float(elongation)}
        components.append((score, lab, meta))

    components.sort(reverse=True, key=lambda item: item[0])
    out = np.zeros_like(mask)
    kept_meta: list[dict[str, float]] = []
    for _score, lab, meta in components[:keep]:
        comp = (labels == lab).astype(np.uint8) * 255
        comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 25)), iterations=1)
        out = cv2.bitwise_or(out, comp)
        kept_meta.append(meta)
    out = cv2.bitwise_and(out, search_mask)
    return out, kept_meta


def fill_bone_halves_preserving_gap(mask: np.ndarray, search_mask: np.ndarray) -> np.ndarray:
    mask = cv2.bitwise_and((mask > 0).astype(np.uint8) * 255, search_mask)
    if np.count_nonzero(mask) < 100:
        return mask

    y0 = int(round(25.0 / PIXEL_SPACING_MM))
    y1 = min(mask.shape[0], int(round(94.0 / PIXEL_SPACING_MM)))
    if y1 <= y0:
        y0, y1 = 0, mask.shape[0]

    col_sums = mask[y0:y1].sum(axis=0).astype(np.float32)
    if col_sums.size < 20 or np.max(col_sums) <= 0:
        return mask
    smooth = cv2.GaussianBlur(col_sums[None, :], (41, 1), 0)[0]
    left = int(mask.shape[1] * 0.35)
    right = int(mask.shape[1] * 0.65)
    split = mask.shape[1] // 2 if right <= left else left + int(np.argmin(smooth[left:right]))

    filled = np.zeros_like(mask)
    for start, end in ((0, split), (split, mask.shape[1])):
        part = mask[:, start:end]
        if np.count_nonzero(part) < 100:
            continue
        part = cv2.morphologyEx(part, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 29)), iterations=1)
        contours, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        side = np.zeros_like(part)
        for contour in contours:
            if cv2.contourArea(contour) >= 500:
                cv2.drawContours(side, [contour], -1, 255, -1)
        filled[:, start:end] = cv2.bitwise_or(filled[:, start:end], side)

    return cv2.bitwise_and(filled, search_mask)


def adaptive_edge_bone_mask_canonical(low_c: np.ndarray, forearm_c: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    gray = normalize_u8(low_c, 1, 99)
    search_mask = cv2.erode(
        (forearm_c > 0).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    search_pixels = int(np.count_nonzero(search_mask))
    if search_pixels < 500:
        return np.zeros(gray.shape, dtype=np.uint8), {"error": "ROI/forearm intersection too small", "search_pixels": search_pixels}

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    equalized = cv2.equalizeHist(clahe)
    blur = cv2.GaussianBlur(equalized, (5, 5), 0)

    candidates: list[tuple[float, np.ndarray, dict[str, Any]]] = []
    for block_size, c_value in ((81, -2), (101, -2), (121, 0), (151, 0)):
        adaptive = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            c_value,
        )
        edges = cv2.Canny(blur, 18, 60)
        edge_closed = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 15)), iterations=1)
        seed = cv2.bitwise_or(adaptive, edge_closed)
        seed = cv2.bitwise_and(seed, search_mask)
        seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 23)), iterations=1)

        filled = fill_bone_halves_preserving_gap(seed, search_mask)
        filled = cv2.dilate(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 9)), iterations=1)
        filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 17)), iterations=1)
        filled = cv2.bitwise_and(filled, search_mask)

        area_fraction = np.count_nonzero(filled) / max(search_pixels, 1)
        distal_zone = np.zeros_like(filled)
        distal_zone[: int(round(35.0 / PIXEL_SPACING_MM)), :] = 255
        distal_pixels = max(int(np.count_nonzero(cv2.bitwise_and(distal_zone, search_mask))), 1)
        distal_fraction = np.count_nonzero(cv2.bitwise_and(distal_zone, filled)) / distal_pixels
        cost = abs(area_fraction - 0.42)
        if area_fraction < 0.24:
            cost += 0.35
        if area_fraction > 0.62:
            cost += 0.45
        if distal_fraction < 0.28:
            cost += 0.25
        meta = {
            "method": "adaptive_edge_halves",
            "block_size": block_size,
            "c_value": c_value,
            "search_pixels": search_pixels,
            "area_fraction": float(area_fraction),
            "distal_area_fraction": float(distal_fraction),
        }
        candidates.append((cost, filled, meta))

    candidates.sort(key=lambda item: item[0])
    best_mask = candidates[0][1]
    best_meta = candidates[0][2]
    best_meta["bone_area_px_canonical"] = int(np.count_nonzero(best_mask))
    return best_mask, best_meta


def local_maxima_1d(values: np.ndarray, keep: int = 50, min_distance: int = 8) -> list[int]:
    values = np.asarray(values, dtype=np.float32)
    peaks: list[tuple[float, int]] = []
    for idx in range(2, len(values) - 2):
        if values[idx] <= 0:
            continue
        if values[idx] >= values[idx - 1] and values[idx] >= values[idx + 1]:
            peaks.append((float(values[idx]), idx))
    if not peaks:
        peaks = [(float(v), int(i)) for i, v in enumerate(values) if v > 0]
    peaks.sort(reverse=True, key=lambda item: item[0])

    selected: list[int] = []
    for _value, idx in peaks:
        if all(abs(idx - prev) >= min_distance for prev in selected):
            selected.append(idx)
        if len(selected) >= keep:
            break
    return selected


def smooth_integer_path(path: np.ndarray, width: int, kernel: int = 21) -> np.ndarray:
    path_f = np.asarray(path, dtype=np.float32)
    if path_f.size < 3:
        return np.clip(np.rint(path_f), 0, width - 1).astype(np.int32)
    kernel = min(kernel, path_f.size if path_f.size % 2 == 1 else path_f.size - 1)
    kernel = max(kernel, 3)
    if kernel % 2 == 0:
        kernel -= 1
    pad = kernel // 2
    padded = np.pad(path_f, (pad, pad), mode="edge")
    smoothed = np.convolve(padded, np.ones(kernel, dtype=np.float32) / kernel, mode="valid")
    return np.clip(np.rint(smoothed), 0, width - 1).astype(np.int32)


def shifted_support(score: np.ndarray, shift: int) -> np.ndarray:
    out = np.zeros_like(score, dtype=np.float32)
    if shift > 0:
        out[:, shift:] = score[:, :-shift]
    elif shift < 0:
        out[:, :shift] = score[:, -shift:]
    else:
        out = score.copy()
    return out


def one_sided_horizontal_mean(score: np.ndarray, side: str, inner_gap: int = 4, width: int = 24) -> np.ndarray:
    h, w = score.shape
    xs = np.arange(w)
    if side == "right":
        left = np.clip(xs + inner_gap, 0, w - 1)
        right = np.clip(xs + inner_gap + width - 1, 0, w - 1)
    elif side == "left":
        left = np.clip(xs - inner_gap - width + 1, 0, w - 1)
        right = np.clip(xs - inner_gap, 0, w - 1)
    else:
        raise ValueError(f"unknown side: {side}")

    right = np.maximum(right, left)
    prefix = np.cumsum(score.astype(np.float32), axis=1)
    totals = prefix[:, right]
    before = left > 0
    if np.any(before):
        totals[:, before] -= prefix[:, left[before] - 1]
    counts = np.maximum(right - left + 1, 1).astype(np.float32)
    return totals / counts[None, :]


def suppress_outer_forearm_air_edges(edge: np.ndarray, bone_score: np.ndarray, side: str) -> tuple[np.ndarray, dict[str, Any]]:
    outside = one_sided_horizontal_mean(bone_score, side=side, inner_gap=5, width=28)
    air_like = np.clip((0.30 - outside) / 0.24, 0.0, 1.0)
    penalty = 1.0 - 0.76 * air_like
    suppressed = edge.astype(np.float32) * penalty.astype(np.float32)
    meta = {
        "side": side,
        "outside_mean_p10": float(np.percentile(outside, 10)),
        "outside_mean_p50": float(np.percentile(outside, 50)),
        "outside_mean_p90": float(np.percentile(outside, 90)),
        "mean_penalty": float(np.mean(penalty)),
    }
    return suppressed, meta


def dynamic_edge_path(
    edge: np.ndarray,
    valid_mask: np.ndarray,
    y0: int,
    y1: int,
    anchor: int,
    window: int,
    max_step: int = 10,
    smooth_weight: float = 0.018,
) -> np.ndarray:
    h, w = edge.shape
    y0 = int(np.clip(y0, 0, h - 1))
    y1 = int(np.clip(y1, y0, h - 1))
    anchor = int(np.clip(anchor, 0, w - 1))
    cols = np.arange(max(0, anchor - window), min(w - 1, anchor + window) + 1, dtype=np.int32)
    if cols.size == 0:
        return np.full(y1 - y0 + 1, anchor, dtype=np.int32)

    steps = y1 - y0 + 1
    costs = np.full((steps, cols.size), np.inf, dtype=np.float32)
    prev = np.full((steps, cols.size), -1, dtype=np.int32)
    col_anchor_delta = (cols.astype(np.float32) - float(anchor)) ** 2

    for step, y in enumerate(range(y0, y1 + 1)):
        y_mm = y * PIXEL_SPACING_MM
        dev_weight = 0.000018 if y_mm < 28.0 else 0.000055
        row_valid = valid_mask[y, cols] > 0
        row_cost = -edge[y, cols].astype(np.float32) + dev_weight * col_anchor_delta
        row_cost += np.where(row_valid, 0.0, 3.0).astype(np.float32)
        if step == 0:
            costs[step] = row_cost
            continue
        for j, col in enumerate(cols):
            lo = max(0, j - max_step)
            hi = min(cols.size, j + max_step + 1)
            transition = costs[step - 1, lo:hi] + smooth_weight * np.abs(cols[lo:hi] - col)
            best = int(np.argmin(transition))
            costs[step, j] = row_cost[j] + transition[best]
            prev[step, j] = lo + best

    j = int(np.argmin(costs[-1]))
    path: list[int] = []
    for step in range(steps - 1, -1, -1):
        path.append(int(cols[j]))
        if step > 0:
            j = int(prev[step, j])
            if j < 0:
                j = int(np.argmin(costs[step - 1]))
    return np.asarray(path[::-1], dtype=np.int32)


def refine_boundary_pair_from_likelihood(
    left_path: np.ndarray,
    right_path: np.ndarray,
    left_edge: np.ndarray,
    right_edge: np.ndarray,
    bone_score: np.ndarray,
    y0: int,
    y1: int,
    base_width: float,
    min_width: int,
    max_width: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    h, w = bone_score.shape
    score_prefix = np.cumsum(bone_score.astype(np.float64), axis=1)
    weak_prefix = np.cumsum((bone_score < 0.22).astype(np.float64), axis=1)
    opt_left = np.asarray(left_path, dtype=np.int32).copy()
    opt_right = np.asarray(right_path, dtype=np.int32).copy()
    accepted = 0
    total_score = 0.0

    def interval_sum(prefix: np.ndarray, y: int, left: int, right: int) -> float:
        if right < left:
            return 0.0
        total = float(prefix[y, right])
        if left > 0:
            total -= float(prefix[y, left - 1])
        return total

    for row_idx, y in enumerate(range(y0, y1 + 1)):
        left0 = int(left_path[row_idx])
        right0 = int(right_path[row_idx])
        if left0 > right0:
            left0, right0 = right0, left0
        y_mm = y * PIXEL_SPACING_MM
        distal = y_mm < 28.0
        tube = max(18, int(w * 0.045))
        if distal:
            tube = max(38, int(w * 0.095))
        if y_mm < 15.0:
            tube += max(8, int(w * 0.025))

        row_max_width = int(min(max_width, max(base_width * (2.35 if distal else 1.65), base_width + (105 if distal else 48))))
        row_min_width = int(max(min_width, base_width * (0.40 if distal else 0.58)))
        left_min = max(0, left0 - tube)
        left_max = min(w - row_min_width - 1, left0 + tube)
        right_min = max(row_min_width, right0 - tube)
        right_max = min(w - 1, right0 + tube)
        if left_max < left_min or right_max < right_min:
            continue

        step = 2 if tube > 26 else 1
        best: tuple[float, int, int] | None = None
        for left in range(left_min, left_max + 1, step):
            min_r = max(right_min, left + row_min_width)
            max_r = min(right_max, left + row_max_width)
            if max_r < min_r:
                continue
            for right in range(min_r, max_r + 1, step):
                width_px = right - left + 1
                interior_mean = interval_sum(score_prefix, y, left, right) / width_px
                weak_fraction = interval_sum(weak_prefix, y, left, right) / width_px
                target_width = base_width * (1.22 if distal else 1.02)
                width_sigma = max(20.0, base_width * (0.80 if distal else 0.34))
                width_penalty = ((width_px - target_width) / width_sigma) ** 2
                path_penalty = (abs(left - left0) + abs(right - right0)) / max(2.0 * tube, 1.0)
                edge_score = float(left_edge[y, left] + right_edge[y, right])
                score = 1.40 * edge_score + 2.20 * interior_mean - 0.95 * weak_fraction - 0.28 * width_penalty - 0.18 * path_penalty
                if best is None or score > best[0]:
                    best = (float(score), left, right)
        if best is None:
            continue
        total_score += best[0]
        accepted += 1
        opt_left[row_idx] = best[1]
        opt_right[row_idx] = best[2]

    opt_left = smooth_integer_path(opt_left, w, kernel=19)
    opt_right = smooth_integer_path(opt_right, w, kernel=19)
    min_allowed = max(min_width, int(base_width * 0.45))
    for idx in range(opt_left.size):
        left = int(opt_left[idx])
        right = int(opt_right[idx])
        if left > right:
            left, right = right, left
        if right - left + 1 < min_allowed:
            center = (left + right) // 2
            left = center - min_allowed // 2
            right = left + min_allowed
        if right - left + 1 > max_width:
            center = (left + right) // 2
            left = center - max_width // 2
            right = left + max_width
        opt_left[idx] = int(np.clip(left, 0, w - 1))
        opt_right[idx] = int(np.clip(right, 0, w - 1))

    meta = {
        "rows_refined": float(accepted),
        "mean_row_score": float(total_score / max(accepted, 1)),
    }
    return opt_left, opt_right, meta


def expand_mask_with_bone_likelihood(
    base_mask: np.ndarray,
    bone_score: np.ndarray,
    search_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    base_mask = (base_mask > 0).astype(np.uint8) * 255
    h, _w = base_mask.shape
    distal_limit = min(h, int(round(35.0 / PIXEL_SPACING_MM)))
    distal_zone = np.zeros_like(base_mask)
    distal_zone[:distal_limit, :] = 255
    distal_tube = cv2.dilate(base_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (39, 11)), iterations=1)
    shaft_tube = cv2.dilate(base_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 7)), iterations=1)
    tube = np.where(distal_zone > 0, distal_tube, shaft_tube).astype(np.uint8)
    tube = cv2.bitwise_and(tube, search_mask)

    candidate = np.zeros_like(base_mask)
    for y in range(h):
        valid = tube[y] > 0
        if np.count_nonzero(valid) < 10:
            continue
        vals = bone_score[y, valid]
        if vals.size < 10:
            continue
        y_mm = y * PIXEL_SPACING_MM
        percentile = 50 if y_mm < 28.0 else 72
        threshold = max(float(np.percentile(vals, percentile)), 0.28 if y_mm < 28.0 else 0.34)
        candidate[y, valid & (bone_score[y] >= threshold)] = 255

    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 7)), iterations=1)
    connected = cv2.bitwise_and(candidate, cv2.dilate(base_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    keep = np.zeros_like(candidate)
    for lab in range(1, n):
        comp = labels == lab
        if stats[lab, cv2.CC_STAT_AREA] < 80:
            continue
        if np.count_nonzero(connected[comp]) == 0:
            continue
        keep[comp] = 255

    expanded = cv2.bitwise_or(base_mask, keep)
    expanded = cv2.morphologyEx(expanded, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11)), iterations=1)
    expanded = cv2.bitwise_and(expanded, search_mask)
    meta = {
        "mode": "edge_path_plus_dual_energy_likelihood_expansion",
        "added_pixels": int(max(0, np.count_nonzero(expanded) - np.count_nonzero(base_mask))),
        "base_pixels": int(np.count_nonzero(base_mask)),
        "expanded_pixels": int(np.count_nonzero(expanded)),
    }
    return expanded, meta


def trim_distal_low_likelihood_tails(
    mask: np.ndarray,
    bone_score: np.ndarray,
    y0: int,
    y1: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    trimmed = (mask > 0).astype(np.uint8) * 255
    h, w = trimmed.shape
    distal_y1 = int(min(y1, round(35.0 / PIXEL_SPACING_MM), h - 1))
    if distal_y1 <= y0:
        return trimmed, {"mode": "distal_low_likelihood_tail_trim", "rows_processed": 0, "pixels_removed": 0}

    removed = 0
    rows_processed = 0
    score_smooth = cv2.GaussianBlur(bone_score.astype(np.float32), (13, 1), 0)
    for y in range(y0, distal_y1 + 1):
        row = trimmed[y] > 0
        if np.count_nonzero(row) < 8:
            continue
        rows_processed += 1
        n, labels, stats, _ = cv2.connectedComponentsWithStats(row.astype(np.uint8)[None, :] * 255, 8)
        new_row = np.zeros(w, dtype=np.uint8)
        for lab in range(1, n):
            start = int(stats[lab, cv2.CC_STAT_LEFT])
            run_w = int(stats[lab, cv2.CC_STAT_WIDTH])
            if run_w < 8:
                continue
            end = start + run_w - 1
            vals = score_smooth[y, start : end + 1]
            if vals.size == 0:
                continue
            peak = float(np.max(vals))
            if peak < 0.22:
                continue
            threshold = max(0.24, peak * 0.36)
            strong = np.flatnonzero(vals >= threshold)
            if strong.size < max(4, run_w * 0.12):
                # A very weak run is more likely soft tissue than bone.
                continue
            keep_start = max(start, start + int(strong[0]) - 5)
            keep_end = min(end, start + int(strong[-1]) + 5)
            if keep_end - keep_start + 1 < max(8, run_w * 0.20):
                center = int(round((keep_start + keep_end) / 2.0))
                half = max(4, int(run_w * 0.10))
                keep_start = max(start, center - half)
                keep_end = min(end, center + half)
            new_row[keep_start : keep_end + 1] = 255
        removed += int(np.count_nonzero(row) - np.count_nonzero(new_row))
        trimmed[y] = new_row

    trimmed = cv2.morphologyEx(trimmed, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7)), iterations=1)
    meta = {
        "mode": "distal_low_likelihood_tail_trim",
        "distal_y0_px": int(y0),
        "distal_y1_px": int(distal_y1),
        "rows_processed": int(rows_processed),
        "pixels_removed": int(max(removed, 0)),
    }
    return trimmed, meta


def bridge_distal_same_bone_fragments(
    mask: np.ndarray,
    search_mask: np.ndarray,
    y0: int,
    y1: int,
    split_x: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    bridged = (mask > 0).astype(np.uint8) * 255
    h, w = bridged.shape
    distal_y1 = int(min(y1, round(35.0 / PIXEL_SPACING_MM), h - 1))
    if distal_y1 <= y0:
        return bridged, {"mode": "distal_same_bone_vertical_bridge", "pixels_added": 0}

    split_x = int(np.clip(split_x, 12, w - 12))
    before = int(np.count_nonzero(bridged))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 37))
    for start, end in ((0, split_x), (split_x, w)):
        if end - start < 24:
            continue
        zone = bridged[y0 : distal_y1 + 1, start:end]
        if np.count_nonzero(zone) < 50:
            continue
        closed = cv2.morphologyEx(zone, cv2.MORPH_CLOSE, kernel, iterations=1)
        closed = cv2.bitwise_and(closed, search_mask[y0 : distal_y1 + 1, start:end])
        bridged[y0 : distal_y1 + 1, start:end] = cv2.bitwise_or(zone, closed)

    meta = {
        "mode": "distal_same_bone_vertical_bridge",
        "distal_y0_px": int(y0),
        "distal_y1_px": int(distal_y1),
        "split_x": int(split_x),
        "pixels_added": int(max(0, np.count_nonzero(bridged) - before)),
    }
    return bridged, meta


def constrain_distal_mask_to_path_corridors(
    mask: np.ndarray,
    paths: dict[str, np.ndarray],
    y0: int,
    y1: int,
    min_gap: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    constrained = (mask > 0).astype(np.uint8) * 255
    h, w = constrained.shape
    distal_y1 = int(min(y1, round(35.0 / PIXEL_SPACING_MM), h - 1))
    if distal_y1 <= y0:
        return constrained, {"mode": "distal_path_corridor_clamp", "pixels_removed": 0, "rows_processed": 0}

    before = int(np.count_nonzero(constrained))
    rows_processed = 0
    for y in range(y0, distal_y1 + 1):
        idx = y - y0
        if idx < 0 or idx >= len(paths["radius_left"]):
            continue
        r_l = int(paths["radius_left"][idx])
        r_r = int(paths["radius_right"][idx])
        u_l = int(paths["ulna_left"][idx])
        u_r = int(paths["ulna_right"][idx])
        if r_l > r_r:
            r_l, r_r = r_r, r_l
        if u_l > u_r:
            u_l, u_r = u_r, u_l
        if r_r + min_gap > u_l:
            mid_gap = int(round((r_r + u_l) / 2.0))
            r_r = mid_gap - min_gap // 2
            u_l = mid_gap + min_gap // 2

        y_mm = y * PIXEL_SPACING_MM
        distal_strength = float(np.clip((35.0 - y_mm) / 27.0, 0.0, 1.0))
        outer_margin = int(round(5.0 + 3.0 * distal_strength))
        inner_margin = int(round(9.0 + 7.0 * distal_strength))
        row_allowed = np.zeros(w, dtype=np.uint8)

        # Left bone: outside is image-left, inside faces the interosseous gap.
        left_start = max(0, r_l - outer_margin)
        left_end = min(w - 1, r_r + inner_margin)
        # Right bone: inside faces the interosseous gap, outside is image-right.
        right_start = max(0, u_l - inner_margin)
        right_end = min(w - 1, u_r + outer_margin)
        if left_end >= left_start:
            row_allowed[left_start : left_end + 1] = 255
        if right_end >= right_start:
            row_allowed[right_start : right_end + 1] = 255
        constrained[y] = np.bitwise_and(constrained[y], row_allowed)
        rows_processed += 1

    constrained = cv2.morphologyEx(constrained, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 5)), iterations=1)
    after = int(np.count_nonzero(constrained))
    meta = {
        "mode": "distal_path_corridor_clamp",
        "distal_y0_px": int(y0),
        "distal_y1_px": int(distal_y1),
        "rows_processed": int(rows_processed),
        "outer_margin_px_minmax": [5, 8],
        "inner_margin_px_minmax": [9, 16],
        "pixels_removed": int(max(0, before - after)),
    }
    return constrained, meta


def refine_distal_paired_boundaries(
    paths: dict[str, np.ndarray],
    left_edge: np.ndarray,
    right_edge: np.ndarray,
    bone_score: np.ndarray,
    forearm_hint: np.ndarray,
    y0: int,
    y1: int,
    left_bone_width: float,
    right_bone_width: float,
    min_row_width: int,
    max_width: int,
    min_gap: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    h, w = bone_score.shape
    distal_y1 = int(min(y1, round(32.0 / PIXEL_SPACING_MM)))
    if distal_y1 <= y0:
        return paths, {"mode": "paired_distal_boundary_refinement", "rows_refined": 0}

    score_prefix = np.cumsum(bone_score.astype(np.float64), axis=1)
    weak_prefix = np.cumsum((bone_score < 0.24).astype(np.float64), axis=1)
    forearm_bounds: list[tuple[int | None, int | None]] = []
    forearm_hint = (forearm_hint > 0).astype(np.uint8)
    for y in range(h):
        xs = np.flatnonzero(forearm_hint[y] > 0)
        if xs.size:
            forearm_bounds.append((int(xs[0]), int(xs[-1])))
        else:
            forearm_bounds.append((None, None))

    def interval_sum(prefix: np.ndarray, y: int, left: int, right: int) -> float:
        left = int(np.clip(left, 0, w - 1))
        right = int(np.clip(right, left, w - 1))
        total = float(prefix[y, right])
        if left > 0:
            total -= float(prefix[y, left - 1])
        return total

    def interval_mean(prefix: np.ndarray, y: int, left: int, right: int) -> float:
        return interval_sum(prefix, y, left, right) / max(right - left + 1, 1)

    def candidate_intervals(
        y: int,
        left0: int,
        right0: int,
        base_width: float,
        side: str,
    ) -> list[tuple[float, int, int, dict[str, float]]]:
        y_mm = y * PIXEL_SPACING_MM
        distal_strength = float(np.clip((32.0 - y_mm) / 24.0, 0.0, 1.0))
        tube = max(34, int(w * (0.090 + 0.100 * distal_strength)))
        min_width = int(max(min_row_width, base_width * (0.50 + 0.10 * (1.0 - distal_strength))))
        max_row_width = int(min(max_width, max(base_width * (1.70 + 1.30 * distal_strength), base_width + 66.0 + 88.0 * distal_strength)))
        target_width = float(np.clip(base_width * (1.05 + 0.95 * distal_strength), min_width, max_row_width))
        width_sigma = max(24.0, base_width * (0.48 + 0.72 * distal_strength))

        left_min = max(0, left0 - tube)
        left_max = min(w - min_width - 1, left0 + tube)
        right_min = max(min_width, right0 - tube)
        right_max = min(w - 1, right0 + tube)
        if left_max < left_min or right_max < right_min:
            return []

        row_forearm_left, row_forearm_right = forearm_bounds[y]
        step = 2
        candidates: list[tuple[float, int, int, dict[str, float]]] = []
        for left in range(left_min, left_max + 1, step):
            min_right = max(right_min, left + min_width)
            max_right = min(right_max, left + max_row_width)
            if max_right < min_right:
                continue
            for right in range(min_right, max_right + 1, step):
                width_px = right - left + 1
                interior = interval_mean(score_prefix, y, left, right)
                weak = interval_mean(weak_prefix, y, left, right)
                edge_pair = float(left_edge[y, left] + right_edge[y, right])
                edge_floor = float(min(left_edge[y, left], right_edge[y, right]))
                width_penalty = ((width_px - target_width) / width_sigma) ** 2
                path_penalty = (abs(left - left0) + abs(right - right0)) / max(2.0 * tube, 1.0)
                soft_penalty = 0.0
                if row_forearm_left is not None and row_forearm_right is not None:
                    if side == "left" and left - row_forearm_left < max(8, int(w * 0.025)):
                        soft_penalty += 0.42 + max(0.0, 0.44 - interior)
                    if side == "right" and row_forearm_right - right < max(8, int(w * 0.025)):
                        soft_penalty += 0.42 + max(0.0, 0.44 - interior)

                # Thin lateral soft-tissue strips often have one strong edge but weak interior.
                edge_imbalance = abs(float(left_edge[y, left]) - float(right_edge[y, right]))
                strip_penalty = max(0.0, edge_imbalance - 0.48) * max(0.0, 0.50 - interior)
                score = (
                    1.35 * edge_pair
                    + 0.45 * edge_floor
                    + 2.45 * interior
                    - 1.25 * weak
                    - 0.32 * width_penalty
                    - 0.24 * path_penalty
                    - soft_penalty
                    - 0.55 * strip_penalty
                )
                candidates.append(
                    (
                        float(score),
                        left,
                        right,
                        {
                            "interior": float(interior),
                            "weak": float(weak),
                            "width": float(width_px),
                            "edge_pair": float(edge_pair),
                            "soft_penalty": float(soft_penalty),
                        },
                    )
                )
        candidates.sort(reverse=True, key=lambda item: item[0])
        return candidates[:24]

    refined = {key: value.copy() for key, value in paths.items()}
    rows_changed = 0
    total_joint_score = 0.0
    for y in range(y0, distal_y1 + 1):
        idx = y - y0
        left_l = int(paths["radius_left"][idx])
        left_r = int(paths["radius_right"][idx])
        right_l = int(paths["ulna_left"][idx])
        right_r = int(paths["ulna_right"][idx])
        if left_l > left_r:
            left_l, left_r = left_r, left_l
        if right_l > right_r:
            right_l, right_r = right_r, right_l

        left_candidates = candidate_intervals(y, left_l, left_r, left_bone_width, "left")
        right_candidates = candidate_intervals(y, right_l, right_r, right_bone_width, "right")
        if not left_candidates or not right_candidates:
            continue

        best: tuple[float, tuple[float, int, int, dict[str, float]], tuple[float, int, int, dict[str, float]]] | None = None
        for cand_left in left_candidates:
            for cand_right in right_candidates:
                _score_l, l0, r0, meta_l = cand_left
                _score_r, l1, r1, meta_r = cand_right
                if r0 + min_gap > l1:
                    continue
                gap_mean = interval_mean(score_prefix, y, r0 + 1, l1 - 1) if l1 > r0 + 1 else 1.0
                joint_score = cand_left[0] + cand_right[0] - 0.58 * gap_mean
                if best is None or joint_score > best[0]:
                    best = (float(joint_score), cand_left, cand_right)
        if best is None:
            continue

        _, best_left, best_right = best
        _, l0, r0, meta_l = best_left
        _, l1, r1, meta_r = best_right
        # Require at least a modest bone-like interior before replacing the stable path row.
        if max(meta_l["interior"], meta_r["interior"]) < 0.30 and y * PIXEL_SPACING_MM > 12.0:
            continue
        refined["radius_left"][idx] = l0
        refined["radius_right"][idx] = r0
        refined["ulna_left"][idx] = l1
        refined["ulna_right"][idx] = r1
        rows_changed += 1
        total_joint_score += best[0]

    for key in refined:
        segment = refined[key][: distal_y1 - y0 + 1]
        refined[key][: distal_y1 - y0 + 1] = smooth_integer_path(segment, w, kernel=13)

    meta = {
        "mode": "paired_distal_boundary_refinement",
        "distal_y0_px": int(y0),
        "distal_y1_px": int(distal_y1),
        "rows_refined": int(rows_changed),
        "mean_joint_score": float(total_joint_score / max(rows_changed, 1)),
    }
    return refined, meta


def edge_path_bone_mask_canonical(
    low_c: np.ndarray,
    forearm_c: np.ndarray,
    low_att_c: np.ndarray | None = None,
    high_att_c: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    gray = normalize_u8(low_c, 1, 99)
    h, w = gray.shape
    if h < 80 or w < 80:
        return np.zeros(gray.shape, dtype=np.uint8), {"error": "canonical ROI too small", "height": h, "width": w}

    forearm_hint = (forearm_c > 0).astype(np.uint8) * 255
    forearm_hint_pixels = int(np.count_nonzero(forearm_hint))
    base_mask = np.ones_like(gray, dtype=np.uint8) * 255
    base_mask[:, :6] = 0
    base_mask[:, max(w - 6, 0) :] = 0
    search_mask = cv2.erode(base_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 3)), iterations=1)
    search_pixels = int(np.count_nonzero(search_mask))
    if search_pixels < 500:
        return np.zeros(gray.shape, dtype=np.uint8), {"error": "ROI/forearm intersection too small", "search_pixels": search_pixels}

    background = cv2.GaussianBlur(gray, (0, 0), 35)
    flat = np.clip(128.0 + gray.astype(np.float32) - background.astype(np.float32), 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(flat)
    smooth = cv2.GaussianBlur(clahe, (5, 5), 0)

    grad_gray = cv2.Scharr(smooth, cv2.CV_32F, 1, 0)
    left_gray = normalize_float_inside(np.maximum(-grad_gray, 0), search_mask, 2, 99)
    right_gray = normalize_float_inside(np.maximum(grad_gray, 0), search_mask, 2, 99)

    if low_att_c is not None and high_att_c is not None:
        att_source = 0.70 * low_att_c.astype(np.float32) + 0.30 * high_att_c.astype(np.float32)
    elif low_att_c is not None:
        att_source = low_att_c.astype(np.float32)
    else:
        att_source = 255.0 - smooth.astype(np.float32)
    att_u8 = normalize_u8(att_source, 1, 99)
    att_smooth = cv2.GaussianBlur(att_u8, (5, 5), 0)
    grad_att = cv2.Scharr(att_smooth, cv2.CV_32F, 1, 0)
    # Raw detector image is dark inside bone; attenuation is bright inside bone.
    left_att = normalize_float_inside(np.maximum(grad_att, 0), search_mask, 2, 99)
    right_att = normalize_float_inside(np.maximum(-grad_att, 0), search_mask, 2, 99)

    adaptive = cv2.adaptiveThreshold(
        smooth,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        91,
        -2,
    )
    adaptive = cv2.bitwise_and(adaptive, search_mask)
    adaptive_score = cv2.GaussianBlur((adaptive.astype(np.float32) / 255.0), (15, 15), 0)
    dark_score = normalize_float_inside((255 - smooth).astype(np.float32), search_mask, 2, 98)
    att_score = normalize_float_inside(att_smooth.astype(np.float32), search_mask, 2, 98)
    bone_score = normalize_float_inside(0.48 * dark_score + 0.36 * att_score + 0.16 * adaptive_score, search_mask, 1, 99)

    support = cv2.blur(bone_score, (35, 1))
    left_support = shifted_support(support, -16)
    right_support = shifted_support(support, 16)
    left_edge = normalize_float_inside((0.70 * left_gray + 0.55 * left_att) * (0.62 + 0.70 * left_support), search_mask, 1, 99)
    right_edge = normalize_float_inside((0.70 * right_gray + 0.55 * right_att) * (0.62 + 0.70 * right_support), search_mask, 1, 99)
    left_outer_edge, left_outer_suppression_meta = suppress_outer_forearm_air_edges(left_edge, bone_score, "left")
    right_outer_edge, right_outer_suppression_meta = suppress_outer_forearm_air_edges(right_edge, bone_score, "right")

    shaft_y0 = int(np.clip(round(25.0 / PIXEL_SPACING_MM), 0, h - 1))
    shaft_y1 = int(np.clip(round(91.0 / PIXEL_SPACING_MM), shaft_y0 + 1, h))
    if shaft_y1 - shaft_y0 < 80:
        shaft_y0 = int(h * 0.25)
        shaft_y1 = int(h * 0.92)

    anchor_mask = search_mask
    if forearm_hint_pixels >= max(500, int(h * w * 0.10)):
        forearm_anchor = cv2.erode(forearm_hint, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 5)), iterations=1)
        forearm_anchor[:, :6] = 0
        forearm_anchor[:, max(w - 6, 0) :] = 0
        if np.count_nonzero(forearm_anchor[shaft_y0:shaft_y1]) >= max(500, int((shaft_y1 - shaft_y0) * w * 0.10)):
            anchor_mask = forearm_anchor

    shaft_mask = anchor_mask[shaft_y0:shaft_y1] > 0
    col_counts = np.sum(shaft_mask, axis=0).astype(np.float32)
    col_counts[col_counts < max(10, (shaft_y1 - shaft_y0) * 0.18)] = np.nan
    left_col = np.nansum(left_edge[shaft_y0:shaft_y1] * shaft_mask, axis=0) / col_counts
    right_col = np.nansum(right_edge[shaft_y0:shaft_y1] * shaft_mask, axis=0) / col_counts
    interior_col = np.nansum(bone_score[shaft_y0:shaft_y1] * shaft_mask, axis=0) / col_counts
    left_col = np.nan_to_num(left_col, nan=0.0, posinf=0.0, neginf=0.0)
    right_col = np.nan_to_num(right_col, nan=0.0, posinf=0.0, neginf=0.0)
    interior_col = np.nan_to_num(interior_col, nan=0.0, posinf=0.0, neginf=0.0)
    left_col = cv2.GaussianBlur(left_col[None, :].astype(np.float32), (31, 1), 0)[0]
    right_col = cv2.GaussianBlur(right_col[None, :].astype(np.float32), (31, 1), 0)[0]
    interior_col = cv2.GaussianBlur(interior_col[None, :].astype(np.float32), (41, 1), 0)[0]

    min_peak_dist = max(7, int(w * 0.012))
    left_peaks = local_maxima_1d(left_col, keep=70, min_distance=min_peak_dist)
    right_peaks = local_maxima_1d(right_col, keep=70, min_distance=min_peak_dist)
    min_width = max(42, int(w * 0.060))
    max_width = max(75, int(w * 0.40))
    edge_margin = max(8, int(w * 0.015))
    pair_candidates: list[dict[str, float]] = []
    prefix = np.r_[0.0, np.cumsum(interior_col.astype(np.float64))]
    for left in left_peaks:
        if left < edge_margin or left >= w - edge_margin:
            continue
        for right in right_peaks:
            width_px = right - left
            if width_px < min_width or width_px > max_width:
                continue
            if right >= w - edge_margin:
                continue
            interior = float((prefix[right] - prefix[left]) / max(width_px, 1))
            width_prior = math.exp(-((width_px - w * 0.13) ** 2) / (2.0 * (w * 0.11) ** 2))
            center = (left + right) / 2.0
            edge_penalty = 0.35 * max(0.0, 1.0 - center / max(w * 0.12, 1.0))
            edge_penalty += 0.35 * max(0.0, 1.0 - (w - center) / max(w * 0.12, 1.0))
            score = float(left_col[left] + right_col[right] + 1.10 * interior + 0.20 * width_prior - edge_penalty)
            pair_candidates.append(
                {
                    "left": float(left),
                    "right": float(right),
                    "width": float(width_px),
                    "center": float(center),
                    "score": score,
                    "interior": interior,
                }
            )

    pair_candidates.sort(reverse=True, key=lambda item: item["score"])
    best_pair_set: tuple[dict[str, float], dict[str, float]] | None = None
    best_score = -np.inf
    min_gap = max(16, int(w * 0.025))
    for idx, pair_a in enumerate(pair_candidates[:240]):
        for pair_b in pair_candidates[idx + 1 : 280]:
            p1, p2 = sorted((pair_a, pair_b), key=lambda item: item["center"])
            if p1["right"] + min_gap > p2["left"]:
                continue
            gap_left = int(round(p1["right"]))
            gap_right = int(round(p2["left"]))
            gap_score = 0.0
            if gap_right > gap_left:
                gap_score = float(np.mean(interior_col[gap_left:gap_right]))
            separation_bonus = min((p2["center"] - p1["center"]) / max(w * 0.38, 1.0), 1.0) * 0.20
            score = p1["score"] + p2["score"] + separation_bonus - 0.45 * gap_score
            if score > best_score:
                best_score = score
                best_pair_set = (p1, p2)

    if best_pair_set is None:
        fallback_mask, fallback_meta = row_interval_bone_mask_canonical(low_c, forearm_c, low_att_c, high_att_c)
        fallback_meta["fallback_reason"] = "edge path pair selection failed"
        fallback_meta["attempted_method"] = "edge_path_boundary_fill"
        return fallback_mask, fallback_meta

    p1, p2 = best_pair_set
    anchors = {
        "radius_left": int(round(p1["left"])),
        "radius_right": int(round(p1["right"])),
        "ulna_left": int(round(p2["left"])),
        "ulna_right": int(round(p2["right"])),
    }

    y0 = int(np.clip(round(8.0 / PIXEL_SPACING_MM), 0, h - 1))
    y1 = int(np.clip(round(ROI_LINES_MM["roi_end"] / PIXEL_SPACING_MM) - 1, y0, h - 1))
    path_window = max(95, int(w * 0.26))
    paths = {
        "radius_left": smooth_integer_path(
            dynamic_edge_path(left_outer_edge, search_mask, y0, y1, anchors["radius_left"], path_window), w
        ),
        "radius_right": smooth_integer_path(
            dynamic_edge_path(right_edge, search_mask, y0, y1, anchors["radius_right"], path_window), w
        ),
        "ulna_left": smooth_integer_path(
            dynamic_edge_path(left_edge, search_mask, y0, y1, anchors["ulna_left"], path_window), w
        ),
        "ulna_right": smooth_integer_path(
            dynamic_edge_path(right_outer_edge, search_mask, y0, y1, anchors["ulna_right"], path_window), w
        ),
    }

    min_row_width = max(34, int(w * 0.048))
    radius_max_row_width = int(min(max_width, max(float(p1["width"]) * 2.05, float(p1["width"]) + 85.0)))
    ulna_max_row_width = int(min(max_width, max(float(p2["width"]) * 2.05, float(p2["width"]) + 85.0)))

    mask = np.zeros_like(gray, dtype=np.uint8)

    def constrain_width(left: int, right: int, anchor_left: int, anchor_right: int, max_row_width: int) -> tuple[int, int]:
        if right - left <= max_row_width:
            return left, right
        left_dev = abs(left - anchor_left)
        right_dev = abs(right - anchor_right)
        if right_dev >= left_dev:
            right = left + max_row_width
        else:
            left = right - max_row_width
        return int(np.clip(left, 0, w - 1)), int(np.clip(right, 0, w - 1))

    for row_idx, y in enumerate(range(y0, y1 + 1)):
        r_l = int(paths["radius_left"][row_idx])
        r_r = int(paths["radius_right"][row_idx])
        u_l = int(paths["ulna_left"][row_idx])
        u_r = int(paths["ulna_right"][row_idx])

        if r_l > r_r:
            r_l, r_r = r_r, r_l
        if u_l > u_r:
            u_l, u_r = u_r, u_l
        r_l, r_r = constrain_width(r_l, r_r, anchors["radius_left"], anchors["radius_right"], radius_max_row_width)
        u_l, u_r = constrain_width(u_l, u_r, anchors["ulna_left"], anchors["ulna_right"], ulna_max_row_width)
        if r_r + min_gap > u_l:
            mid_gap = int(round((r_r + u_l) / 2.0))
            r_r = mid_gap - min_gap // 2
            u_l = mid_gap + min_gap // 2
        if r_r - r_l >= min_row_width:
            mask[y, max(r_l, 0) : min(r_r + 1, w)] = 255
        if u_r - u_l >= min_row_width:
            mask[y, max(u_l, 0) : min(u_r + 1, w)] = 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 11)), iterations=1)
    mask = cv2.bitwise_and(mask, base_mask)
    mask, likelihood_refine_meta = expand_mask_with_bone_likelihood(mask, bone_score, search_mask)
    bridge_split_x = int(round((float(p1["right"]) + float(p2["left"])) / 2.0))
    mask, distal_bridge_meta = bridge_distal_same_bone_fragments(mask, search_mask, y0, y1, bridge_split_x)
    mask, distal_trim_meta = trim_distal_low_likelihood_tails(mask, bone_score, y0, y1)
    mask, distal_corridor_meta = constrain_distal_mask_to_path_corridors(mask, paths, y0, y1, min_gap)
    distal_zone = np.zeros_like(mask)
    distal_zone[: int(round(30.0 / PIXEL_SPACING_MM)), :] = 255
    distal_pixels = max(int(np.count_nonzero(cv2.bitwise_and(distal_zone, base_mask))), 1)
    meta = {
        "method": "edge_path_boundary_fill",
        "search_pixels": search_pixels,
        "forearm_hint_pixels": forearm_hint_pixels,
        "shaft_y0_px": shaft_y0,
        "shaft_y1_px": shaft_y1,
        "path_y0_px": y0,
        "path_y1_px": y1,
        "anchors": anchors,
        "pair_scores": [float(p1["score"]), float(p2["score"])],
        "pair_widths_px": [float(p1["width"]), float(p2["width"])],
        "max_row_widths_px": [float(radius_max_row_width), float(ulna_max_row_width)],
        "outer_edge_air_suppression": {
            "radius_left": left_outer_suppression_meta,
            "ulna_right": right_outer_suppression_meta,
        },
        "hybrid_refinement": {
            "mode": "edge_path_likelihood_expansion_with_distal_tail_trim",
            "likelihood_expansion": likelihood_refine_meta,
            "distal_bridge": distal_bridge_meta,
            "distal_tail_trim": distal_trim_meta,
            "distal_path_corridor": distal_corridor_meta,
        },
        "area_fraction": float(np.count_nonzero(mask) / max(search_pixels, 1)),
        "distal_area_fraction": float(np.count_nonzero(cv2.bitwise_and(distal_zone, mask)) / distal_pixels),
        "bone_area_px_canonical": int(np.count_nonzero(mask)),
    }
    return mask, meta


def row_interval_bone_mask_canonical(
    low_c: np.ndarray,
    forearm_c: np.ndarray,
    low_att_c: np.ndarray | None = None,
    high_att_c: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    gray = normalize_u8(low_c, 1, 99)
    search_mask = cv2.erode(
        (forearm_c > 0).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    search_pixels = int(np.count_nonzero(search_mask))
    if search_pixels < 500:
        return np.zeros(gray.shape, dtype=np.uint8), {"error": "ROI/forearm intersection too small", "search_pixels": search_pixels}

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    smooth = cv2.GaussianBlur(clahe, (11, 1), 0)
    dark_score = normalize_float_inside((255 - smooth).astype(np.float32), search_mask, 3, 98)
    if low_att_c is not None and high_att_c is not None:
        att_u8 = normalize_u8(0.70 * low_att_c + 0.30 * high_att_c, 1, 99)
        att_score = normalize_float_inside(att_u8.astype(np.float32), search_mask, 3, 98)
        score_image = 0.58 * dark_score + 0.42 * att_score
    else:
        score_image = dark_score
    score_image = cv2.GaussianBlur(score_image, (9, 1), 0)
    h, w = smooth.shape
    mask = np.zeros_like(gray, dtype=np.uint8)
    split = w // 2
    min_width = max(8, int(w * 0.018))
    max_width = max(45, int(w * 0.40))

    for y in range(h):
        valid = search_mask[y] > 0
        if np.count_nonzero(valid) < w * 0.15:
            continue
        y_mm = y * PIXEL_SPACING_MM
        percentile = 58 if y_mm < 28 else 54

        for start, end in ((0, split + int(w * 0.10)), (split - int(w * 0.10), w)):
            start = max(start, 0)
            end = min(end, w)
            side_valid = valid[start:end]
            if np.count_nonzero(side_valid) < max(20, (end - start) * 0.12):
                continue
            threshold = float(np.percentile(score_image[y, start:end][side_valid], percentile))
            dark_row = np.zeros(w, dtype=np.uint8)
            dark_row[start:end] = ((score_image[y, start:end] >= threshold) & side_valid).astype(np.uint8) * 255
            dark_row = cv2.morphologyEx(
                dark_row[None, :],
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (17, 1)),
            )[0]
            segment = dark_row[start:end][None, :]
            n, labels, stats, _ = cv2.connectedComponentsWithStats(segment, 8)
            runs: list[tuple[float, int, int]] = []
            for lab in range(1, n):
                x = int(stats[lab, cv2.CC_STAT_LEFT])
                run_w = int(stats[lab, cv2.CC_STAT_WIDTH])
                if run_w < min_width or run_w > max_width:
                    continue
                global_x = start + x
                if np.count_nonzero(search_mask[y, global_x : global_x + run_w]) < run_w * 0.8:
                    continue
                mean_score = float(np.mean(score_image[y, global_x : global_x + run_w]))
                center_bonus = 1.0 - min(abs((global_x + run_w / 2) - w / 2) / (w / 2), 1.0) * 0.15
                width_bonus = min(run_w / max(min_width, 1), 6.0)
                score = mean_score * 100.0 * center_bonus + width_bonus
                runs.append((score, global_x, run_w))
            if not runs:
                continue
            runs.sort(reverse=True)
            _, global_x, run_w = runs[0]
            mask[y, global_x : global_x + run_w] = 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 35)), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7)), iterations=1)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 9)), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 19)), iterations=1)
    mask = cv2.bitwise_and(mask, search_mask)

    distal_zone = np.zeros_like(mask)
    distal_zone[: int(round(30.0 / PIXEL_SPACING_MM)), :] = 255
    distal_pixels = max(int(np.count_nonzero(cv2.bitwise_and(distal_zone, search_mask))), 1)
    meta = {
        "method": "row_interval_adaptive",
        "search_pixels": search_pixels,
        "area_fraction": float(np.count_nonzero(mask) / max(search_pixels, 1)),
        "distal_area_fraction": float(np.count_nonzero(cv2.bitwise_and(distal_zone, mask)) / distal_pixels),
        "split_x": split,
        "bone_area_px_canonical": int(np.count_nonzero(mask)),
    }
    return mask, meta


def refine_bone_mask_in_roi(
    low: np.ndarray,
    low_att: np.ndarray,
    high_att: np.ndarray,
    forearm_mask: np.ndarray,
    roi: RoiPlacement,
) -> tuple[np.ndarray, dict[str, Any]]:
    low_c = remap_to_roi(low, roi, cv2.INTER_LINEAR, float(np.median(low)))
    low_att_c = remap_to_roi(low_att, roi, cv2.INTER_LINEAR, 0)
    high_att_c = remap_to_roi(high_att, roi, cv2.INTER_LINEAR, 0)
    forearm_c = remap_to_roi(forearm_mask, roi, cv2.INTER_NEAREST, 0)
    best_mask_c, best_meta = edge_path_bone_mask_canonical(low_c, forearm_c, low_att_c, high_att_c)
    best_mask = canonical_mask_to_image(best_mask_c, low.shape, roi)
    roi_mask = roi_region_mask(low.shape, roi, 0.0, ROI_LINES_MM["roi_end"])
    best_mask = cv2.bitwise_and(best_mask, roi_mask)
    best_meta.update(
        {
            "bone_area_px": int(np.count_nonzero(best_mask)),
            "final_area_fraction": float(
                np.count_nonzero(best_mask) / max(float(best_meta.get("search_pixels", 1)), 1.0)
            ),
            "roi_width_px": float(roi.width_px),
        }
    )
    return best_mask, best_meta


def estimate_roi_width(bone_mask: np.ndarray, styloid: tuple[float, float], down: np.ndarray, right: np.ndarray) -> float:
    ys, xs = np.nonzero(bone_mask)
    if len(xs) < 100:
        return 320.0
    pts = np.column_stack([xs, ys]).astype(np.float32)
    origin = np.asarray(styloid, dtype=np.float32)
    rel = pts - origin
    along_mm = rel @ down * PIXEL_SPACING_MM
    across = rel @ right
    band = (along_mm >= 10) & (along_mm <= 94)
    if np.count_nonzero(band) < 50:
        band = (along_mm >= 0) & (along_mm <= 120)
    if np.count_nonzero(band) < 50:
        return 320.0
    width = float(np.percentile(across[band], 98) - np.percentile(across[band], 2) + 80.0)
    return float(np.clip(width, 260.0, 640.0))


class RoiSelector:
    def __init__(self, image_u8: np.ndarray, bone_mask: np.ndarray, patient: str):
        self.image_u8 = image_u8
        self.bone_mask = bone_mask
        self.patient = patient
        self.result: RoiPlacement | None = None
        self.drag_start: tuple[float, float] | None = None
        self.drag_origin: tuple[float, float] | None = None
        self.root = tk.Tk()
        self.root.title(f"DEXA ROI placement - {patient}")
        self.root.protocol("WM_DELETE_WINDOW", self.cancel)

        h, w = image_u8.shape[:2]
        screen_w = max(self.root.winfo_screenwidth(), 900)
        screen_h = max(self.root.winfo_screenheight(), 700)
        max_w = screen_w - 120
        max_h = screen_h - 190
        self.scale = min(max_w / w, max_h / h, 1.0)
        disp = self.make_display_image()
        if self.scale != 1.0:
            disp = cv2.resize(disp, (int(w * self.scale), int(h * self.scale)), interpolation=cv2.INTER_AREA)
        ok, png = cv2.imencode(".png", disp)
        if not ok:
            raise RuntimeError("Could not encode ROI preview")
        self.photo = tk.PhotoImage(data=base64.b64encode(png).decode("ascii"))
        self.roi_top_center = np.array([w * 0.5, h * 0.10], dtype=np.float32)
        self.roi_angle_deg = 0.0
        self.roi_width_px = min(ROI_DEFAULT_WIDTH_PX, w * 0.75)

        self.instructions = tk.Label(
            self.root,
            text=(
                "Place the red 0 mm top edge at the ulna styloid tip, then align the box down the forearm. "
                "Set the width just wide enough to include both radius and ulna with a small soft-tissue margin; "
                "avoid including the full forearm envelope. Drag to move; mouse wheel/Rotate turns it; Width +/- adjusts coverage."
            ),
            font=("Segoe UI", 10),
            justify=tk.LEFT,
            wraplength=max_w,
        )
        self.instructions.pack(fill=tk.X, padx=8, pady=6)
        buttons = tk.Frame(self.root)
        buttons.pack(fill=tk.X, padx=8, pady=8)
        tk.Button(buttons, text="Accept ROI", command=self.accept, bg="#d8f5d1").pack(side=tk.LEFT, padx=4)
        tk.Button(buttons, text="Rotate -5", command=lambda: self.rotate(-5.0)).pack(side=tk.LEFT, padx=4)
        tk.Button(buttons, text="Rotate -1", command=lambda: self.rotate(-1.0)).pack(side=tk.LEFT, padx=4)
        tk.Button(buttons, text="Rotate +1", command=lambda: self.rotate(1.0)).pack(side=tk.LEFT, padx=4)
        tk.Button(buttons, text="Rotate +5", command=lambda: self.rotate(5.0)).pack(side=tk.LEFT, padx=4)
        tk.Button(buttons, text="Width -25", command=lambda: self.adjust_width(-25.0)).pack(side=tk.LEFT, padx=4)
        tk.Button(buttons, text="Width +25", command=lambda: self.adjust_width(25.0)).pack(side=tk.LEFT, padx=4)
        tk.Button(buttons, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=4)
        tk.Button(buttons, text="Cancel", command=self.cancel).pack(side=tk.RIGHT, padx=4)
        self.canvas = tk.Canvas(self.root, width=self.photo.width(), height=self.photo.height(), cursor="fleur")
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", lambda _event: self.rotate(-1.0))
        self.canvas.bind("<Button-5>", lambda _event: self.rotate(1.0))
        self.redraw()

    def make_display_image(self) -> np.ndarray:
        return cv2.cvtColor(self.image_u8, cv2.COLOR_GRAY2BGR)

    def current_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        theta = math.radians(self.roi_angle_deg)
        down = np.array([math.sin(theta), math.cos(theta)], dtype=np.float32)
        right = np.array([down[1], -down[0]], dtype=np.float32)
        return down, right

    def current_roi(self) -> RoiPlacement:
        down, right = self.current_vectors()
        axis = self.roi_top_center + down * (ROI_LINES_MM["roi_end"] / PIXEL_SPACING_MM)
        return RoiPlacement(
            styloid_xy=(float(self.roi_top_center[0]), float(self.roi_top_center[1])),
            axis_xy=(float(axis[0]), float(axis[1])),
            unit_down=(float(down[0]), float(down[1])),
            unit_right=(float(right[0]), float(right[1])),
            width_px=float(self.roi_width_px),
        )

    def roi_points(self) -> tuple[list[np.ndarray], list[tuple[str, float, np.ndarray, np.ndarray]]]:
        down, right = self.current_vectors()
        sty = self.roi_top_center
        half_w = self.roi_width_px / 2.0
        end_px = ROI_LINES_MM["roi_end"] / PIXEL_SPACING_MM
        corners = [
            sty - right * half_w,
            sty + right * half_w,
            sty + down * end_px + right * half_w,
            sty + down * end_px - right * half_w,
        ]
        lines = []
        for label, mm in ROI_LINES_MM.items():
            p0 = sty + down * (mm / PIXEL_SPACING_MM) - right * half_w
            p1 = sty + down * (mm / PIXEL_SPACING_MM) + right * half_w
            lines.append((label, mm, p0, p1))
        return corners, lines

    def redraw(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        corners, lines = self.roi_points()
        self.draw_poly(corners, "yellow", 2)
        for label, mm, p0, p1 in lines:
            color = "orange" if label != "styloid" else "red"
            self.draw_line(p0, p1, color, 2)
            _, right = self.current_vectors()
            text_p = p1 + right * 8
            self.canvas.create_text(
                text_p[0] * self.scale,
                text_p[1] * self.scale,
                text=f"{int(mm)} mm",
                fill=color,
                anchor=tk.W,
                font=("Segoe UI", 9, "bold"),
            )
        center = self.roi_top_center * self.scale
        r = 5
        self.canvas.create_oval(center[0] - r, center[1] - r, center[0] + r, center[1] + r, fill="red", outline="white")

    def on_press(self, event: tk.Event) -> None:
        self.drag_start = (float(event.x) / self.scale, float(event.y) / self.scale)
        self.drag_origin = (float(self.roi_top_center[0]), float(self.roi_top_center[1]))

    def on_drag(self, event: tk.Event) -> None:
        if self.drag_start is None or self.drag_origin is None:
            return
        x = float(event.x) / self.scale
        y = float(event.y) / self.scale
        dx = x - self.drag_start[0]
        dy = y - self.drag_start[1]
        self.roi_top_center = np.array([self.drag_origin[0] + dx, self.drag_origin[1] + dy], dtype=np.float32)
        self.redraw()

    def on_release(self, _event: tk.Event) -> None:
        self.drag_start = None
        self.drag_origin = None

    def on_mousewheel(self, event: tk.Event) -> None:
        step = -1.0 if event.delta > 0 else 1.0
        self.rotate(step)

    def rotate(self, delta_deg: float) -> None:
        self.roi_angle_deg += delta_deg
        self.redraw()

    def adjust_width(self, delta_px: float) -> None:
        self.roi_width_px = float(np.clip(self.roi_width_px + delta_px, ROI_MIN_WIDTH_PX, ROI_MAX_WIDTH_PX))
        self.redraw()

    def draw_poly(self, pts: list[np.ndarray], color: str, width: int) -> None:
        flat = []
        for p in pts:
            flat.extend([float(p[0]) * self.scale, float(p[1]) * self.scale])
        self.canvas.create_polygon(*flat, outline=color, fill="", width=width)

    def draw_line(self, p0: np.ndarray, p1: np.ndarray, color: str, width: int) -> None:
        self.canvas.create_line(
            p0[0] * self.scale,
            p0[1] * self.scale,
            p1[0] * self.scale,
            p1[1] * self.scale,
            fill=color,
            width=width,
        )

    def reset(self) -> None:
        h, w = self.image_u8.shape[:2]
        self.roi_top_center = np.array([w * 0.5, h * 0.10], dtype=np.float32)
        self.roi_angle_deg = 0.0
        self.roi_width_px = min(ROI_DEFAULT_WIDTH_PX, w * 0.75)
        self.redraw()

    def accept(self) -> None:
        self.result = self.current_roi()
        self.root.quit()
        self.root.destroy()

    def cancel(self) -> None:
        self.result = None
        self.root.quit()
        self.root.destroy()

    def run(self) -> RoiPlacement | None:
        self.root.mainloop()
        return self.result


def roi_coordinate_fields(shape: tuple[int, int], roi: RoiPlacement) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    pts = np.stack([xx, yy], axis=-1).astype(np.float32)
    origin = np.asarray(roi.styloid_xy, dtype=np.float32)
    down = np.asarray(roi.unit_down, dtype=np.float32)
    right = np.asarray(roi.unit_right, dtype=np.float32)
    rel = pts - origin
    along_mm = (rel @ down) * PIXEL_SPACING_MM
    across_px = rel @ right
    return along_mm.astype(np.float32), across_px.astype(np.float32)


def roi_region_mask(shape: tuple[int, int], roi: RoiPlacement, start_mm: float, end_mm: float) -> np.ndarray:
    along_mm, across_px = roi_coordinate_fields(shape, roi)
    mask = (along_mm >= start_mm) & (along_mm < end_mm) & (np.abs(across_px) <= roi.width_px / 2.0)
    return mask.astype(np.uint8) * 255


def compute_k(low: np.ndarray, high: np.ndarray, bg_low: np.ndarray, bg_high: np.ndarray, soft_mask: np.ndarray) -> float:
    if np.count_nonzero(soft_mask) < 100:
        return 1.0
    m = soft_mask > 0
    vals = [
        float(np.mean(low[m])),
        float(np.mean(high[m])),
        float(np.mean(bg_low[m])),
        float(np.mean(bg_high[m])),
    ]
    if any(v <= 0 or not np.isfinite(v) for v in vals):
        return 1.0
    low_i, high_i, bg_low_i, bg_high_i = vals
    denom = math.log(bg_high_i / high_i)
    if abs(denom) < EPS:
        return 1.0
    k = math.log(bg_low_i / low_i) / denom
    if not np.isfinite(k) or k <= 0:
        return 1.0
    return float(np.clip(k, 0.2, 5.0))


def compute_bmd_map(
    low: np.ndarray,
    high: np.ndarray,
    bg_low: np.ndarray,
    bg_high: np.ndarray,
    k: float | np.ndarray,
) -> np.ndarray:
    log_l = np.log(np.maximum(bg_low, EPS) / np.maximum(low, EPS))
    log_h = np.log(np.maximum(bg_high, EPS) / np.maximum(high, EPS))
    k_arr = np.asarray(k, dtype=np.float32)
    denom = MU_B_L - k_arr * MU_B_H
    if np.isscalar(denom) or denom.ndim == 0:
        if abs(float(denom)) < EPS:
            denom = np.asarray(EPS, dtype=np.float32)
    else:
        denom = denom.astype(np.float32)
        near_zero = np.abs(denom) < EPS
        denom[near_zero] = EPS
    bmd = (log_l - k * log_h) / denom
    bmd = bmd * CALIBRATION_FACTOR_50_70
    bmd[~np.isfinite(bmd)] = 0
    bmd[bmd < 0] = 0
    return bmd.astype(np.float32)


def compute_local_k_map(
    low: np.ndarray,
    high: np.ndarray,
    bg_low: np.ndarray,
    bg_high: np.ndarray,
    forearm_mask: np.ndarray,
    bone_mask: np.ndarray,
    roi: RoiPlacement,
    global_k: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    shape = low.shape
    along_mm, across_px = roi_coordinate_fields(shape, roi)
    roi_full = ((along_mm >= 0.0) & (along_mm < ROI_LINES_MM["roi_end"]) & (np.abs(across_px) <= roi.width_px / 2.0)).astype(np.uint8) * 255
    bone_dilated = cv2.dilate((bone_mask > 0).astype(np.uint8) * 255, np.ones((11, 11), np.uint8), iterations=1)
    forearm_eroded = cv2.erode((forearm_mask > 0).astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    soft_base = cv2.bitwise_and(roi_full, forearm_eroded)
    soft_base = cv2.bitwise_and(soft_base, cv2.bitwise_not(bone_dilated))

    region_centers: list[float] = []
    region_values: list[float] = []
    meta: dict[str, Any] = {
        "mode": "roi_regional_local_k_interpolated",
        "global_k": float(global_k),
        "regions": {},
    }
    reliability_weights = {"UD": 0.35, "MID": 0.85, "ONE_THIRD": 1.0}
    for region, (start_mm, end_mm) in REGION_BANDS_MM.items():
        region_mask = roi_region_mask(shape, roi, start_mm, end_mm)
        soft_mask = cv2.bitwise_and(soft_base, region_mask)
        soft_pixels = int(np.count_nonzero(soft_mask))
        if soft_pixels >= 1000:
            raw_k = compute_k(low, high, bg_low, bg_high, soft_mask)
            raw_valid = True
        else:
            raw_k = global_k
            raw_valid = False
        reliability = min(1.0, soft_pixels / 12000.0) * reliability_weights.get(region, 0.75)
        blended_k = float((1.0 - reliability) * global_k + reliability * raw_k)
        blended_k = float(np.clip(blended_k, 0.2, 5.0))
        center = (start_mm + end_mm) / 2.0
        region_centers.append(center)
        region_values.append(blended_k)
        meta["regions"][region] = {
            "start_mm": float(start_mm),
            "end_mm": float(end_mm),
            "center_mm": float(center),
            "raw_k": float(raw_k),
            "blended_k": blended_k,
            "soft_pixels": soft_pixels,
            "reliability": float(reliability),
            "raw_valid": raw_valid,
        }

    if not region_centers:
        return np.full(shape, float(global_k), dtype=np.float32), meta

    order = np.argsort(region_centers)
    centers = np.asarray(region_centers, dtype=np.float32)[order]
    values = np.asarray(region_values, dtype=np.float32)[order]
    k_map = np.full(shape, float(global_k), dtype=np.float32)
    in_roi = roi_full > 0
    k_map[in_roi] = np.interp(along_mm[in_roi], centers, values, left=float(values[0]), right=float(values[-1])).astype(np.float32)
    k_map = cv2.GaussianBlur(k_map, (0, 0), 9)
    k_map[~in_roi] = float(global_k)
    meta["k_min"] = float(np.min(k_map[in_roi])) if np.any(in_roi) else float(global_k)
    meta["k_max"] = float(np.max(k_map[in_roi])) if np.any(in_roi) else float(global_k)
    meta["k_mean_roi"] = float(np.mean(k_map[in_roi])) if np.any(in_roi) else float(global_k)
    return k_map.astype(np.float32), meta


def age_bin(age: int) -> str | None:
    for start in range(25, 85, 5):
        end = start + 4
        if start <= age <= end:
            return f"{start}-{end}"
    return None


def score_for(region: str, bmd: float, gender: str, age: int) -> tuple[float | None, float | None, float | None, str | None]:
    bin_name = age_bin(age)
    gender = str(gender).upper()[0]
    if bin_name is None or region not in NORM_TABLES or gender not in NORM_TABLES[region]:
        return None, None, None, bin_name
    ref = NORM_TABLES[region][gender].get(bin_name)
    if ref is None:
        return None, None, None, bin_name
    mean, sd = ref
    if sd <= 0:
        return None, mean, sd, bin_name
    return (bmd - mean) / sd, mean, sd, bin_name


def pooled_peak_reference(region: str) -> tuple[float | None, float | None, int | None]:
    if region not in NORM_TABLES or region not in NORM_PEAK_COUNTS:
        return None, None, None
    groups: list[tuple[int, float, float]] = []
    for gender in ("F", "M"):
        ref = NORM_TABLES[region].get(gender, {}).get("25-29")
        n = NORM_PEAK_COUNTS[region].get(gender)
        if ref is None or n is None:
            continue
        mean, sd = ref
        groups.append((int(n), float(mean), float(sd)))
    if not groups:
        return None, None, None
    total_n = sum(n for n, _mean, _sd in groups)
    if total_n <= 1:
        return None, None, total_n
    pooled_mean = sum(n * mean for n, mean, _sd in groups) / total_n
    ss = 0.0
    for n, mean, sd in groups:
        ss += (n - 1) * (sd**2)
        ss += n * ((mean - pooled_mean) ** 2)
    pooled_sd = math.sqrt(ss / (total_n - 1))
    return pooled_mean, pooled_sd, total_n


def t_score_for(region: str, bmd: float) -> tuple[float | None, float | None, float | None, int | None]:
    mean, sd, n = pooled_peak_reference(region)
    if mean is None or sd is None or sd <= 0:
        return None, mean, sd, n
    return (bmd - mean) / sd, mean, sd, n


def overlay_mask(gray_u8: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.35) -> np.ndarray:
    bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    overlay = bgr.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(bgr, 1 - alpha, overlay, alpha, 0)


def draw_roi_overlay(gray_u8: np.ndarray, roi: RoiPlacement, bone_mask: np.ndarray) -> np.ndarray:
    img = overlay_mask(gray_u8, bone_mask, (40, 220, 255), 0.25)
    sty = np.asarray(roi.styloid_xy, dtype=np.float32)
    down = np.asarray(roi.unit_down, dtype=np.float32)
    right = np.asarray(roi.unit_right, dtype=np.float32)
    half_w = roi.width_px / 2.0
    end_px = ROI_LINES_MM["roi_end"] / PIXEL_SPACING_MM
    corners = np.array(
        [
            sty - right * half_w,
            sty + right * half_w,
            sty + down * end_px + right * half_w,
            sty + down * end_px - right * half_w,
        ],
        dtype=np.int32,
    )
    cv2.polylines(img, [corners], True, (0, 255, 255), 2, cv2.LINE_AA)
    for label, mm in ROI_LINES_MM.items():
        p0 = (sty + down * (mm / PIXEL_SPACING_MM) - right * half_w).astype(int)
        p1 = (sty + down * (mm / PIXEL_SPACING_MM) + right * half_w).astype(int)
        color = (0, 180, 255) if label != "styloid" else (0, 0, 255)
        cv2.line(img, tuple(p0), tuple(p1), color, 2, cv2.LINE_AA)
        cv2.putText(img, f"{int(mm)} mm", tuple(p1 + np.array([6, 0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def save_bmd_heatmap(path: Path, gray_u8: np.ndarray, bmd: np.ndarray, roi_bone_mask: np.ndarray) -> None:
    base = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    vals = bmd[roi_bone_mask > 0]
    if vals.size == 0:
        cv2.imwrite(str(path), base)
        return
    hi = max(float(np.percentile(vals, 98)), EPS)
    heat_u8 = np.clip(bmd / hi, 0, 1)
    heat_u8 = (heat_u8 * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
    out = base.copy()
    out[roi_bone_mask > 0] = cv2.addWeighted(base[roi_bone_mask > 0], 0.35, heat[roi_bone_mask > 0], 0.65, 0)
    cv2.imwrite(str(path), out)


def discover_inputs(data_dir: Path) -> tuple[list[PatientInput], Path, Path]:
    bg_dir = data_dir / "images_bg"
    bg_low = sorted(bg_dir.glob("ri_single_50_*.tif"))
    bg_high = sorted(bg_dir.glob("ri_single_70_*.tif"))
    if len(bg_low) != 1 or len(bg_high) != 1:
        raise RuntimeError("Expected exactly one 50 kV and one 70 kV background TIFF in data/images_bg")

    demographics = pd.read_excel(data_dir / "Patient age and gender.xlsx")
    demographics["Patient Name"] = demographics["Patient Name"].astype(str).str.strip().str.lower()
    demo_by_patient = demographics.set_index("Patient Name").to_dict(orient="index")

    patients: list[PatientInput] = []
    for folder in sorted(data_dir.glob("images_*")):
        if not folder.is_dir() or folder.name == "images_bg":
            continue
        patient = folder.name.removeprefix("images_").lower()
        lows = sorted(folder.glob("ri_single_50_*.tif"))
        highs = sorted(folder.glob("ri_single_70_*.tif"))
        if len(lows) != 1 or len(highs) != 1:
            raise RuntimeError(f"{folder.name}: expected exactly one 50 kV and one 70 kV TIFF")
        if patient not in demo_by_patient:
            raise RuntimeError(f"{patient}: missing demographics row")
        demo = demo_by_patient[patient]
        patients.append(
            PatientInput(
                patient=patient,
                low_path=lows[0],
                high_path=highs[0],
                gender=str(demo["Gender"]).strip().upper(),
                age=int(demo["Age"]),
            )
        )
    return patients, bg_low[0], bg_high[0]


def process_patient(
    patient: PatientInput,
    bg_low_raw: np.ndarray,
    bg_high_raw: np.ndarray,
    output_dir: Path,
    crop_px: int,
    gui: bool = True,
) -> dict[str, Any]:
    patient_dir = output_dir / patient.patient
    patient_dir.mkdir(parents=True, exist_ok=True)

    low_raw = crop_edges(read_tiff(patient.low_path), crop_px)
    high_raw = crop_edges(read_tiff(patient.high_path), crop_px)
    bg_low = crop_edges(bg_low_raw, crop_px)
    bg_high = crop_edges(bg_high_raw, crop_px)

    low, high, bg_low, bg_high, orientation_meta = standardize_orientation(low_raw, high_raw, bg_low, bg_high)
    low_att = attenuation(bg_low, low)
    high_att = attenuation(bg_high, high)

    warp, _, reg_cc = register_high_to_low(low_att, high_att)
    high_reg = apply_warp(high, warp)
    bg_high_reg = apply_warp(bg_high, warp)
    high_att_reg = attenuation(bg_high_reg, high_reg)

    forearm_mask = make_forearm_mask(low_att)
    preliminary_bone_mask, preliminary_bone_meta = segment_bones(low_att, high_att_reg, forearm_mask)
    image_u8 = normalize_u8(low, 1, 99)

    cv2.imwrite(str(patient_dir / "analysis_image.png"), image_u8)
    cv2.imwrite(str(patient_dir / "forearm_mask_overlay.png"), overlay_mask(image_u8, forearm_mask, (0, 220, 80)))
    cv2.imwrite(
        str(patient_dir / "preliminary_bone_mask_overlay.png"),
        overlay_mask(image_u8, preliminary_bone_mask, (40, 220, 255)),
    )
    before = cv2.addWeighted(normalize_u8(low_att), 0.5, normalize_u8(high_att), 0.5, 0)
    after = cv2.addWeighted(normalize_u8(low_att), 0.5, normalize_u8(high_att_reg), 0.5, 0)
    cv2.imwrite(str(patient_dir / "registration_before_blend.png"), before)
    cv2.imwrite(str(patient_dir / "registration_after_blend.png"), after)

    if not gui:
        raise RuntimeError("ROI placement requires GUI mode for this analysis script")

    roi = RoiSelector(image_u8, preliminary_bone_mask, patient.patient).run()
    if roi is None:
        raise RuntimeError(f"{patient.patient}: ROI placement was cancelled")

    bone_mask, roi_bone_meta = refine_bone_mask_in_roi(low, low_att, high_att_reg, forearm_mask, roi)
    cv2.imwrite(str(patient_dir / "bone_mask_overlay.png"), overlay_mask(image_u8, bone_mask, (40, 220, 255)))

    roi_masks: dict[str, np.ndarray] = {}
    roi_bone_mask = np.zeros_like(bone_mask)
    for region, (start, end) in REGION_BANDS_MM.items():
        mask = roi_region_mask(bone_mask.shape, roi, start, end)
        roi_masks[region] = mask
        roi_bone_mask = cv2.bitwise_or(roi_bone_mask, cv2.bitwise_and(mask, bone_mask))

    soft_mask = cv2.bitwise_and(forearm_mask, cv2.bitwise_not(cv2.dilate(bone_mask, np.ones((9, 9), np.uint8), iterations=1)))
    k_global = compute_k(low, high_reg, bg_low, bg_high_reg, soft_mask)
    k_map, k_meta = compute_local_k_map(low, high_reg, bg_low, bg_high_reg, forearm_mask, bone_mask, roi, k_global)
    bmd_map = compute_bmd_map(low, high_reg, bg_low, bg_high_reg, k_map)

    cv2.imwrite(str(patient_dir / "roi_overlay.png"), draw_roi_overlay(image_u8, roi, bone_mask))
    save_bmd_heatmap(patient_dir / "bmd_heatmap.png", image_u8, bmd_map, roi_bone_mask)
    np.save(patient_dir / "bmd_map.npy", bmd_map.astype(np.float32))
    np.save(patient_dir / "k_map.npy", k_map.astype(np.float32))
    cv2.imwrite(str(patient_dir / "roi_bone_mask.png"), roi_bone_mask)
    cv2.imwrite(str(patient_dir / "bone_mask.png"), bone_mask)
    cv2.imwrite(str(patient_dir / "forearm_mask.png"), forearm_mask)

    result: dict[str, Any] = {
        "patient": patient.patient,
        "gender": patient.gender,
        "age": patient.age,
        "age_bin": age_bin(patient.age),
        "low_image": str(patient.low_path),
        "high_image": str(patient.high_path),
        "crop_px": crop_px,
        "pixel_spacing_mm": PIXEL_SPACING_MM,
        "registration_cc": reg_cc,
        "registration_warp": json.dumps(warp.tolist()),
        "soft_tissue_k": k_global,
        "soft_tissue_k_global": k_global,
        "bmd_k_mode": k_meta["mode"],
        "local_k_min": k_meta["k_min"],
        "local_k_max": k_meta["k_max"],
        "local_k_mean_roi": k_meta["k_mean_roi"],
        "bone_area_px": int(np.count_nonzero(bone_mask)),
        "forearm_area_px": int(np.count_nonzero(forearm_mask)),
    }
    for region, region_k_meta in k_meta.get("regions", {}).items():
        result[f"{region}_soft_tissue_k_raw"] = region_k_meta.get("raw_k")
        result[f"{region}_soft_tissue_k"] = region_k_meta.get("blended_k")
        result[f"{region}_soft_tissue_k_reliability"] = region_k_meta.get("reliability")
        result[f"{region}_soft_tissue_pixels"] = region_k_meta.get("soft_pixels")

    for region, mask in roi_masks.items():
        region_bone = (mask > 0) & (bone_mask > 0)
        result[f"{region}_bone_pixels"] = int(np.count_nonzero(region_bone))
        if np.count_nonzero(region_bone) > 0:
            bmd = float(np.mean(bmd_map[region_bone]))
        else:
            bmd = float("nan")
        result[f"{region}_bmd_g_cm2"] = bmd
        if region in ("UD", "ONE_THIRD") and np.isfinite(bmd):
            z_score, z_ref_mean, z_ref_sd, _ = score_for(region, bmd, patient.gender, patient.age)
            t_score, t_ref_mean, t_ref_sd, t_ref_n = t_score_for(region, bmd)
            result[f"{region}_score_age_sex"] = z_score
            result[f"{region}_z_score"] = z_score
            result[f"{region}_z_ref_mean"] = z_ref_mean
            result[f"{region}_z_ref_sd"] = z_ref_sd
            result[f"{region}_t_score"] = t_score
            result[f"{region}_t_ref_mean_pooled_25_29"] = t_ref_mean
            result[f"{region}_t_ref_sd_pooled_25_29"] = t_ref_sd
            result[f"{region}_t_ref_n_pooled_25_29"] = t_ref_n

    sidecar = {
        "patient": {
            "patient": patient.patient,
            "low_path": str(patient.low_path),
            "high_path": str(patient.high_path),
            "gender": patient.gender,
            "age": patient.age,
        },
        "roi": asdict(roi),
        "orientation": orientation_meta,
        "registration_warp": warp.tolist(),
        "registration_cc": reg_cc,
        "preliminary_segmentation": preliminary_bone_meta,
        "roi_refined_segmentation": roi_bone_meta,
        "bmd_k": k_meta,
        "result": result,
    }
    (patient_dir / "analysis_metadata.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return result


def write_outputs(results: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "dexa_results.csv", index=False)
    df.to_excel(output_dir / "dexa_results.xlsx", index=False)
    generate_summary_plots(df, output_dir)


def generate_summary_plots(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    stale_combined_score_plot = plot_dir / "z_and_t_scores_by_region.png"
    if stale_combined_score_plot.exists():
        stale_combined_score_plot.unlink()
    df = df.copy().sort_values("patient")
    patients = df["patient"].astype(str).tolist()
    x = np.arange(len(df))

    def numeric_col(name: str) -> pd.Series:
        if name not in df.columns:
            return pd.Series([np.nan] * len(df), index=df.index)
        return pd.to_numeric(df[name], errors="coerce")

    plt.style.use("seaborn-v0_8-whitegrid")

    bmd_regions = [
        ("UD", "UD_bmd_g_cm2", "#4C78A8"),
        ("MID", "MID_bmd_g_cm2", "#F58518"),
        ("1/3", "ONE_THIRD_bmd_g_cm2", "#54A24B"),
    ]
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.0), 5.2))
    width = 0.25
    for i, (label, col, color) in enumerate(bmd_regions):
        ax.bar(x + (i - 1) * width, numeric_col(col), width=width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(patients, rotation=35, ha="right")
    ax.set_ylabel("BMD (g/cm^2)")
    ax.set_title("Forearm BMD by ROI")
    ax.legend(title="Region")
    fig.tight_layout()
    fig.savefig(plot_dir / "bmd_by_patient_and_region.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.9), 5.2))
    t_vals = numeric_col("ONE_THIRD_t_score")
    colors = np.where(t_vals <= -2.5, "#D62728", np.where(t_vals <= -1.0, "#F58518", "#54A24B"))
    ax.bar(x, t_vals, color=colors)
    ax.axhline(-1.0, color="#F58518", linestyle="--", linewidth=1.5, label="-1.0")
    ax.axhline(-2.5, color="#D62728", linestyle="--", linewidth=1.5, label="-2.5")
    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.set_ylim(-2.5, 2.5)
    ax.set_yticks([-2.5, -2.0, -1.0, 0.0, 1.0, 2.0, 2.5])
    ax.set_xticks(x)
    ax.set_xticklabels(patients, rotation=35, ha="right")
    ax.set_ylabel("T-score")
    ax.set_title("Primary Diagnostic View: 1/3 Forearm T-score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "one_third_t_score_primary.png", dpi=220)
    plt.close(fig)

    def make_score_by_region_plot(score_kind: str, title: str, filename: str) -> None:
        score_regions = [
            ("UD", f"UD_{score_kind}_score", "#4C78A8"),
            ("1/3", f"ONE_THIRD_{score_kind}_score", "#54A24B"),
        ]
        fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.0), 5.2))
        for i, (label, col, color) in enumerate(score_regions):
            ax.bar(x + (i - 0.5) * width, numeric_col(col), width=width, label=label, color=color)
        ax.axhline(0, color="#333333", linewidth=1.0)
        ax.axhline(-1.0, color="#F58518", linestyle="--", linewidth=1.4, label="-1.0")
        ax.axhline(-2.5, color="#D62728", linestyle="--", linewidth=1.4, label="-2.5")
        ax.set_ylim(-2.5, 2.5)
        ax.set_yticks([-2.5, -2.0, -1.0, 0.0, 1.0, 2.0, 2.5])
        ax.set_xticks(x)
        ax.set_xticklabels(patients, rotation=35, ha="right")
        ax.set_ylabel(f"{score_kind.upper()}-score")
        ax.set_title(title)
        ax.legend(title="Region / reference")
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=220)
        plt.close(fig)

    make_score_by_region_plot("z", "Age/Sex-Matched Z-scores by Region", "z_scores_by_region.png")
    make_score_by_region_plot("t", "Pooled Peak T-scores by Region", "t_scores_by_region.png")

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    age = numeric_col("age")
    for label, col, color in bmd_regions:
        ax.scatter(age, numeric_col(col), label=label, s=70, color=color, alpha=0.9)
    for _, row in df.iterrows():
        if pd.notna(row.get("age")):
            ax.text(float(row["age"]) + 0.15, float(row.get("ONE_THIRD_bmd_g_cm2", np.nan)), str(row["patient"]), fontsize=8)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("BMD (g/cm^2)")
    ax.set_title("BMD vs Age")
    ax.legend(title="Region")
    fig.tight_layout()
    fig.savefig(plot_dir / "bmd_vs_age.png", dpi=220)
    plt.close(fig)

    pixel_cols = [
        ("UD", "UD_bone_pixels", "#4C78A8"),
        ("MID", "MID_bone_pixels", "#F58518"),
        ("1/3", "ONE_THIRD_bone_pixels", "#54A24B"),
    ]
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.0), 5.2))
    bottom = np.zeros(len(df))
    for label, col, color in pixel_cols:
        vals = numeric_col(col).fillna(0).to_numpy()
        ax.bar(x, vals, bottom=bottom, label=label, color=color)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(patients, rotation=35, ha="right")
    ax.set_ylabel("Bone mask pixels")
    ax.set_title("QC: Bone Mask Pixels Used per ROI")
    ax.legend(title="Region")
    fig.tight_layout()
    fig.savefig(plot_dir / "qc_bone_pixels_by_region.png", dpi=220)
    plt.close(fig)

    generate_bmd_heatmap_montage(df, output_dir, plot_dir)


def generate_bmd_heatmap_montage(df: pd.DataFrame, output_dir: Path, plot_dir: Path) -> None:
    records: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    all_values: list[np.ndarray] = []
    for _, row in df.sort_values("patient").iterrows():
        patient = str(row.get("patient", ""))
        patient_dir = output_dir / patient
        bmd_path = patient_dir / "bmd_map.npy"
        mask_path = patient_dir / "roi_bone_mask.png"
        image_path = patient_dir / "analysis_image.png"
        if not bmd_path.exists() or not mask_path.exists() or not image_path.exists():
            continue
        bmd = np.load(bmd_path).astype(np.float32)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or gray is None or mask.shape != bmd.shape or gray.shape != bmd.shape:
            continue
        vals = bmd[mask > 0]
        vals = vals[np.isfinite(vals)]
        vals = vals[vals > 0]
        if vals.size == 0:
            continue
        all_values.append(vals)
        records.append((patient, gray, bmd, mask))

    if not records or not all_values:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.concatenate(all_values)
    vmin = max(0.0, float(np.percentile(values, 1)))
    vmax = float(np.percentile(values, 99))
    if vmax <= vmin:
        vmax = vmin + 0.1

    cols = min(4, max(1, len(records)))
    rows = int(math.ceil(len(records) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.6), squeeze=False)
    cmap = plt.get_cmap("turbo")
    mappable = None
    for ax in axes.ravel():
        ax.axis("off")

    for idx, (patient, gray, bmd, mask) in enumerate(records):
        ax = axes[idx // cols][idx % cols]
        ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
        masked_bmd = np.ma.masked_where(mask <= 0, bmd)
        mappable = ax.imshow(masked_bmd, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.76)
        ax.set_title(patient, fontsize=10)
        ax.axis("off")

    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), fraction=0.025, pad=0.015)
        cbar.set_label("BMD (g/cm^2)")
    fig.suptitle("Forearm BMD Heatmaps, Shared Color Scale", fontsize=14)
    fig.savefig(plot_dir / "bmd_heatmap_montage_shared_colorbar.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_analysis(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir) if args.data_dir else root / "data"
    output_dir = Path(args.output_dir) if args.output_dir else root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    patients, bg_low_path, bg_high_path = discover_inputs(data_dir)
    if args.check_inputs:
        print(f"Found {len(patients)} patients.")
        for patient in patients:
            print(f"  {patient.patient}: {patient.gender}, age {patient.age}")
        print(f"Background 50 kV: {bg_low_path}")
        print(f"Background 70 kV: {bg_high_path}")
        return 0

    bg_low_raw = read_tiff(bg_low_path)
    bg_high_raw = read_tiff(bg_high_path)

    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for patient in patients:
        try:
            print(f"Processing {patient.patient}...")
            stale_error = output_dir / f"{patient.patient}_error.txt"
            if stale_error.exists():
                stale_error.unlink()
            results.append(process_patient(patient, bg_low_raw, bg_high_raw, output_dir, args.crop_px, gui=not args.no_gui))
            write_outputs(results, output_dir)
        except Exception as exc:
            tb = traceback.format_exc()
            errors.append(f"{patient.patient}: {exc}")
            (output_dir / f"{patient.patient}_error.txt").write_text(tb, encoding="utf-8")
            print(tb)
            if args.stop_on_error:
                raise

    write_outputs(results, output_dir)
    summary = f"Processed {len(results)} patients. Outputs saved to {output_dir}."
    if errors:
        summary += "\n\nErrors:\n" + "\n".join(errors)
    print(summary)
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("DEXA analysis complete", summary)
        root.destroy()
    except Exception:
        pass
    return 0 if results else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forearm DEXA analysis")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--crop-px", type=int, default=CROP_PX)
    parser.add_argument("--check-inputs", action="store_true", help="Validate folder layout and demographics, then exit.")
    parser.add_argument("--no-gui", action="store_true", help="For validation only; analysis requires GUI ROI placement.")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_analysis(parse_args()))
