"""Texture processing for front and back garment images.

Supports both flat-lay garment photos (transparent PNGs or studio backgrounds)
and photos of human models wearing the garment (automatically isolating the
shirt/sweatshirt with MediaPipe Pose + SelfieSegmentation + GrabCut while
removing head, neck, hands, pants, and background). Packs a 2048x1024 UV atlas.
"""

from __future__ import annotations

import io
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image


ATLAS_WIDTH = 2048
ATLAS_HEIGHT = 1024
HALF_SIZE = 1024

_MP_LOCK = threading.Lock()
_POSE_MODEL = None
_SEG_MODEL = None


def _get_mediapipe_models():
    """Lazy-load MediaPipe Pose and SelfieSegmentation models once per process."""
    global _POSE_MODEL, _SEG_MODEL
    if _POSE_MODEL is None or _SEG_MODEL is None:
        with _MP_LOCK:
            if _POSE_MODEL is None:
                _POSE_MODEL = mp.solutions.pose.Pose(
                    static_image_mode=True,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                )
            if _SEG_MODEL is None:
                _SEG_MODEL = mp.solutions.selfie_segmentation.SelfieSegmentation(
                    model_selection=1
                )
    return _POSE_MODEL, _SEG_MODEL


@dataclass
class ExtractedGarment:
    """Processed garment image with tight crop, torso bounds, and sleeve/length metadata."""

    rgba: np.ndarray  # H x W x 4 uint8
    mask: np.ndarray  # H x W uint8 (0 or 255)
    torso_x_range: Tuple[int, int]  # (x_left, x_right) in cropped image coordinates
    torso_y_range: Tuple[int, int]  # (y_top, y_bottom) of torso in cropped image coordinates
    dominant_color: Tuple[int, int, int]  # RGB
    is_long_sleeve: bool = False
    from_human: bool = False
    torso_aspect: float = 1.55  # height / width of torso (lower = cropped top)


def _load_image_rgba(image_bytes: bytes) -> np.ndarray:
    """Load raw image bytes as an RGBA uint8 numpy array."""
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    return np.array(pil_img, dtype=np.uint8)


def _segment_worn_silhouette_fallback(
    rgb: np.ndarray,
    person_mask: np.ndarray,
) -> Optional[Tuple[np.ndarray, Tuple[int, int], bool]]:
    """Fallback human-garment isolator using SelfieSegmentation width profile + GrabCut.

    Used when MediaPipe Pose misses a back-facing or 3/4-turned model photo.
    """
    h, w = rgb.shape[:2]
    ys_p, xs_p = np.where(person_mask)
    if len(ys_p) < h * w * 0.08:
        return None

    y_top_p = int(ys_p.min())
    row_widths = person_mask.sum(axis=1)

    # Locate widest torso band in middle 25%..75% of image height
    mid_y0 = max(y_top_p + int(h * 0.12), int(h * 0.25))
    mid_y1 = min(h - 1, int(h * 0.80))
    if mid_y1 <= mid_y0 + 10:
        return None

    torso_y_peak = mid_y0 + int(np.argmax(row_widths[mid_y0:mid_y1]))
    max_w = int(row_widths[torso_y_peak])
    if max_w < w * 0.20:
        return None

    # Find neck constriction (narrowest width between head top and torso peak)
    search_y0 = y_top_p + max(4, int((torso_y_peak - y_top_p) * 0.25))
    search_y1 = max(search_y0 + 2, torso_y_peak - max(4, int((torso_y_peak - y_top_p) * 0.15)))
    neck_y = search_y0 + int(np.argmin(row_widths[search_y0:search_y1]))
    neck_w = int(row_widths[neck_y])

    # Collar sits slightly below neck notch
    y_collar_top = min(torso_y_peak - 5, neck_y + max(4, int(max_w * 0.06)))
    if neck_w > max_w * 0.68:
        y_collar_top = y_top_p + int(max_w * 0.36)

    y_pants = min(h, int(y_collar_top + max_w * 1.15))

    torso_row_xs = np.where(person_mask[torso_y_peak] > 0)[0]
    tx_center = 0.5 * (int(torso_row_xs[0]) + int(torso_row_xs[-1]))

    gc = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc[~person_mask] = cv2.GC_BGD
    gc[:y_collar_top, :] = cv2.GC_BGD
    if y_pants < h:
        gc[y_pants:, :] = cv2.GC_BGD

    upper = person_mask.copy()
    upper[:y_collar_top, :] = False
    upper[y_pants:, :] = False
    gc[upper] = cv2.GC_PR_FGD

    # Definite foreground: central chest/back rectangle
    cx0 = max(0, int(tx_center - max_w * 0.22))
    cx1 = min(w, int(tx_center + max_w * 0.22))
    cy0 = min(h - 2, int(y_collar_top + max_w * 0.16))
    cy1 = min(h - 1, int(y_pants - max_w * 0.18))
    if cx1 > cx0 and cy1 > cy0:
        gc[cy0:cy1, cx0:cx1] = cv2.GC_FGD

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, gc, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

    garment_mask = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    garment_mask = _clean_mask(garment_mask)
    if np.sum(garment_mask > 0) < h * w * 0.04:
        return None

    raw_tx0 = max(0, int(tx_center - max_w * 0.30))
    raw_tx1 = min(w - 1, int(tx_center + max_w * 0.30))
    return garment_mask, (raw_tx0, raw_tx1), False


def _try_segment_worn_garment(
    rgb: np.ndarray,
    expect_human: bool = False,
) -> Optional[Tuple[np.ndarray, Tuple[int, int], bool]]:
    """Detect if a human model is wearing the shirt; if so, isolate only the garment.

    Returns (full_image_mask, (raw_torso_x0, raw_torso_x1), is_long_sleeve) or None
    if the image is a flat-lay product photo without a person.
    """
    h, w = rgb.shape[:2]
    pose, seg = _get_mediapipe_models()

    with _MP_LOCK:
        pose_res = pose.process(rgb)
        seg_res = seg.process(rgb)
        person_mask = seg_res.segmentation_mask > 0.45

    if not pose_res.pose_landmarks:
        if expect_human:
            return _segment_worn_silhouette_fallback(rgb, person_mask)
        return None

    lm = pose_res.pose_landmarks.landmark
    # Relax visibility threshold for back-facing / 3/4 poses; estimate hips if occluded
    if min(lm[11].visibility, lm[12].visibility) < 0.30:
        if expect_human:
            return _segment_worn_silhouette_fallback(rgb, person_mask)
        return None

    y_sh_norm = 0.5 * (lm[11].y + lm[12].y)
    sh_w_norm = abs(lm[11].x - lm[12].x)
    if sh_w_norm < 0.10:
        if expect_human:
            return _segment_worn_silhouette_fallback(rgb, person_mask)
        return None

    if min(lm[23].visibility, lm[24].visibility) >= 0.30:
        y_hip_norm = 0.5 * (lm[23].y + lm[24].y)
    else:
        y_hip_norm = min(0.92, y_sh_norm + 1.18 * sh_w_norm)

    torso_h_norm = y_hip_norm - y_sh_norm
    if torso_h_norm < 0.12:
        if expect_human:
            return _segment_worn_silhouette_fallback(rgb, person_mask)
        return None

    y_sh = y_sh_norm * h
    y_hip = y_hip_norm * h
    torso_h = torso_h_norm * h
    x_sh_l = min(lm[11].x, lm[12].x) * w
    x_sh_r = max(lm[11].x, lm[12].x) * w
    sh_w = x_sh_r - x_sh_l

    # Confirm a head/neck region is actually present (unless paired front photo already proved human)
    if not expect_human:
        head_y0 = max(0, int(y_sh - 0.55 * torso_h))
        head_y1 = max(head_y0 + 1, int(y_sh - 0.12 * torso_h))
        head_box = person_mask[head_y0:head_y1, max(0, int(x_sh_l)):min(w, int(x_sh_r))]
        has_face = lm[0].visibility > 0.55 and lm[0].y < y_sh_norm - 0.04
        has_ears = (lm[7].visibility > 0.45 or lm[8].visibility > 0.45) and min(lm[7].y, lm[8].y) < y_sh_norm
        has_head_fg = head_box.size > 0 and float(np.mean(head_box)) > 0.14
        if not (has_face or has_ears or has_head_fg):
            return None

    # Build GrabCut seed mask: eliminate background, head/neck skin, hands, and jeans
    y_collar_top = max(0, int(y_sh - 0.14 * torso_h))
    ear_ys = [lm[idx].y for idx in (7, 8) if lm[idx].visibility > 0.40 and lm[idx].y < y_sh_norm]
    if ear_ys:
        y_collar_top = max(y_collar_top, int(max(ear_ys) * h + 0.03 * torso_h))
    y_collar_top = min(int(y_sh), y_collar_top)

    y_pants = min(h, int(y_hip + 0.14 * torso_h))

    gc = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc[~person_mask] = cv2.GC_BGD
    gc[:y_collar_top, :] = cv2.GC_BGD
    if y_pants < h:
        gc[y_pants:, :] = cv2.GC_BGD

    upper_body = person_mask.copy()
    upper_body[:y_collar_top, :] = False
    upper_body[min(h, int(y_hip + 0.10 * torso_h)):, :] = False
    gc[upper_body] = cv2.GC_PR_FGD

    # Definite foreground: core chest & upper sleeves
    cx0 = max(0, int(x_sh_l + 0.16 * sh_w))
    cx1 = min(w, int(x_sh_r - 0.16 * sh_w))
    cy0 = max(0, int(y_sh + 0.10 * torso_h))
    cy1 = min(h, int(y_hip - 0.12 * torso_h))
    if cx1 > cx0 and cy1 > cy0:
        gc[cy0:cy1, cx0:cx1] = cv2.GC_FGD

    for sh_i, el_i in ((11, 13), (12, 14)):
        if lm[el_i].visibility > 0.40:
            p1 = (int(np.clip(lm[sh_i].x * w, 0, w - 1)), int(np.clip(lm[sh_i].y * h, 0, h - 1)))
            p2 = (int(np.clip(lm[el_i].x * w, 0, w - 1)), int(np.clip(lm[el_i].y * h, 0, h - 1)))
            cv2.line(gc, p1, p2, int(cv2.GC_FGD), max(6, int(w * 0.04)))

    # Remove exposed neck skin (and bare hands near wrists)
    neck_zone = np.zeros((h, w), dtype=bool)
    ny0 = y_collar_top
    ny1 = min(h, int(y_sh + 0.11 * torso_h))
    nx0 = max(0, int(x_sh_l + 0.22 * sh_w))
    nx1 = min(w, int(x_sh_r - 0.22 * sh_w))
    if ny1 > ny0 and nx1 > nx0:
        neck_zone[ny0:ny1, nx0:nx1] = True

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lab_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Suppress pants/jeans pixels near the hem using bottom-center pants color sample
    pants_y0 = min(h - 1, int(y_hip + 0.16 * torso_h))
    if pants_y0 < h - 8 and cx1 > cx0 and cy1 > cy0:
        pants_mask_sample = person_mask[pants_y0:, cx0:cx1]
        if np.sum(pants_mask_sample) > 40:
            pants_lab = np.median(lab_full[pants_y0:, cx0:cx1][pants_mask_sample], axis=0)
            chest_lab = np.median(lab_full[cy0:cy1, cx0:cx1].reshape(-1, 3), axis=0)
            if float(np.linalg.norm(pants_lab - chest_lab)) > 24.0:
                hem_y0 = max(0, int(y_hip - 0.08 * torso_h))
                d_pants = np.linalg.norm(lab_full[hem_y0:y_pants] - pants_lab, axis=2)
                d_chest = np.linalg.norm(lab_full[hem_y0:y_pants] - chest_lab, axis=2)
                is_pants_px = (d_pants < 22.0) & (d_pants + 8.0 < d_chest)
                gc[hem_y0:y_pants][is_pants_px] = cv2.GC_BGD
    if lm[0].visibility > 0.55 and lm[0].y < y_sh_norm:
        fx = int(np.clip(lm[0].x * w, 0, w - 1))
        fy = int(np.clip(lm[0].y * h, 0, h - 1))
        rf = max(5, int(sh_w * 0.06))
        face_crop = bgr[max(0, fy - rf):min(h, fy + rf), max(0, fx - rf):min(w, fx + rf)]
        if face_crop.size > 0:
            face_lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
            face_med = np.median(face_lab.reshape(-1, 3), axis=0)
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            skin_dist = np.linalg.norm(lab - face_med, axis=2)
            gc[neck_zone & (skin_dist < 28.0)] = cv2.GC_BGD

            for wr_i in (15, 16):
                if lm[wr_i].visibility > 0.40:
                    wx = int(np.clip(lm[wr_i].x * w, 0, w - 1))
                    wy = int(np.clip(lm[wr_i].y * h, 0, h - 1))
                    wr_mask = np.zeros((h, w), dtype=bool)
                    rw = max(10, int(sh_w * 0.12))
                    wr_mask[max(0, wy - rw):min(h, wy + rw), max(0, wx - rw):min(w, wx + rw)] = True
                    gc[wr_mask & (skin_dist < 24.0)] = cv2.GC_BGD
    else:
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        skin_ycrcb = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
        gc[neck_zone & skin_ycrcb] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, gc, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

    garment_mask = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    garment_mask = _clean_mask(garment_mask)
    if np.sum(garment_mask > 0) < h * w * 0.04:
        if expect_human:
            return _segment_worn_silhouette_fallback(rgb, person_mask)
        return None

    forearm_covs = []
    for el_i, wr_i in ((13, 15), (14, 16)):
        if lm[el_i].visibility > 0.40 and lm[wr_i].visibility > 0.35:
            hits = 0
            samples = 20
            for t in np.linspace(0.20, 0.85, samples):
                px = int(np.clip((lm[el_i].x * (1.0 - t) + lm[wr_i].x * t) * w, 0, w - 1))
                py = int(np.clip((lm[el_i].y * (1.0 - t) + lm[wr_i].y * t) * h, 0, h - 1))
                if garment_mask[py, px] > 0:
                    hits += 1
            forearm_covs.append(hits / samples)
    is_long_sleeve = bool(forearm_covs and max(forearm_covs) >= 0.45)

    raw_tx0 = max(0, int(x_sh_l - 0.03 * sh_w))
    raw_tx1 = min(w - 1, int(x_sh_r + 0.03 * sh_w))
    return garment_mask, (raw_tx0, raw_tx1), is_long_sleeve


def _segment_garment(rgba: np.ndarray) -> np.ndarray:
    """Return a binary foreground mask (0 or 255) isolating a flat-lay garment."""
    h, w = rgba.shape[:2]
    alpha = rgba[:, :, 3]

    # Case 1: Image already carries meaningful transparency
    transparent_ratio = float(np.mean(alpha < 220))
    if transparent_ratio > 0.03:
        mask = np.where(alpha >= 128, 255, 0).astype(np.uint8)
        return _clean_mask(mask)

    # Case 2: Solid / studio background — sample border pixels
    rgb = rgba[:, :, :3]
    border = np.concatenate(
        [
            rgb[0, :, :],
            rgb[-1, :, :],
            rgb[:, 0, :],
            rgb[:, -1, :],
        ],
        axis=0,
    )
    bg_median = np.median(border, axis=0)
    bg_std = np.std(border, axis=0)
    tolerance = float(np.clip(np.linalg.norm(bg_std) * 2.2 + 22.0, 22.0, 68.0))

    diff = np.linalg.norm(rgb.astype(np.float32) - bg_median.astype(np.float32), axis=2)
    fg_candidate = (diff > tolerance).astype(np.uint8) * 255

    # Flood-fill from image corners to find background connected to borders
    bg_seed_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    fillable = (diff <= tolerance).astype(np.uint8) * 255
    flood = fillable.copy()

    seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    for sx, sy in seeds:
        if flood[sy, sx] == 255:
            cv2.floodFill(flood, bg_seed_mask, (sx, sy), 128)

    mask = np.where(flood == 128, 0, 255).astype(np.uint8)

    fg_ratio = float(np.mean(mask > 0))
    if fg_ratio < 0.08:
        mask = fg_candidate
        fg_ratio = float(np.mean(mask > 0))

    if fg_ratio < 0.05:
        mask = np.full((h, w), 255, dtype=np.uint8)

    return _clean_mask(mask)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Remove small speckles and keep largest foreground component."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if num_labels <= 1:
        return cleaned

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = int(np.argmax(areas)) + 1
    return np.where(labels == largest_idx, 255, 0).astype(np.uint8)


def _inpaint_background(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill non-garment pixels with nearest inner garment pixel to prevent UV bleed."""
    inv_mask = np.where(mask == 0, 255, 0).astype(np.uint8)
    if np.all(inv_mask == 0) or np.all(mask == 0):
        return rgb

    _, labels = cv2.distanceTransformWithLabels(
        inv_mask, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    fg_coords = np.argwhere(mask > 0)
    nearest_idx = np.clip(labels - 1, 0, len(fg_coords) - 1)
    ny = fg_coords[nearest_idx, 0]
    nx = fg_coords[nearest_idx, 1]
    out = rgb[ny, nx]
    fg = mask > 0
    out[fg] = rgb[fg]
    return out


def _estimate_flatlay_torso_and_sleeves(mask: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int], bool, float]:
    """Locate torso X/Y bounds, flat-lay long-sleeve presence, and height/width aspect ratio.

    Scanning contiguous center runs handles cropped tops with long hanging sleeves
    (where sleeves extend well below the cropped waistband).
    """
    h, w = mask.shape
    cx = w // 2
    spans = []
    for r in range(h):
        row = mask[r]
        if row[cx] == 0:
            sub = np.where(row[int(w * 0.40) : int(w * 0.60)] > 0)[0]
            if len(sub) == 0:
                spans.append((r, 0, 0, 0))
                continue
            seed = int(w * 0.40) + int(sub[len(sub) // 2])
        else:
            seed = cx
        lx, rx = seed, seed
        while lx > 0 and row[lx - 1] > 0:
            lx -= 1
        while rx < w - 1 and row[rx + 1] > 0:
            rx += 1
        spans.append((r, lx, rx, rx - lx + 1))

    valid = [s for s in spans if s[3] > w * 0.15]
    if not valid:
        return (0, w - 1), (0, h), False, 1.55

    ty0, ty1 = valid[0][0], valid[-1][0] + 1
    th = max(1, ty1 - ty0)

    # Bottom 35% of torso avoids sleeves attached near upper chest/armpits
    lower = [s for s in valid if s[0] >= ty0 + int(th * 0.65)]
    if not lower:
        lower = valid
    widths = [s[3] for s in lower]
    min_w = float(np.percentile(widths, 8))
    best = [s for s in lower if s[3] <= min_w * 1.10]
    tx0 = int(np.median([s[1] for s in best])) if best else 0
    tx1 = int(np.median([s[2] for s in best])) if best else (w - 1)

    if (tx1 - tx0) < w * 0.30:
        tx0, tx1 = 0, w - 1
    tw = max(1, tx1 - tx0)

    # Measure shoulder top inside torso bounds (excludes tall hoods rising above shoulders)
    sh_ys = []
    for col_x in (min(w - 1, tx0 + 4), max(0, tx1 - 4)):
        col_fg = np.where(mask[:, col_x] > 0)[0]
        if len(col_fg) > 0:
            sh_ys.append(int(col_fg.min()))
    sh_y = float(np.mean(sh_ys)) if sh_ys else float(ty0)
    if sh_y > ty0 + th * 0.35:
        sh_y = float(ty0)

    torso_aspect = float(max(0.55, (ty1 - sh_y) / tw))

    # Detect flat-lay long sleeves from outer wing reach below torso mid-height
    left_wing = mask[:, :tx0]
    right_wing = mask[:, tx1:]
    wing_rows = np.where(
        (left_wing > 0).any(axis=1) if left_wing.size else np.zeros(h, dtype=bool)
        | ((right_wing > 0).any(axis=1) if right_wing.size else np.zeros(h, dtype=bool))
    )[0]
    wing_y_max = int(wing_rows.max()) if len(wing_rows) else ty0
    wing_reach = tx0 + (w - tx1)
    is_long_sleeve = bool(wing_y_max > ty0 + int(th * 0.72) and wing_reach >= tw * 0.60)

    # When long sleeves hang well below the torso hem, the garment is cropped
    if is_long_sleeve and wing_y_max > ty1 + 10:
        crop_ratio = float(np.clip((ty1 / max(1, wing_y_max)) / 0.88, 0.65, 1.0))
        torso_aspect = float(torso_aspect * crop_ratio)

    return (max(0, tx0), min(w - 1, tx1)), (ty0, ty1), is_long_sleeve, torso_aspect


def extract_garment(image_bytes: bytes, expect_human: bool = False) -> ExtractedGarment:
    """Segment, tight-crop, and analyze an uploaded garment image (flat-lay or human-worn)."""
    rgba = _load_image_rgba(image_bytes)
    rgb_full = rgba[:, :, :3]

    worn_result = None
    transparent_ratio = float(np.mean(rgba[:, :, 3] < 220))
    if transparent_ratio <= 0.03:
        worn_result = _try_segment_worn_garment(rgb_full, expect_human=expect_human)

    from_human = worn_result is not None
    if worn_result is not None:
        mask, raw_torso_bounds, is_long_sleeve = worn_result
    else:
        mask = _segment_garment(rgba)
        raw_torso_bounds = None
        is_long_sleeve = False

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        h, w = rgba.shape[:2]
        x0, y0, x1, y1 = 0, 0, w, h
        mask = np.full((h, w), 255, dtype=np.uint8)
    else:
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1

    cropped_rgba = rgba[y0:y1, x0:x1].copy()
    cropped_mask = mask[y0:y1, x0:x1].copy()

    # Erode mask slightly before inpainting to strip 1-3px background/skin fringes
    erode_px = max(2, min(6, min(cropped_mask.shape) // 90))
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
    inner_mask = cv2.erode(cropped_mask, erode_kernel, iterations=1)
    if np.sum(inner_mask > 0) < np.sum(cropped_mask > 0) * 0.5:
        inner_mask = cropped_mask

    fg_pixels = cropped_rgba[inner_mask > 0][:, :3]
    if len(fg_pixels) > 0:
        med = np.median(fg_pixels, axis=0).astype(int)
        dominant_color = (int(med[0]), int(med[1]), int(med[2]))
    else:
        dominant_color = (220, 220, 220)

    filled_rgb = _inpaint_background(cropped_rgba[:, :, :3], inner_mask)
    filled_rgba = np.dstack([filled_rgb, np.full(cropped_mask.shape, 255, dtype=np.uint8)])

    ch, cw = inner_mask.shape
    flat_tx, flat_ty, flat_long, torso_aspect = _estimate_flatlay_torso_and_sleeves(inner_mask)

    if raw_torso_bounds is not None:
        tx0 = max(0, min(cw - 2, raw_torso_bounds[0] - x0))
        tx1 = max(tx0 + 1, min(cw - 1, raw_torso_bounds[1] - x0))
        if (tx1 - tx0) < cw * 0.35:
            torso_bounds = flat_tx
        else:
            torso_bounds = (tx0, tx1)
        torso_y_bounds = (0, ch)
        torso_aspect = max(1.35, float(ch / max(1, torso_bounds[1] - torso_bounds[0])))
    else:
        torso_bounds = flat_tx
        torso_y_bounds = flat_ty
        is_long_sleeve = flat_long

    return ExtractedGarment(
        rgba=filled_rgba,
        mask=inner_mask,
        torso_x_range=torso_bounds,
        torso_y_range=torso_y_bounds,
        dominant_color=dominant_color,
        is_long_sleeve=is_long_sleeve,
        from_human=from_human,
        torso_aspect=torso_aspect,
    )


def _extract_clean_sleeve_patch(
    rgb: np.ndarray,
    mask: np.ndarray,
    x_slice: slice,
    y_slice: slice,
    fallback_rgb: np.ndarray,
    fallback_mask: np.ndarray,
    target_w: int,
    target_h: int,
) -> Image.Image:
    """Extract a crisp, streak-free fabric swatch from sleeve/shoulder and tile at 1:1 grain scale."""
    sub_rgb = rgb[y_slice, x_slice]
    sub_mask = mask[y_slice, x_slice]

    best_patch = None
    best_size = 0
    for region_rgb, region_mask in ((sub_rgb, sub_mask), (fallback_rgb, fallback_mask)):
        if region_rgb.size == 0 or region_mask.size == 0 or np.sum(region_mask > 0) < 64:
            continue
        dist = cv2.distanceTransform((region_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        max_r = float(dist.max())
        if max_r >= 10.0:
            cy, cx = np.unravel_index(int(np.argmax(dist)), dist.shape)
            half = max(6, int(max_r * 0.70))
            y0, y1 = max(0, cy - half), min(region_rgb.shape[0], cy + half)
            x0, x1 = max(0, cx - half), min(region_rgb.shape[1], cx + half)
            patch = region_rgb[y0:y1, x0:x1]
            if min(patch.shape[:2]) > best_size:
                best_patch = patch
                best_size = min(patch.shape[:2])
            if best_size >= 28:
                break

    if best_patch is not None and best_size >= 8:
        if np.sum(fallback_mask > 0) > 16:
            fb_mean = fallback_rgb[fallback_mask > 0].mean(axis=0)
            patch_mean = np.maximum(1.0, best_patch.mean(axis=(0, 1)))
            gain = np.clip(fb_mean / patch_mean, 0.75, 1.35)
            best_patch = np.clip(best_patch.astype(np.float32) * gain, 0, 255).astype(np.uint8)

        h_flip = np.concatenate([best_patch, best_patch[:, ::-1]], axis=1)
        tile_2x2 = np.concatenate([h_flip, h_flip[::-1, :]], axis=0)
        tile_h, tile_w = tile_2x2.shape[:2]
        cell_w = max(48, min(target_w, tile_w * 2))
        cell_h = max(48, min(target_h // 2, tile_h * 2))
        cell = np.array(
            Image.fromarray(tile_2x2, mode="RGB").resize((cell_w, cell_h), Image.Resampling.LANCZOS)
        )
        rep_y = (target_h + cell_h - 1) // cell_h
        rep_x = (target_w + cell_w - 1) // cell_w
        tiled = np.tile(cell, (rep_y, rep_x, 1))[:target_h, :target_w]
        return Image.fromarray(tiled, mode="RGB")

    return Image.fromarray(fallback_rgb, mode="RGB").resize(
        (target_w, target_h), Image.Resampling.LANCZOS
    )


def _extract_hanging_sleeve(
    rgb: np.ndarray,
    mask: np.ndarray,
    x_slice: slice,
    fallback_rgb: np.ndarray,
    target_w: int,
    target_h: int,
) -> Image.Image:
    """Extract the full vertical sleeve (shoulder at top -> cuff at bottom) for worn/long sleeves."""
    sub_rgb = rgb[:, x_slice]
    sub_mask = mask[:, x_slice]
    if sub_rgb.size == 0 or sub_mask.size == 0 or np.sum(sub_mask > 0) < 64:
        return Image.fromarray(fallback_rgb, mode="RGB").resize(
            (target_w, target_h), Image.Resampling.LANCZOS
        )

    ys, xs = np.where(sub_mask > 0)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop_rgb = sub_rgb[y0:y1, x0:x1]
    crop_mask = sub_mask[y0:y1, x0:x1]

    # Resample each row's genuine sleeve foreground span across target_w (eliminates empty corner bleed)
    rows_out = []
    min_px = max(6, crop_mask.shape[1] // 6)
    for r in range(crop_mask.shape[0]):
        r_xs = np.where(crop_mask[r] > 0)[0]
        if len(r_xs) >= min_px:
            row_img = Image.fromarray(crop_rgb[r, r_xs[0] : r_xs[-1] + 1][None, :, :], mode="RGB").resize(
                (target_w, 1), Image.Resampling.BILINEAR
            )
            rows_out.append(np.array(row_img)[0])

    if len(rows_out) >= 12:
        sleeve_arr = np.stack(rows_out, axis=0)
        return Image.fromarray(sleeve_arr, mode="RGB").resize(
            (target_w, target_h), Image.Resampling.LANCZOS
        )

    filled = _inpaint_background(crop_rgb, crop_mask)
    return Image.fromarray(filled, mode="RGB").resize(
        (target_w, target_h), Image.Resampling.LANCZOS
    )


def _pack_garment_tile(garment: ExtractedGarment, size: int = HALF_SIZE) -> Image.Image:
    """Pack a single garment side (front or back) into a standardized square UV tile."""
    h, w = garment.rgba.shape[:2]
    tx0, tx1 = garment.torso_x_range
    tx1 = max(tx0 + 1, tx1)
    ty0, ty1 = garment.torso_y_range
    ty1 = max(ty0 + 2, min(h, ty1))

    inset = max(1, (tx1 - tx0) // 65)
    crop_x0 = min(tx1 - 1, tx0 + inset)
    crop_x1 = max(crop_x0 + 1, tx1 - inset)

    rgb = garment.rgba[:, :, :3]
    mask = garment.mask
    torso_crop = Image.fromarray(rgb[ty0:ty1, crop_x0:crop_x1], mode="RGB")

    sleeve_y_end = max(2, int(h * (0.95 if garment.is_long_sleeve else 0.55)))
    torso_px_left = int(size * 0.18)
    torso_px_right = int(size * 0.82)
    torso_px_w = torso_px_right - torso_px_left
    shoulder_strip_w = max(12, (crop_x1 - crop_x0) // 5)

    if garment.is_long_sleeve or garment.from_human:
        left_resized = _extract_hanging_sleeve(
            rgb=rgb,
            mask=mask,
            x_slice=slice(0, tx0),
            fallback_rgb=rgb[ty0:ty1, crop_x0 : crop_x0 + shoulder_strip_w],
            target_w=torso_px_left,
            target_h=size,
        )
        right_resized = _extract_hanging_sleeve(
            rgb=rgb,
            mask=mask,
            x_slice=slice(tx1, w),
            fallback_rgb=rgb[ty0:ty1, max(crop_x0, crop_x1 - shoulder_strip_w) : crop_x1],
            target_w=size - torso_px_right,
            target_h=size,
        )
    else:
        left_resized = _extract_clean_sleeve_patch(
            rgb=rgb,
            mask=mask,
            x_slice=slice(0, tx0),
            y_slice=slice(0, sleeve_y_end),
            fallback_rgb=rgb[ty0:ty1, crop_x0 : crop_x0 + shoulder_strip_w],
            fallback_mask=mask[ty0:ty1, crop_x0 : crop_x0 + shoulder_strip_w],
            target_w=torso_px_left,
            target_h=size,
        )
        right_resized = _extract_clean_sleeve_patch(
            rgb=rgb,
            mask=mask,
            x_slice=slice(tx1, w),
            y_slice=slice(0, sleeve_y_end),
            fallback_rgb=rgb[ty0:ty1, max(crop_x0, crop_x1 - shoulder_strip_w) : crop_x1],
            fallback_mask=mask[ty0:ty1, max(crop_x0, crop_x1 - shoulder_strip_w) : crop_x1],
            target_w=size - torso_px_right,
            target_h=size,
        )

    tile = Image.new("RGB", (size, size), garment.dominant_color)
    torso_resized = torso_crop.resize((torso_px_w, size), Image.Resampling.LANCZOS)

    tile.paste(left_resized, (0, 0))
    tile.paste(torso_resized, (torso_px_left, 0))
    tile.paste(right_resized, (torso_px_right, 0))
    return tile


def build_garment_atlas_with_meta(front_bytes: bytes, back_bytes: bytes) -> Tuple[bytes, bool, float]:
    """Build a 2048x1024 PNG atlas and return (png_bytes, is_long_sleeve, torso_aspect)."""
    front = extract_garment(front_bytes)
    back = extract_garment(back_bytes, expect_human=front.from_human)
    if back.from_human and not front.from_human:
        front = extract_garment(front_bytes, expect_human=True)

    is_long_sleeve = front.is_long_sleeve or back.is_long_sleeve
    if is_long_sleeve:
        front.is_long_sleeve = True
        back.is_long_sleeve = True

    torso_aspect = min(front.torso_aspect, back.torso_aspect)
    front_tile = _pack_garment_tile(front, HALF_SIZE)
    back_tile = _pack_garment_tile(back, HALF_SIZE)

    atlas = Image.new("RGB", (ATLAS_WIDTH, ATLAS_HEIGHT))
    atlas.paste(front_tile, (0, 0))
    atlas.paste(back_tile, (HALF_SIZE, 0))

    out = io.BytesIO()
    atlas.save(out, format="PNG", optimize=True)
    return out.getvalue(), is_long_sleeve, torso_aspect


def build_garment_atlas(front_bytes: bytes, back_bytes: bytes) -> bytes:
    """Build a 2048x1024 PNG texture atlas combining front (left) and back (right)."""
    png_bytes, _, _ = build_garment_atlas_with_meta(front_bytes, back_bytes)
    return png_bytes
