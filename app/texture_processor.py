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
    """Processed garment image with tight crop, torso bounds, and sleeve type."""

    rgba: np.ndarray  # H x W x 4 uint8
    mask: np.ndarray  # H x W uint8 (0 or 255)
    torso_x_range: Tuple[int, int]  # (x_left, x_right) in cropped image coordinates
    dominant_color: Tuple[int, int, int]  # RGB
    is_long_sleeve: bool = False


def _load_image_rgba(image_bytes: bytes) -> np.ndarray:
    """Load raw image bytes as an RGBA uint8 numpy array."""
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    return np.array(pil_img, dtype=np.uint8)


def _try_segment_worn_garment(
    rgb: np.ndarray,
) -> Optional[Tuple[np.ndarray, Tuple[int, int], bool]]:
    """Detect if a human model is wearing the shirt; if so, isolate only the garment.

    Returns (full_image_mask, (raw_torso_x0, raw_torso_x1), is_long_sleeve) or None
    if the image is a flat-lay product photo without a person.
    """
    h, w = rgb.shape[:2]
    pose, seg = _get_mediapipe_models()

    with _MP_LOCK:
        pose_res = pose.process(rgb)
        if not pose_res.pose_landmarks:
            return None
        lm = pose_res.pose_landmarks.landmark
        # Require visible shoulders (11, 12) and hips (23, 24)
        if min(lm[11].visibility, lm[12].visibility, lm[23].visibility, lm[24].visibility) < 0.65:
            return None

        y_sh_norm = 0.5 * (lm[11].y + lm[12].y)
        y_hip_norm = 0.5 * (lm[23].y + lm[24].y)
        torso_h_norm = y_hip_norm - y_sh_norm
        sh_w_norm = abs(lm[11].x - lm[12].x)
        if torso_h_norm < 0.14 or sh_w_norm < 0.12:
            return None

        seg_res = seg.process(rgb)
        person_mask = seg_res.segmentation_mask > 0.45

    y_sh = y_sh_norm * h
    y_hip = y_hip_norm * h
    torso_h = torso_h_norm * h
    x_sh_l = min(lm[11].x, lm[12].x) * w
    x_sh_r = max(lm[11].x, lm[12].x) * w
    sh_w = x_sh_r - x_sh_l

    # Confirm a head/neck region is actually present so flat-lay shirts never trigger this
    head_y0 = max(0, int(y_sh - 0.55 * torso_h))
    head_y1 = max(head_y0 + 1, int(y_sh - 0.14 * torso_h))
    head_box = person_mask[head_y0:head_y1, max(0, int(x_sh_l)):min(w, int(x_sh_r))]
    has_face = lm[0].visibility > 0.65 and lm[0].y < y_sh_norm - 0.04
    has_ears = lm[7].visibility > 0.60 and lm[8].visibility > 0.60 and min(lm[7].y, lm[8].y) < y_sh_norm
    has_head_fg = head_box.size > 0 and float(np.mean(head_box)) > 0.18
    if not (has_face or has_ears or has_head_fg):
        return None

    # Build GrabCut seed mask: eliminate background, head/neck skin, hands, and jeans
    y_collar_top = max(0, int(y_sh - 0.14 * torso_h))
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
        if lm[el_i].visibility > 0.5:
            p1 = (int(np.clip(lm[sh_i].x * w, 0, w - 1)), int(np.clip(lm[sh_i].y * h, 0, h - 1)))
            p2 = (int(np.clip(lm[el_i].x * w, 0, w - 1)), int(np.clip(lm[el_i].y * h, 0, h - 1)))
            cv2.line(gc, p1, p2, int(cv2.GC_FGD), max(6, int(w * 0.04)))

    # Remove exposed neck skin (and bare hands near wrists)
    neck_zone = np.zeros((h, w), dtype=bool)
    ny0 = y_collar_top
    ny1 = min(h, int(y_sh + 0.11 * torso_h))
    nx0 = max(0, int(x_sh_l + 0.24 * sh_w))
    nx1 = min(w, int(x_sh_r - 0.24 * sh_w))
    if ny1 > ny0 and nx1 > nx0:
        neck_zone[ny0:ny1, nx0:nx1] = True

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if lm[0].visibility > 0.65 and lm[0].y < y_sh_norm:
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
                if lm[wr_i].visibility > 0.45:
                    wx = int(np.clip(lm[wr_i].x * w, 0, w - 1))
                    wy = int(np.clip(lm[wr_i].y * h, 0, h - 1))
                    wr_mask = np.zeros((h, w), dtype=bool)
                    rw = max(10, int(sh_w * 0.12))
                    wr_mask[max(0, wy - rw):min(h, wy + rw), max(0, wx - rw):min(w, wx + rw)] = True
                    gc[wr_mask & (skin_dist < 24.0)] = cv2.GC_BGD
    else:
        # Back view: detect skin in neck zone using YCrCb bounds
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
        return None

    # Auto-detect long sleeve vs short sleeve from forearm garment coverage
    forearm_covs = []
    for el_i, wr_i in ((13, 15), (14, 16)):
        if lm[el_i].visibility > 0.5 and lm[wr_i].visibility > 0.4:
            hits = 0
            samples = 20
            for t in np.linspace(0.20, 0.85, samples):
                px = int(np.clip((lm[el_i].x * (1.0 - t) + lm[wr_i].x * t) * w, 0, w - 1))
                py = int(np.clip((lm[el_i].y * (1.0 - t) + lm[wr_i].y * t) * h, 0, h - 1))
                if garment_mask[py, px] > 0:
                    hits += 1
            forearm_covs.append(hits / samples)
    is_long_sleeve = bool(forearm_covs and max(forearm_covs) >= 0.45)

    # Torso X bounds from shoulder landmarks (separates torso from hanging arms)
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


def _estimate_torso_bounds(mask: np.ndarray) -> Tuple[int, int]:
    """Estimate horizontal torso bounds (excluding protruding flat-lay sleeves)."""
    h, w = mask.shape
    y_start = int(h * 0.48)
    y_end = int(h * 0.88)
    band = mask[y_start:y_end, :]

    left_edges = []
    right_edges = []
    for row in band:
        xs = np.where(row > 0)[0]
        if len(xs) > w * 0.2:
            left_edges.append(int(xs[0]))
            right_edges.append(int(xs[-1]))

    if not left_edges:
        return 0, w - 1

    torso_left = int(np.percentile(left_edges, 25))
    torso_right = int(np.percentile(right_edges, 75))

    if (torso_right - torso_left) < w * 0.35:
        return 0, w - 1

    return max(0, torso_left), min(w - 1, torso_right)


def extract_garment(image_bytes: bytes) -> ExtractedGarment:
    """Segment, tight-crop, and analyze an uploaded garment image (flat-lay or human-worn)."""
    rgba = _load_image_rgba(image_bytes)
    rgb_full = rgba[:, :, :3]

    worn_result = None
    transparent_ratio = float(np.mean(rgba[:, :, 3] < 220))
    if transparent_ratio <= 0.03:
        worn_result = _try_segment_worn_garment(rgb_full)

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
    if raw_torso_bounds is not None:
        tx0 = max(0, min(cw - 2, raw_torso_bounds[0] - x0))
        tx1 = max(tx0 + 1, min(cw - 1, raw_torso_bounds[1] - x0))
        if (tx1 - tx0) < cw * 0.35:
            torso_bounds = _estimate_torso_bounds(inner_mask)
        else:
            torso_bounds = (tx0, tx1)
    else:
        torso_bounds = _estimate_torso_bounds(inner_mask)

    return ExtractedGarment(
        rgba=filled_rgba,
        mask=inner_mask,
        torso_x_range=torso_bounds,
        dominant_color=dominant_color,
        is_long_sleeve=is_long_sleeve,
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


def _pack_garment_tile(garment: ExtractedGarment, size: int = HALF_SIZE) -> Image.Image:
    """Pack a single garment side (front or back) into a standardized square UV tile."""
    h, w = garment.rgba.shape[:2]
    tx0, tx1 = garment.torso_x_range
    tx1 = max(tx0 + 1, tx1)

    inset = max(1, (tx1 - tx0) // 65)
    crop_x0 = min(tx1 - 1, tx0 + inset)
    crop_x1 = max(crop_x0 + 1, tx1 - inset)

    rgb = garment.rgba[:, :, :3]
    mask = garment.mask
    torso_crop = Image.fromarray(rgb[:, crop_x0:crop_x1], mode="RGB")

    sleeve_y_end = max(2, int(h * (0.85 if garment.is_long_sleeve else 0.55)))
    torso_px_left = int(size * 0.18)
    torso_px_right = int(size * 0.82)
    torso_px_w = torso_px_right - torso_px_left
    shoulder_strip_w = max(12, (crop_x1 - crop_x0) // 5)

    left_resized = _extract_clean_sleeve_patch(
        rgb=rgb,
        mask=mask,
        x_slice=slice(0, tx0),
        y_slice=slice(0, sleeve_y_end),
        fallback_rgb=rgb[:sleeve_y_end, crop_x0 : crop_x0 + shoulder_strip_w],
        fallback_mask=mask[:sleeve_y_end, crop_x0 : crop_x0 + shoulder_strip_w],
        target_w=torso_px_left,
        target_h=size,
    )
    right_resized = _extract_clean_sleeve_patch(
        rgb=rgb,
        mask=mask,
        x_slice=slice(tx1, w),
        y_slice=slice(0, sleeve_y_end),
        fallback_rgb=rgb[:sleeve_y_end, max(crop_x0, crop_x1 - shoulder_strip_w) : crop_x1],
        fallback_mask=mask[:sleeve_y_end, max(crop_x0, crop_x1 - shoulder_strip_w) : crop_x1],
        target_w=size - torso_px_right,
        target_h=size,
    )

    tile = Image.new("RGB", (size, size), garment.dominant_color)
    torso_resized = torso_crop.resize((torso_px_w, size), Image.Resampling.LANCZOS)

    tile.paste(left_resized, (0, 0))
    tile.paste(torso_resized, (torso_px_left, 0))
    tile.paste(right_resized, (torso_px_right, 0))
    return tile


def build_garment_atlas_with_meta(front_bytes: bytes, back_bytes: bytes) -> Tuple[bytes, bool]:
    """Build a 2048x1024 PNG atlas and return (png_bytes, is_long_sleeve)."""
    front = extract_garment(front_bytes)
    back = extract_garment(back_bytes)

    is_long_sleeve = front.is_long_sleeve or back.is_long_sleeve
    front_tile = _pack_garment_tile(front, HALF_SIZE)
    back_tile = _pack_garment_tile(back, HALF_SIZE)

    atlas = Image.new("RGB", (ATLAS_WIDTH, ATLAS_HEIGHT))
    atlas.paste(front_tile, (0, 0))
    atlas.paste(back_tile, (HALF_SIZE, 0))

    out = io.BytesIO()
    atlas.save(out, format="PNG", optimize=True)
    return out.getvalue(), is_long_sleeve


def build_garment_atlas(front_bytes: bytes, back_bytes: bytes) -> bytes:
    """Build a 2048x1024 PNG texture atlas combining front (left) and back (right)."""
    png_bytes, _ = build_garment_atlas_with_meta(front_bytes, back_bytes)
    return png_bytes
