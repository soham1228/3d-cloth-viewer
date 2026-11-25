"""Texture processing for front and back garment images.

Extracts the garment from background (supporting transparent PNGs and solid/studio
backgrounds), locates torso vs. sleeve regions, inpaints outer margins with edge
fabric colors to eliminate UV bleed, and packs a 2048x1024 texture atlas for the
3D garment mesh.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


ATLAS_WIDTH = 2048
ATLAS_HEIGHT = 1024
HALF_SIZE = 1024


@dataclass
class ExtractedGarment:
    """Processed garment image with tight crop and torso bounds."""

    rgba: np.ndarray  # H x W x 4 uint8
    mask: np.ndarray  # H x W uint8 (0 or 255)
    torso_x_range: Tuple[int, int]  # (x_left, x_right) in cropped image coordinates
    dominant_color: Tuple[int, int, int]  # RGB


def _load_image_rgba(image_bytes: bytes) -> np.ndarray:
    """Load raw image bytes as an RGBA uint8 numpy array."""
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    return np.array(pil_img, dtype=np.uint8)


def _segment_garment(rgba: np.ndarray) -> np.ndarray:
    """Return a binary foreground mask (0 or 255) isolating the garment."""
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

    # Pixels marked 128 in flood are background connected to corners; everything else is foreground
    mask = np.where(flood == 128, 0, 255).astype(np.uint8)

    # Fallback if mask covers almost nothing or almost everything (e.g., busy photo)
    fg_ratio = float(np.mean(mask > 0))
    if fg_ratio < 0.08:
        mask = fg_candidate
        fg_ratio = float(np.mean(mask > 0))

    if fg_ratio < 0.05:
        # Assume the whole image is a close-up fabric/design texture
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

    # Keep the largest non-background component
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = int(np.argmax(areas)) + 1
    result = np.where(labels == largest_idx, 255, 0).astype(np.uint8)
    return result


def _inpaint_background(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill non-garment pixels with nearest inner garment pixel to prevent UV bleed."""
    inv_mask = np.where(mask == 0, 255, 0).astype(np.uint8)
    if np.all(inv_mask == 0) or np.all(mask == 0):
        return rgb

    # Nearest foreground pixel copy (zero halo / zero background fringe bleed)
    _, labels = cv2.distanceTransformWithLabels(
        inv_mask, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    fg_coords = np.argwhere(mask > 0)  # (K, 2) in (y, x)
    # OpenCV labels are 1-indexed into foreground pixels in raster order
    nearest_idx = np.clip(labels - 1, 0, len(fg_coords) - 1)
    ny = fg_coords[nearest_idx, 0]
    nx = fg_coords[nearest_idx, 1]
    out = rgb[ny, nx]
    fg = mask > 0
    out[fg] = rgb[fg]
    return out


def _estimate_torso_bounds(mask: np.ndarray) -> Tuple[int, int]:
    """Estimate the horizontal torso bounds (excluding protruding sleeves).

    In a standard shirt flat-lay, sleeves widen the upper 45% of the garment,
    whereas the lower 50% (y in 48%..88%) reflects the torso width.
    """
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

    # Sanity check: torso should be at least 35% of image width
    if (torso_right - torso_left) < w * 0.35:
        return 0, w - 1

    return max(0, torso_left), min(w - 1, torso_right)


def extract_garment(image_bytes: bytes) -> ExtractedGarment:
    """Segment, tight-crop, and analyze an uploaded garment image."""
    rgba = _load_image_rgba(image_bytes)
    mask = _segment_garment(rgba)

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

    # Erode mask slightly before inpainting to strip 1-3px white/studio anti-alias fringes
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

    # Inpaint background + outer fringe with inner garment edge colors so no background leaks at seams
    filled_rgb = _inpaint_background(cropped_rgba[:, :, :3], inner_mask)
    filled_rgba = np.dstack([filled_rgb, np.full(cropped_mask.shape, 255, dtype=np.uint8)])

    torso_bounds = _estimate_torso_bounds(inner_mask)
    return ExtractedGarment(
        rgba=filled_rgba,
        mask=inner_mask,
        torso_x_range=torso_bounds,
        dominant_color=dominant_color,
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

    # Prefer a large clean patch (larger radius = sharper fabric grain); fallback to shoulder torso
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
            # Only keep sleeve patch directly if it's reasonably sharp (>= 28px); else prefer larger shoulder patch
            if best_size >= 28:
                break

    if best_patch is not None and best_size >= 8:
        # Build seamless 2x2 mirrored tile, then repeat at natural fabric scale instead of stretching 40x
        h_flip = np.concatenate([best_patch, best_patch[:, ::-1]], axis=1)
        tile_2x2 = np.concatenate([h_flip, h_flip[::-1, :]], axis=0)
        # Scale tile_2x2 so each original pixel stays crisp (around 1.5x-2x of input)
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

    # Inset torso crop slightly (1.5%) to avoid edge shadows/fringes along side seams
    inset = max(1, (tx1 - tx0) // 65)
    crop_x0 = min(tx1 - 1, tx0 + inset)
    crop_x1 = max(crop_x0 + 1, tx1 - inset)

    rgb = garment.rgba[:, :, :3]
    mask = garment.mask
    torso_crop = Image.fromarray(rgb[:, crop_x0:crop_x1], mode="RGB")

    sleeve_y_end = max(2, int(h * 0.55))
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


def build_garment_atlas(front_bytes: bytes, back_bytes: bytes) -> bytes:
    """Build a 2048x1024 PNG texture atlas combining front (left) and back (right)."""
    front = extract_garment(front_bytes)
    back = extract_garment(back_bytes)

    front_tile = _pack_garment_tile(front, HALF_SIZE)
    back_tile = _pack_garment_tile(back, HALF_SIZE)

    atlas = Image.new("RGB", (ATLAS_WIDTH, ATLAS_HEIGHT))
    atlas.paste(front_tile, (0, 0))
    atlas.paste(back_tile, (HALF_SIZE, 0))

    out = io.BytesIO()
    atlas.save(out, format="PNG", optimize=True)
    return out.getvalue()
