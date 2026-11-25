"""Procedural 3D mannequin + garment mesh builder and glTF (.glb) assembler."""

from __future__ import annotations

from typing import Dict, List, Literal, Tuple

import numpy as np

from app.glb_writer import MaterialSpec, MeshPrimitiveData, write_glb
from app.texture_processor import build_garment_atlas

GarmentType = Literal["tshirt", "vneck", "long_sleeve", "shirt", "polo"]
MannequinStyle = Literal["studio_charcoal", "matte_white", "warm_beige", "matte_black"]

MANNEQUIN_COLORS: Dict[str, Tuple[float, float, float, float]] = {
    "studio_charcoal": (0.16, 0.17, 0.19, 1.0),
    "matte_white": (0.90, 0.89, 0.87, 1.0),
    "warm_beige": (0.76, 0.68, 0.59, 1.0),
    "matte_black": (0.07, 0.07, 0.08, 1.0),
}

# Anatomical profile control knots (y from -0.26 hip base to +0.55 neck top)
_KNOTS_Y = np.array([-0.26, -0.20, -0.08, 0.04, 0.18, 0.28, 0.34, 0.375, 0.405, 0.43, 0.52])
_KNOTS_RX = np.array([0.166, 0.170, 0.156, 0.150, 0.180, 0.194, 0.196, 0.150, 0.102, 0.068, 0.056])
_KNOTS_RF = np.array([0.100, 0.103, 0.095, 0.094, 0.120, 0.114, 0.096, 0.084, 0.072, 0.062, 0.055])
_KNOTS_RB = np.array([0.096, 0.098, 0.090, 0.087, 0.104, 0.102, 0.092, 0.082, 0.071, 0.061, 0.055])


def _smoothstep(edge0: float, edge1: float, x: np.ndarray | float) -> np.ndarray | float:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _torso_profile(y: np.ndarray, extra_offset: float = 0.0, waist_ease: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return anatomical (rx, rz_front, rz_back) across vertical height y."""
    rx = np.interp(y, _KNOTS_Y, _KNOTS_RX)
    rf = np.interp(y, _KNOTS_Y, _KNOTS_RF)
    rb = np.interp(y, _KNOTS_Y, _KNOTS_RB)

    if extra_offset != 0.0 or waist_ease != 0.0:
        ease = extra_offset + waist_ease * np.exp(-((y - 0.0) / 0.16) ** 2)
        rx = rx + ease
        rf = rf + ease
        rb = rb + ease

    return rx, rf, rb


def _shoulder_ridge_y(abs_x: np.ndarray, garment_offset: float = 0.015) -> np.ndarray:
    """Return the top shoulder ridge height y for horizontal coordinate |x| in [0.082, 0.211].

    This is the exact inverse of the shoulder slope rx(y) so front & back garment
    shells meet seamlessly along z = 0 over the entire shoulder.
    """
    # Shoulder spans from neck edge x_neck (~0.082) out to shoulder crest x_max (~0.196 + offset)
    x_neck = 0.068 + garment_offset
    x_crest = 0.196 + garment_offset
    s = np.clip((abs_x - x_neck) / max(1e-5, x_crest - x_neck), 0.0, 1.0)
    # Matches _KNOTS_Y / _KNOTS_RX shoulder descent from y=0.422 down to y=0.340
    return 0.422 - 0.082 * (s ** 1.15)


def _eval_torso_point(
    x: float,
    y: float,
    is_front: bool,
    extra_offset: float = 0.0,
    waist_ease: float = 0.0,
) -> np.ndarray:
    """Evaluate a point (x, y, z) on the torso/garment surface guaranteed to satisfy the elliptical shell."""
    rx_arr, rf_arr, rb_arr = _torso_profile(np.array([y]), extra_offset=extra_offset, waist_ease=waist_ease)
    rx = float(rx_arr[0])
    rz = float(rf_arr[0] if is_front else rb_arr[0])

    x_clamped = float(np.clip(x, -rx, rx))
    u = abs(x_clamped) / max(1e-6, rx)

    shoulder_blend = float(_smoothstep(0.18, 0.34, y)) * (1.0 - float(_smoothstep(0.36, 0.43, y)))
    pow_exp = 0.94 - 0.10 * shoulder_blend

    # Super-ellipse z from x: (|x|/rx)^(2/p) + (|z|/rz)^(2/p) = 1
    p_inv = 2.0 / pow_exp
    z_mag = rz * (max(0.0, 1.0 - (u ** p_inv)) ** (1.0 / p_inv))
    z = z_mag if is_front else -z_mag

    z_shift = 0.006 * float(np.sin((y - 0.05) * 5.5))
    return np.array([x_clamped, y, z + z_shift], dtype=np.float32)


def _eval_torso_ring(
    y: float,
    theta: np.ndarray,
    extra_offset: float = 0.0,
    waist_ease: float = 0.0,
) -> np.ndarray:
    """Evaluate 3D points (N, 3) on a closed mannequin torso cross-section at height y."""
    rx_arr, rf_arr, rb_arr = _torso_profile(np.array([y]), extra_offset=extra_offset, waist_ease=waist_ease)
    rx_v = float(rx_arr[0])
    rf_v = float(rf_arr[0])
    rb_v = float(rb_arr[0])

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    rz = np.where(cos_t >= 0.0, rf_v, rb_v)

    shoulder_blend = float(_smoothstep(0.18, 0.34, y)) * (1.0 - float(_smoothstep(0.36, 0.43, y)))
    pow_exp = 0.94 - 0.10 * shoulder_blend

    x = rx_v * np.sign(sin_t) * (np.abs(sin_t) ** pow_exp)
    z = rz * np.sign(cos_t) * (np.abs(cos_t) ** pow_exp)

    z_shift = 0.006 * float(np.sin((y - 0.05) * 5.5))
    return np.stack([x, np.full_like(x, y), z + z_shift], axis=-1)


def _compute_smooth_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Compute area-weighted smooth vertex normals from triangle soup."""
    normals = np.zeros_like(positions, dtype=np.float64)
    tris = indices.reshape(-1, 3)
    p0 = positions[tris[:, 0]].astype(np.float64)
    p1 = positions[tris[:, 1]].astype(np.float64)
    p2 = positions[tris[:, 2]].astype(np.float64)

    face_normals = np.cross(p1 - p0, p2 - p0)
    np.add.at(normals, tris[:, 0], face_normals)
    np.add.at(normals, tris[:, 1], face_normals)
    np.add.at(normals, tris[:, 2], face_normals)

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.where(lengths < 1e-8, 1.0, lengths)
    return (normals / lengths).astype(np.float32)


def _triangulate_grid(
    grid: np.ndarray,
    uvs: np.ndarray,
    material_index: int,
    name: str,
    wrap_col: bool = False,
    flip_winding: bool = False,
) -> MeshPrimitiveData:
    """Convert a (R, C, 3) position grid into a smooth-normal MeshPrimitiveData."""
    rows, cols, _ = grid.shape
    positions = grid.reshape(-1, 3).astype(np.float32)
    uv_flat = uvs.reshape(-1, 2).astype(np.float32)

    indices: List[int] = []
    c_limit = cols if wrap_col else cols - 1
    for r in range(rows - 1):
        for c in range(c_limit):
            c_next = (c + 1) % cols
            v00 = r * cols + c
            v01 = r * cols + c_next
            v10 = (r + 1) * cols + c
            v11 = (r + 1) * cols + c_next
            if flip_winding:
                indices.extend([v00, v11, v01, v00, v10, v11])
            else:
                indices.extend([v00, v01, v11, v00, v11, v10])

    idx_arr = np.array(indices, dtype=np.uint32)
    normals = _compute_smooth_normals(positions, idx_arr)

    return MeshPrimitiveData(
        name=name,
        positions=positions,
        normals=normals,
        uvs=uv_flat,
        indices=idx_arr,
        material_index=material_index,
    )


def _merge_primitives(parts: List[MeshPrimitiveData], name: str, material_index: int) -> MeshPrimitiveData:
    """Combine multiple primitives sharing a material into one draw call."""
    pos_list, nrm_list, uv_list, idx_list = [], [], [], []
    offset = 0
    for p in parts:
        pos_list.append(p.positions)
        nrm_list.append(p.normals)
        uv_list.append(p.uvs)
        idx_list.append(p.indices + offset)
        offset += len(p.positions)

    return MeshPrimitiveData(
        name=name,
        positions=np.vstack(pos_list),
        normals=np.vstack(nrm_list),
        uvs=np.vstack(uv_list),
        indices=np.concatenate(idx_list),
        material_index=material_index,
    )


def _build_cap_disk(
    ring: np.ndarray,
    center: np.ndarray,
    reverse: bool,
    material_index: int,
) -> MeshPrimitiveData:
    """Create a flat end cap from a closed boundary ring."""
    n = len(ring)
    positions = np.vstack([center[None, :], ring]).astype(np.float32)
    uvs = np.zeros((n + 1, 2), dtype=np.float32)
    indices: List[int] = []
    for i in range(n):
        nxt = (i + 1) % n
        if reverse:
            indices.extend([0, nxt + 1, i + 1])
        else:
            indices.extend([0, i + 1, nxt + 1])
    idx_arr = np.array(indices, dtype=np.uint32)
    normals = _compute_smooth_normals(positions, idx_arr)
    return MeshPrimitiveData(
        name="cap",
        positions=positions,
        normals=normals,
        uvs=uvs,
        indices=idx_arr,
        material_index=material_index,
    )


def _build_tube_along_curve(
    centers: np.ndarray,
    radii: np.ndarray,
    ntheta: int = 32,
    cap_start: bool = False,
    cap_end: bool = False,
    material_index: int = 0,
    name: str = "tube",
) -> MeshPrimitiveData:
    """Build a smooth capped cylinder/arm along 3D centerline points."""
    n_pts = len(centers)
    thetas = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)

    grid = np.zeros((n_pts, ntheta, 3), dtype=np.float32)
    uvs = np.zeros((n_pts, ntheta, 2), dtype=np.float32)

    prev_normal = np.array([0.0, 0.0, 1.0])
    for i in range(n_pts):
        if i == 0:
            tangent = centers[1] - centers[0]
        elif i == n_pts - 1:
            tangent = centers[-1] - centers[-2]
        else:
            tangent = centers[i + 1] - centers[i - 1]
        tangent = tangent / max(1e-8, float(np.linalg.norm(tangent)))

        n_vec = prev_normal - np.dot(prev_normal, tangent) * tangent
        n_norm = float(np.linalg.norm(n_vec))
        if n_norm < 1e-5:
            ref = np.array([1.0, 0.0, 0.0]) if abs(tangent[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            n_vec = np.cross(tangent, ref)
            n_norm = float(np.linalg.norm(n_vec))
        n_vec = n_vec / n_norm
        b_vec = np.cross(tangent, n_vec)
        prev_normal = n_vec

        r = radii[i]
        ring = (
            centers[i][None, :]
            + r * (np.cos(thetas)[:, None] * n_vec[None, :] + np.sin(thetas)[:, None] * b_vec[None, :])
        )
        grid[i] = ring
        uvs[i, :, 0] = thetas / (2.0 * np.pi)
        uvs[i, :, 1] = i / max(1, n_pts - 1)

    parts = [_triangulate_grid(grid, uvs, material_index, name, wrap_col=True)]
    if cap_start:
        parts.append(_build_cap_disk(grid[0], centers[0], reverse=True, material_index=material_index))
    if cap_end:
        parts.append(_build_cap_disk(grid[-1], centers[-1], reverse=False, material_index=material_index))

    return _merge_primitives(parts, name=name, material_index=material_index)


def _build_mannequin_body(material_index: int) -> MeshPrimitiveData:
    """Build sculpted mannequin torso, spherical neck cap, and arms."""
    ntheta = 64
    thetas = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)

    # Torso + cylindrical neck up to y=0.52, then smooth dome rings up to y=0.558
    ys_body = np.linspace(-0.26, 0.52, 42)
    dome_angles = np.linspace(0.0, np.pi / 2.0, 10)[1:]  # 0 -> equator, pi/2 -> pole
    ny = len(ys_body) + len(dome_angles)

    grid = np.zeros((ny, ntheta, 3), dtype=np.float32)
    uvs = np.zeros((ny, ntheta, 2), dtype=np.float32)

    for iy, y in enumerate(ys_body):
        grid[iy] = _eval_torso_ring(float(y), thetas)
        uvs[iy, :, 0] = thetas / (2.0 * np.pi)
        uvs[iy, :, 1] = iy / (ny - 1)

    base_ring = grid[len(ys_body) - 1]
    base_center_x = float(np.mean(base_ring[:, 0]))
    base_center_z = float(np.mean(base_ring[:, 2]))
    base_y = float(ys_body[-1])
    dome_height = 0.038

    for j, ang in enumerate(dome_angles):
        iy = len(ys_body) + j
        scale = float(np.cos(ang))
        y_dome = base_y + dome_height * float(np.sin(ang))
        ring = base_ring.copy()
        ring[:, 0] = base_center_x + (ring[:, 0] - base_center_x) * scale
        ring[:, 2] = base_center_z + (ring[:, 2] - base_center_z) * scale
        ring[:, 1] = y_dome
        grid[iy] = ring
        uvs[iy, :, 0] = thetas / (2.0 * np.pi)
        uvs[iy, :, 1] = iy / (ny - 1)

    torso_prim = _triangulate_grid(grid, uvs, material_index, "mannequin_torso", wrap_col=True)
    bottom_cap = _build_cap_disk(grid[0], np.array([0.0, -0.262, 0.0]), reverse=True, material_index=material_index)

    arm_parts = [torso_prim, bottom_cap]
    for side in (-1.0, 1.0):
        t_vals = np.linspace(0.0, 1.0, 22)
        centers = np.column_stack(
            [
                side * (0.165 + 0.145 * t_vals),
                0.282 - 0.39 * t_vals,
                -0.004 + 0.032 * (t_vals ** 1.4),
            ]
        )
        # Keep upper arm slimmer than sleeve so mannequin never pokes through sleeve
        radii = 0.047 - 0.011 * t_vals
        radii[-2] *= 0.88
        radii[-1] *= 0.65
        arm_parts.append(
            _build_tube_along_curve(
                centers,
                radii,
                ntheta=28,
                cap_start=True,
                cap_end=True,
                material_index=material_index,
                name="mannequin_arm",
            )
        )

    return _merge_primitives(arm_parts, name="Mannequin", material_index=material_index)


def _build_stand(material_index: int) -> MeshPrimitiveData:
    """Build sleek boutique stand pole and base plate."""
    pole_centers = np.array([[0.0, -0.60, 0.0], [0.0, -0.42, 0.0], [0.0, -0.24, 0.0]])
    pole_radii = np.array([0.017, 0.016, 0.016])
    pole = _build_tube_along_curve(
        pole_centers, pole_radii, ntheta=24, cap_start=True, cap_end=True, material_index=material_index
    )

    base_centers = np.array([[0.0, -0.622, 0.0], [0.0, -0.614, 0.0], [0.0, -0.602, 0.0]])
    base_radii = np.array([0.185, 0.185, 0.172])
    base = _build_tube_along_curve(
        base_centers, base_radii, ntheta=48, cap_start=True, cap_end=True, material_index=material_index
    )
    return _merge_primitives([pole, base], name="DisplayStand", material_index=material_index)


def _collar_y_at_x(abs_x: np.ndarray, is_front: bool, garment_type: GarmentType, neck_half_w: float) -> np.ndarray:
    """Compute neckline height inside the neck opening |x| <= neck_half_w."""
    t = np.clip(abs_x / neck_half_w, 0.0, 1.0)
    shoulder_neck_y = 0.422
    if not is_front:
        return 0.402 + (shoulder_neck_y - 0.402) * (t ** 1.8)

    if garment_type == "vneck":
        dip_y = 0.305
        return dip_y + (shoulder_neck_y - dip_y) * (t ** 1.25)
    elif garment_type in ("shirt", "polo"):
        dip_y = 0.335
        return dip_y + (shoulder_neck_y - dip_y) * (t ** 1.65)
    else:
        dip_y = 0.340
        return dip_y + (shoulder_neck_y - dip_y) * (t ** 1.85)


def _build_garment_torso_half(
    is_front: bool,
    garment_type: GarmentType,
    material_index: int,
) -> MeshPrimitiveData:
    """Build front or back garment torso+shoulder shell that completely encloses the shoulders."""
    ny = 48
    nx = 65
    y_bottom = -0.235
    garment_offset = 0.015
    waist_ease = 0.011
    neck_half_w = 0.083

    # Horizontal parameter u_x in [-1, +1] from viewer-left to viewer-right
    # Front view: -1 is -X, +1 is +X. Back view: -1 is +X, +1 is -X.
    u_xs = np.linspace(-1.0, 1.0, nx)
    x_dir = u_xs if is_front else -u_xs

    # Max shoulder half-width at shoulder crest (y=0.34)
    rx_crest = float(_torso_profile(np.array([0.34]), extra_offset=garment_offset)[0][0])

    # Compute top boundary y_top for each horizontal column u_x
    x_top_target = x_dir * rx_crest
    abs_x_top = np.abs(x_top_target)
    in_neck = abs_x_top <= neck_half_w

    y_tops = np.where(
        in_neck,
        _collar_y_at_x(abs_x_top, is_front=is_front, garment_type=garment_type, neck_half_w=neck_half_w),
        _shoulder_ridge_y(abs_x_top, garment_offset=garment_offset),
    )

    grid = np.zeros((ny, nx, 3), dtype=np.float32)
    uvs = np.zeros((ny, nx, 2), dtype=np.float32)

    u_start, u_end = (0.095, 0.405) if is_front else (0.595, 0.905)
    s_vals = np.linspace(0.0, 1.0, ny)

    for ix in range(nx):
        u_col = float(u_xs[ix])
        xd = float(x_dir[ix])
        y_top = float(y_tops[ix])
        x_top = float(x_top_target[ix])
        is_shoulder = not bool(in_neck[ix])

        for iy, s in enumerate(s_vals):
            y_val = y_bottom + float(s) * (y_top - y_bottom)
            rx_y = float(_torso_profile(np.array([y_val]), extra_offset=garment_offset, waist_ease=waist_ease)[0][0])

            if is_shoulder:
                # Below crest (y <= 0.34), scale with rx_y; above crest, converge smoothly to x_top (= rx_y at y_top)
                if y_val <= 0.34:
                    x_val = xd * rx_y
                else:
                    t_up = (y_val - 0.34) / max(1e-5, y_top - 0.34)
                    x_at_crest = xd * rx_crest
                    x_val = x_at_crest + t_up * (x_top - x_at_crest)
                    # Ensure x_val stays within rx_y so z is real and reaches 0 cleanly at y_top
                    if abs(x_val) > rx_y:
                        x_val = np.sign(x_val) * rx_y
            else:
                # Inside neck column: keep x smoothly spanning [xd*rx_y .. x_top]
                if y_val <= 0.34:
                    x_val = xd * rx_y
                else:
                    t_up = (y_val - 0.34) / max(1e-5, y_top - 0.34)
                    x_at_crest = xd * rx_crest
                    x_val = x_at_crest + t_up * (x_top - x_at_crest)
                    x_val = float(np.clip(x_val, -rx_y * 0.98, rx_y * 0.98))

            # At the very top ridge of shoulders or side edges, snap z cleanly to ridge so front & back join tightly
            pt = _eval_torso_point(x_val, y_val, is_front=is_front, extra_offset=garment_offset, waist_ease=waist_ease)
            if is_shoulder and iy == ny - 1:
                pt[2] = 0.006 * float(np.sin((y_val - 0.05) * 5.5))
            if ix == 0 or ix == nx - 1:
                pt[2] = 0.006 * float(np.sin((y_val - 0.05) * 5.5))

            grid[iy, ix] = pt

            x_frac = 0.5 * (u_col + 1.0)
            u = u_start + x_frac * (u_end - u_start)
            v_norm = float(np.clip((0.422 - y_val) / (0.422 - y_bottom), 0.0, 1.0))
            v = 0.03 + v_norm * 0.94
            uvs[iy, ix] = [u, v]

    # Note: for front, columns go -X -> +X (CCW outward); for back, columns go +X -> -X (also CCW outward from back!)
    return _triangulate_grid(
        grid,
        uvs,
        material_index,
        "garment_front" if is_front else "garment_back",
        wrap_col=False,
    )


def _build_collar_trim(garment_type: GarmentType, material_index: int) -> MeshPrimitiveData:
    """Build tight, realistic ribbed collar/neckband around the neck hole only."""
    ntheta = 64
    thetas = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    neck_half_w = 0.083
    garment_offset = 0.015

    collar_h = 0.024 if garment_type in ("shirt", "polo") else 0.015
    collar_thick = 0.005

    ny = 4
    grid = np.zeros((ny, ntheta, 3), dtype=np.float32)
    uvs = np.zeros((ny, ntheta, 2), dtype=np.float32)

    for ix, theta in enumerate(thetas):
        sin_t = float(np.sin(theta))
        cos_t = float(np.cos(theta))
        is_front = cos_t >= 0.0
        x_val = sin_t * neck_half_w
        y_base = float(_collar_y_at_x(np.array([abs(x_val)]), is_front, garment_type, neck_half_w)[0]) - 0.003

        u_sample = 0.25 if is_front else 0.75
        for iy in range(ny):
            if iy == 0:
                y_cur, off = y_base, garment_offset - 0.001
            elif iy == 1:
                y_cur, off = y_base + collar_h * 0.55, garment_offset + collar_thick
            elif iy == 2:
                y_cur, off = y_base + collar_h, garment_offset + collar_thick * 0.6
            else:
                y_cur, off = y_base + collar_h * 0.65, garment_offset - 0.007

            # Keep collar tightly hugging the neck circumference
            rx_y, rf_y, rb_y = _torso_profile(np.array([y_cur]), extra_offset=off)
            rz_y = float(rf_y[0] if is_front else rb_y[0])
            rx_neck = min(float(rx_y[0]), neck_half_w + off - garment_offset + 0.004)
            z_cur = (rz_y * abs(cos_t)) * (1.0 if is_front else -1.0)
            z_shift = 0.006 * float(np.sin((y_cur - 0.05) * 5.5))
            grid[iy, ix] = [sin_t * rx_neck, y_cur, z_cur + z_shift]
            uvs[iy, ix] = [u_sample, 0.12]

    return _triangulate_grid(grid, uvs, material_index, "garment_collar", wrap_col=True)


def _build_sleeve(
    side: float,  # -1.0 = mannequin right (-X), +1.0 = mannequin left (+X)
    garment_type: GarmentType,
    material_index: int,
) -> MeshPrimitiveData:
    """Build one garment sleeve anchored flush inside the shoulder armhole with zero gap."""
    is_long = garment_type in ("long_sleeve", "shirt")
    max_t = 0.88 if is_long else 0.43

    ns = 20
    ntheta_half = 20

    s_vals = np.linspace(0.0, max_t, ns)
    s_frac = s_vals / max_t

    # Arm centerline matches mannequin arm inside the shoulder
    centers = np.column_stack(
        [
            side * (0.165 + 0.145 * s_vals),
            0.282 - 0.39 * s_vals,
            -0.004 + 0.032 * (s_vals ** 1.4),
        ]
    )
    # Slightly tucked radius at shoulder tapering cleanly toward cuff
    radii_z = (0.072 - 0.008 * s_frac) if not is_long else (0.072 - 0.022 * s_frac)
    radii_y = (0.070 - 0.008 * s_frac) if not is_long else (0.070 - 0.022 * s_frac)

    # Parallel-transport arm frames for outer rings
    arm_n, arm_b = [], []
    prev_n = np.array([0.0, 0.0, 1.0])
    for i in range(ns):
        if i == 0:
            tan = centers[1] - centers[0]
        elif i == ns - 1:
            tan = centers[-1] - centers[-2]
        else:
            tan = centers[i + 1] - centers[i - 1]
        tan = tan / max(1e-8, float(np.linalg.norm(tan)))
        n_vec = prev_n - np.dot(prev_n, tan) * tan
        n_vec = n_vec / max(1e-8, float(np.linalg.norm(n_vec)))
        b_vec = np.cross(tan, n_vec)
        prev_n = n_vec
        arm_n.append(n_vec)
        arm_b.append(b_vec)

    # Vertical armhole basis at s=0 so sleeve top tucks seamlessly under the shoulder ridge
    hole_n = np.array([0.0, 0.0, 1.0])
    hole_b = np.array([0.0, -1.0, 0.0]) if side > 0 else np.array([0.0, 1.0, 0.0])

    parts: List[MeshPrimitiveData] = []
    for is_front in (True, False):
        phis = (
            np.linspace(-np.pi / 2.0, np.pi / 2.0, ntheta_half)
            if is_front
            else np.linspace(np.pi / 2.0, 1.5 * np.pi, ntheta_half)
        )

        grid = np.zeros((ns, ntheta_half, 3), dtype=np.float32)
        uvs = np.zeros((ns, ntheta_half, 2), dtype=np.float32)

        if is_front:
            u_shoulder, u_cuff = (0.415, 0.495) if side > 0 else (0.085, 0.005)
        else:
            u_shoulder, u_cuff = (0.585, 0.505) if side > 0 else (0.915, 0.995)

        for is_idx in range(ns):
            sf = float(s_frac[is_idx])
            c = centers[is_idx]
            rz = float(radii_z[is_idx])
            ry = float(radii_y[is_idx])

            # Blend smoothly from vertical armhole frame (s=0) to perpendicular arm frame (s >= 0.35)
            w_arm = float(_smoothstep(0.0, 0.35, sf))
            n_vec = (1.0 - w_arm) * hole_n + w_arm * arm_n[is_idx]
            n_vec = n_vec / max(1e-8, float(np.linalg.norm(n_vec)))
            b_vec = (1.0 - w_arm) * hole_b + w_arm * arm_b[is_idx]
            b_vec = b_vec / max(1e-8, float(np.linalg.norm(b_vec)))

            for ip, phi in enumerate(phis):
                radial = np.cos(phi) * (rz * n_vec) + np.sin(phi) * (ry * b_vec)
                pt = c + radial
                # Tuck upper-shoulder side of armhole slightly inward inside torso shoulder seam
                if sf < 0.25:
                    y_rel = float(radial[1])
                    if y_rel > 0:
                        pt[0] -= side * (1.0 - sf / 0.25) * 0.012 * (y_rel / max(1e-4, ry))
                grid[is_idx, ip] = pt

                u = u_shoulder + sf * (u_cuff - u_shoulder)
                y_dir = float(radial[1] / max(1e-5, float(np.linalg.norm(radial))))
                v = 0.48 - 0.32 * y_dir
                uvs[is_idx, ip] = [u, v]

        parts.append(
            _triangulate_grid(
                grid,
                uvs,
                material_index,
                "garment_sleeve",
                wrap_col=False,
                flip_winding=False,
            )
        )

    return _merge_primitives(parts, name="garment_sleeve", material_index=material_index)


def generate_mannequin_garment_glb(
    front_image_bytes: bytes,
    back_image_bytes: bytes,
    garment_type: GarmentType = "tshirt",
    mannequin_style: MannequinStyle = "studio_charcoal",
    include_stand: bool = True,
) -> bytes:
    """Generate a complete 3D mannequin wearing the uploaded front/back garment as a .glb."""
    atlas_png = build_garment_atlas(front_image_bytes, back_image_bytes)
    mannequin_rgba = MANNEQUIN_COLORS.get(mannequin_style, MANNEQUIN_COLORS["studio_charcoal"])

    materials = [
        MaterialSpec(
            name="MannequinMaterial",
            base_color_factor=mannequin_rgba,
            metallic_factor=0.05,
            roughness_factor=0.62,
            double_sided=False,
            use_base_color_texture=False,
        ),
        MaterialSpec(
            name="GarmentFabricMaterial",
            base_color_factor=(1.0, 1.0, 1.0, 1.0),
            metallic_factor=0.0,
            roughness_factor=0.88,
            double_sided=True,
            use_base_color_texture=True,
        ),
        MaterialSpec(
            name="StandMetalMaterial",
            base_color_factor=(0.14, 0.14, 0.16, 1.0),
            metallic_factor=0.75,
            roughness_factor=0.35,
            double_sided=False,
            use_base_color_texture=False,
        ),
    ]

    mannequin_mesh = _build_mannequin_body(material_index=0)
    garment_parts = [
        _build_garment_torso_half(is_front=True, garment_type=garment_type, material_index=1),
        _build_garment_torso_half(is_front=False, garment_type=garment_type, material_index=1),
        _build_collar_trim(garment_type=garment_type, material_index=1),
        _build_sleeve(side=-1.0, garment_type=garment_type, material_index=1),
        _build_sleeve(side=1.0, garment_type=garment_type, material_index=1),
    ]
    garment_mesh = _merge_primitives(garment_parts, name="Garment", material_index=1)

    primitives = [mannequin_mesh, garment_mesh]
    if include_stand:
        primitives.append(_build_stand(material_index=2))

    return write_glb(primitives=primitives, materials=materials, texture_png=atlas_png)
