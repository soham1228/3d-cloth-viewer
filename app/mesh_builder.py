"""Procedural 3D mannequin + garment mesh builder and glTF (.glb) assembler."""

from __future__ import annotations

from typing import Dict, List, Literal, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator

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

# Smooth athletic profile control knots (y from -0.22 hip to +0.50 neck)
_KNOTS_Y = np.array([-0.22, -0.15, -0.04, 0.06, 0.18, 0.28, 0.352, 0.382, 0.408, 0.43, 0.50])
_KNOTS_RX = np.array([0.152, 0.154, 0.144, 0.140, 0.164, 0.172, 0.168, 0.130, 0.086, 0.054, 0.047])
_KNOTS_RF = np.array([0.094, 0.095, 0.089, 0.088, 0.112, 0.106, 0.088, 0.076, 0.063, 0.051, 0.047])
_KNOTS_RB = np.array([0.090, 0.091, 0.085, 0.083, 0.098, 0.095, 0.085, 0.074, 0.062, 0.050, 0.047])

_RX_SPLINE = PchipInterpolator(_KNOTS_Y, _KNOTS_RX, extrapolate=True)
_RF_SPLINE = PchipInterpolator(_KNOTS_Y, _KNOTS_RF, extrapolate=True)
_RB_SPLINE = PchipInterpolator(_KNOTS_Y, _KNOTS_RB, extrapolate=True)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray | float) -> np.ndarray | float:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _torso_profile(
    y: np.ndarray,
    extra_offset: float = 0.0,
    waist_ease: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return C1-smooth anatomical (rx, rz_front, rz_back) across vertical height y."""
    y_clamped = np.clip(y, -0.22, 0.50)
    rx = _RX_SPLINE(y_clamped)
    rf = _RF_SPLINE(y_clamped)
    rb = _RB_SPLINE(y_clamped)

    if extra_offset != 0.0 or waist_ease != 0.0:
        ease = extra_offset + waist_ease * np.exp(-((y - 0.0) / 0.16) ** 2)
        rx = rx + ease
        rf = rf + ease
        rb = rb + ease

    return rx, rf, rb


def _neck_half_width(garment_offset: float = 0.015) -> float:
    """Exact horizontal radius of the garment torso shell at collar height y=0.422."""
    return float(_torso_profile(np.array([0.422]), extra_offset=garment_offset)[0][0])


def _shoulder_ridge_y(abs_x: np.ndarray, garment_offset: float = 0.015) -> np.ndarray:
    """Return top shoulder ridge height y from collar edge (0.422) to athletic shoulder crest (0.352)."""
    x_neck = _neck_half_width(garment_offset)
    x_crest = float(_torso_profile(np.array([0.352]), extra_offset=garment_offset)[0][0])
    s = np.clip((abs_x - x_neck) / max(1e-5, x_crest - x_neck), 0.0, 1.0)
    return 0.422 - 0.070 * (s ** 1.1)


def _eval_torso_x_at_yz(
    side: float,
    y: float,
    z: float,
    extra_offset: float = 0.015,
    waist_ease: float = 0.012,
) -> float:
    """Solve exact x on the torso shell for a given (y, z) so sleeve armhole sits flush on torso."""
    rx_arr, rf_arr, rb_arr = _torso_profile(np.array([y]), extra_offset=extra_offset, waist_ease=waist_ease)
    rx = float(rx_arr[0])
    z_shift = 0.005 * float(np.sin((y - 0.05) * 5.5))
    z_rel = z - z_shift
    rz = float(rf_arr[0] if z_rel >= 0.0 else rb_arr[0])

    shoulder_blend = float(_smoothstep(0.18, 0.335, y)) * (1.0 - float(_smoothstep(0.36, 0.43, y)))
    pow_exp = 0.94 - 0.08 * shoulder_blend
    p_inv = 2.0 / pow_exp
    u_z = min(0.999, abs(z_rel) / max(1e-6, rz))
    x_mag = rx * (max(0.0, 1.0 - (u_z ** p_inv)) ** (1.0 / p_inv))
    return float(side * x_mag)


def _eval_torso_point(
    x: float,
    y: float,
    is_front: bool,
    extra_offset: float = 0.0,
    waist_ease: float = 0.0,
) -> np.ndarray:
    """Evaluate a point (x, y, z) on the smooth torso/garment super-ellipse shell."""
    rx_arr, rf_arr, rb_arr = _torso_profile(np.array([y]), extra_offset=extra_offset, waist_ease=waist_ease)
    rx = float(rx_arr[0])
    rz = float(rf_arr[0] if is_front else rb_arr[0])

    x_clamped = float(np.clip(x, -rx, rx))
    u = abs(x_clamped) / max(1e-6, rx)

    shoulder_blend = float(_smoothstep(0.18, 0.335, y)) * (1.0 - float(_smoothstep(0.36, 0.43, y)))
    pow_exp = 0.94 - 0.08 * shoulder_blend

    p_inv = 2.0 / pow_exp
    z_mag = rz * (max(0.0, 1.0 - (u ** p_inv)) ** (1.0 / p_inv))
    z = z_mag if is_front else -z_mag

    z_shift = 0.005 * float(np.sin((y - 0.05) * 5.5))
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

    shoulder_blend = float(_smoothstep(0.18, 0.335, y)) * (1.0 - float(_smoothstep(0.36, 0.43, y)))
    pow_exp = 0.94 - 0.08 * shoulder_blend

    x = rx_v * np.sign(sin_t) * (np.abs(sin_t) ** pow_exp)
    z = rz * np.sign(cos_t) * (np.abs(cos_t) ** pow_exp)

    z_shift = 0.005 * float(np.sin((y - 0.05) * 5.5))
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
            # Alternate quad diagonal in checkerboard pattern to prevent diagonal shading bands
            use_alt = ((r + c) % 2) == 1
            if flip_winding:
                if use_alt:
                    indices.extend([v00, v10, v01, v01, v10, v11])
                else:
                    indices.extend([v00, v11, v01, v00, v10, v11])
            else:
                if use_alt:
                    indices.extend([v00, v01, v10, v01, v11, v10])
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


def _arm_centerline(side: float, t_vals: np.ndarray) -> np.ndarray:
    """Relaxed mannequin arm with subtle outward shoulder curve and forward elbow bend."""
    elbow_bend = np.clip((t_vals - 0.35) / 0.65, 0.0, 1.0)
    return np.column_stack(
        [
            side * (0.145 + 0.098 * t_vals + 0.008 * np.sin(np.pi * t_vals)),
            0.278 - 0.415 * t_vals,
            -0.004 + 0.034 * (elbow_bend ** 1.35),
        ]
    )


def _build_mannequin_body(material_index: int) -> MeshPrimitiveData:
    """Build sculpted mannequin torso (tucked cleanly inside shirt hem), neck dome, and arms."""
    ntheta = 64
    thetas = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)

    # Torso ends at y=-0.185 inside garment hem (y=-0.215); sleek neck ends at y=0.495 + 0.028 dome
    ys_body = np.linspace(-0.185, 0.495, 42)
    dome_angles = np.linspace(0.0, np.pi / 2.0, 10)[1:]
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
    dome_height = 0.028

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
    bottom_cap = _build_cap_disk(grid[0], np.array([0.0, -0.187, 0.0]), reverse=True, material_index=material_index)

    arm_parts = [torso_prim, bottom_cap]
    for side in (-1.0, 1.0):
        t_vals = np.linspace(0.0, 1.0, 26)
        centers = _arm_centerline(side, t_vals)
        # Anatomical upper arm -> forearm -> sleek wrist taper
        radii = 0.038 - 0.014 * t_vals + 0.002 * np.sin(np.pi * t_vals)
        radii[-2] *= 0.86
        radii[-1] *= 0.62
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
    pole_centers = np.array([[0.0, -0.58, 0.0], [0.0, -0.38, 0.0], [0.0, -0.185, 0.0]])
    pole_radii = np.array([0.016, 0.015, 0.015])
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
        return 0.404 + (shoulder_neck_y - 0.404) * (t ** 1.8)

    if garment_type == "vneck":
        dip_y = 0.305
        return dip_y + (shoulder_neck_y - dip_y) * (t ** 1.25)
    elif garment_type in ("shirt", "polo"):
        dip_y = 0.335
        return dip_y + (shoulder_neck_y - dip_y) * (t ** 1.65)
    else:
        dip_y = 0.342
        return dip_y + (shoulder_neck_y - dip_y) * (t ** 1.85)


def _build_garment_torso_half(
    is_front: bool,
    garment_type: GarmentType,
    material_index: int,
) -> MeshPrimitiveData:
    """Build front or back garment torso+shoulder shell with natural proportions and smooth sides."""
    ny = 52
    nx = 65
    y_bottom = -0.205
    garment_offset = 0.012
    waist_ease = 0.008
    neck_half_w = _neck_half_width(garment_offset)

    thetas_col = np.linspace(-np.pi / 2.0, np.pi / 2.0, nx)
    u_xs = np.sin(thetas_col)
    x_dir = u_xs if is_front else -u_xs

    # Athletic shoulder crest at y=0.352
    y_crest = 0.352
    rx_crest = float(_torso_profile(np.array([y_crest]), extra_offset=garment_offset)[0][0])

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
        th_col = float(thetas_col[ix])
        u_col = float(u_xs[ix])
        xd = float(x_dir[ix])
        y_top = float(y_tops[ix])
        x_top = float(x_top_target[ix])
        is_shoulder = not bool(in_neck[ix])

        for iy, s in enumerate(s_vals):
            y_val = y_bottom + float(s) * (y_top - y_bottom)
            rx_y = float(_torso_profile(np.array([y_val]), extra_offset=garment_offset, waist_ease=waist_ease)[0][0])

            if y_val <= y_crest:
                x_val = xd * rx_y
            else:
                t_up = (y_val - y_crest) / max(1e-5, y_top - y_crest)
                x_at_crest = xd * rx_crest
                x_val = x_at_crest + t_up * (x_top - x_at_crest)
                if is_shoulder:
                    if abs(x_val) > rx_y:
                        x_val = np.sign(x_val) * rx_y
                else:
                    x_val = float(np.clip(x_val, -rx_y, rx_y))

            pt = _eval_torso_point(x_val, y_val, is_front=is_front, extra_offset=garment_offset, waist_ease=waist_ease)
            if s < 0.55 and ix not in (0, nx - 1):
                drape_fade = (1.0 - float(s) / 0.55) ** 1.6
                drape_z = 0.0020 * float(np.cos(th_col)) * float(np.cos(5.0 * th_col)) * drape_fade
                pt[2] += drape_z if is_front else -drape_z
            if (is_shoulder and iy == ny - 1) or ix == 0 or ix == nx - 1:
                pt[2] = 0.005 * float(np.sin((y_val - 0.05) * 5.5))

            grid[iy, ix] = pt

            # Keep center print proportions true to 2D photo while gently wrapping rounded sides
            arc_u = th_col / (np.pi / 2.0)
            x_frac = 0.5 + 0.5 * (0.82 * u_col + 0.18 * arc_u)
            u = u_start + x_frac * (u_end - u_start)
            v_norm = float(np.clip((0.422 - y_val) / (0.422 - y_bottom), 0.0, 1.0))
            v = 0.03 + v_norm * 0.94
            uvs[iy, ix] = [u, v]

    return _triangulate_grid(
        grid,
        uvs,
        material_index,
        "garment_front" if is_front else "garment_back",
        wrap_col=False,
    )


def _build_collar_trim(garment_type: GarmentType, material_index: int) -> MeshPrimitiveData:
    """Build smooth ribbed collar band tapering cleanly at shoulder edges (no corner horns)."""
    ntheta = 64
    thetas = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    garment_offset = 0.012
    neck_half_w = _neck_half_width(garment_offset)

    collar_h = 0.020 if garment_type in ("shirt", "polo") else 0.012
    ny = 4
    grid = np.zeros((ny, ntheta, 3), dtype=np.float32)
    uvs = np.zeros((ny, ntheta, 2), dtype=np.float32)

    for ix, theta in enumerate(thetas):
        sin_t = float(np.sin(theta))
        cos_t = float(np.cos(theta))
        is_front = cos_t >= 0.0

        x_rim = sin_t * neck_half_w
        y_base = float(_collar_y_at_x(np.array([abs(x_rim)]), is_front, garment_type, neck_half_w)[0])
        local_h = collar_h * (0.35 + 0.65 * abs(cos_t))

        u_sample = 0.25 if is_front else 0.75
        for iy in range(ny):
            if iy == 0:
                y_cur, off = y_base - 0.002, garment_offset
            elif iy == 1:
                y_cur, off = y_base + local_h * 0.45, garment_offset + 0.002
            elif iy == 2:
                y_cur, off = y_base + local_h * 0.85, garment_offset + 0.0005
            else:
                y_cur, off = y_base + local_h * 0.50, 0.004

            rx_cur = float(_torso_profile(np.array([y_cur]), extra_offset=off)[0][0])
            x_cur = float(np.clip(x_rim, -rx_cur * 0.99, rx_cur * 0.99))
            pt = _eval_torso_point(x_cur, y_cur, is_front=is_front, extra_offset=off, waist_ease=0.0)
            grid[iy, ix] = pt
            uvs[iy, ix] = [u_sample, 0.12]

    return _triangulate_grid(grid, uvs, material_index, "garment_collar", wrap_col=True)


def _build_sleeve(
    side: float,  # -1.0 = mannequin right (-X), +1.0 = mannequin left (+X)
    garment_type: GarmentType,
    material_index: int,
) -> MeshPrimitiveData:
    """Build one garment sleeve flowing strictly outward from the torso armhole (no pinch)."""
    is_long = garment_type in ("long_sleeve", "shirt")
    max_t = 0.88 if is_long else 0.40

    ns = 24
    ntheta_half = 24
    s_vals = np.linspace(0.0, max_t, ns)
    s_frac = s_vals / max_t

    arm_pts = _arm_centerline(side, s_vals)
    centers = arm_pts.copy()
    centers[:, 0] = arm_pts[:, 0] + side * 0.022 * (1.0 - s_frac)
    centers[:, 1] = arm_pts[:, 1] + 0.016 * (1.0 - s_frac)

    radii_z = (0.060 - 0.006 * s_frac) if not is_long else (0.060 - 0.019 * s_frac)
    radii_y = (0.056 - 0.006 * s_frac) if not is_long else (0.056 - 0.019 * s_frac)

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

            w_arm = float(_smoothstep(0.0, 0.45, sf))
            n_vec = (1.0 - w_arm) * hole_n + w_arm * arm_n[is_idx]
            n_vec = n_vec / max(1e-8, float(np.linalg.norm(n_vec)))
            b_vec = (1.0 - w_arm) * hole_b + w_arm * arm_b[is_idx]
            b_vec = b_vec / max(1e-8, float(np.linalg.norm(b_vec)))

            w_tube = float(_smoothstep(0.0, 0.45, sf))

            for ip, phi in enumerate(phis):
                radial = np.cos(phi) * (rz * n_vec) + np.sin(phi) * (ry * b_vec)
                pt_tube = c + radial

                y_seam = float(np.clip(centers[0, 1] + np.sin(phi) * (radii_y[0] * hole_b[1]), 0.220, 0.352))
                z_seam = float(centers[0, 2] + np.cos(phi) * radii_z[0])
                x_seam = _eval_torso_x_at_yz(side, y_seam, z_seam, extra_offset=0.0125, waist_ease=0.008)
                pt_seam = np.array([x_seam, y_seam, z_seam], dtype=np.float32)

                grid[is_idx, ip] = (1.0 - w_tube) * pt_seam + w_tube * pt_tube

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
