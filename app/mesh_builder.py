"""Procedural 3D mannequin + garment mesh builder and glTF (.glb) assembler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator

from app.glb_writer import MaterialSpec, MeshPrimitiveData, write_glb
from app.texture_processor import build_garment_atlas_with_meta, build_trousers_atlas_with_meta

GarmentType = Literal["tshirt", "vneck", "long_sleeve", "shirt", "polo", "trousers", "shorts"]
MannequinStyle = Literal["studio_charcoal", "matte_white", "warm_beige", "matte_black"]
MannequinGender = Literal["male", "female"]

MANNEQUIN_COLORS: Dict[str, Tuple[float, float, float, float]] = {
    "studio_charcoal": (0.16, 0.17, 0.19, 1.0),
    "matte_white": (0.90, 0.89, 0.87, 1.0),
    "warm_beige": (0.76, 0.68, 0.59, 1.0),
    "matte_black": (0.07, 0.07, 0.08, 1.0),
}

_KNOTS_Y = np.array([-0.22, -0.15, -0.04, 0.06, 0.18, 0.28, 0.352, 0.382, 0.408, 0.43, 0.50])


@dataclass(frozen=True)
class _BodyProfile:
    rx_spline: PchipInterpolator
    rf_spline: PchipInterpolator
    rb_spline: PchipInterpolator
    arm_x_start: float
    arm_x_span: float
    arm_r_start: float
    arm_r_taper: float
    sleeve_rz: float
    sleeve_ry: float
    y_crest: float


_BODY_PROFILES: Dict[str, _BodyProfile] = {
    "male": _BodyProfile(
        rx_spline=PchipInterpolator(
            _KNOTS_Y,
            np.array([0.152, 0.153, 0.147, 0.145, 0.172, 0.180, 0.176, 0.136, 0.090, 0.057, 0.050]),
            extrapolate=True,
        ),
        rf_spline=PchipInterpolator(
            _KNOTS_Y,
            np.array([0.094, 0.095, 0.090, 0.090, 0.115, 0.110, 0.090, 0.078, 0.065, 0.053, 0.049]),
            extrapolate=True,
        ),
        rb_spline=PchipInterpolator(
            _KNOTS_Y,
            np.array([0.090, 0.091, 0.086, 0.085, 0.101, 0.098, 0.087, 0.076, 0.064, 0.052, 0.049]),
            extrapolate=True,
        ),
        arm_x_start=0.152,
        arm_x_span=0.102,
        arm_r_start=0.041,
        arm_r_taper=0.015,
        sleeve_rz=0.062,
        sleeve_ry=0.058,
        y_crest=0.354,
    ),
    "female": _BodyProfile(
        rx_spline=PchipInterpolator(
            _KNOTS_Y,
            np.array([0.166, 0.160, 0.122, 0.110, 0.142, 0.144, 0.140, 0.108, 0.072, 0.044, 0.039]),
            extrapolate=True,
        ),
        rf_spline=PchipInterpolator(
            _KNOTS_Y,
            np.array([0.100, 0.096, 0.076, 0.072, 0.110, 0.092, 0.078, 0.066, 0.054, 0.042, 0.039]),
            extrapolate=True,
        ),
        rb_spline=PchipInterpolator(
            _KNOTS_Y,
            np.array([0.102, 0.098, 0.078, 0.074, 0.088, 0.086, 0.076, 0.065, 0.053, 0.041, 0.039]),
            extrapolate=True,
        ),
        arm_x_start=0.122,
        arm_x_span=0.085,
        arm_r_start=0.030,
        arm_r_taper=0.011,
        sleeve_rz=0.050,
        sleeve_ry=0.047,
        y_crest=0.346,
    ),
}


def _get_profile(gender: MannequinGender) -> _BodyProfile:
    return _BODY_PROFILES.get(gender, _BODY_PROFILES["male"])


def _female_bust_bump(x: np.ndarray | float, y: float) -> np.ndarray | float:
    """Smooth bilateral bust projection for female mannequin & garment front surface."""
    bust_y = float(np.exp(-((y - 0.185) / 0.062) ** 2))
    if bust_y < 1e-4:
        return 0.0 * x if isinstance(x, np.ndarray) else 0.0
    bust_x = np.exp(-((np.abs(x) - 0.054) / 0.044) ** 2)
    return 0.036 * bust_y * bust_x


def _smoothstep(edge0: float, edge1: float, x: np.ndarray | float) -> np.ndarray | float:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _torso_profile(
    y: np.ndarray,
    gender: MannequinGender = "male",
    extra_offset: float = 0.0,
    waist_ease: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return C1-smooth anatomical (rx, rz_front, rz_back) across vertical height y."""
    prof = _get_profile(gender)
    y_clamped = np.clip(y, -0.22, 0.50)
    rx = prof.rx_spline(y_clamped)
    rf = prof.rf_spline(y_clamped)
    rb = prof.rb_spline(y_clamped)

    if extra_offset != 0.0 or waist_ease != 0.0:
        eff_ease = waist_ease * (0.35 if gender == "female" else 1.0)
        ease = extra_offset + eff_ease * np.exp(-((y - 0.0) / 0.16) ** 2)
        rx = rx + ease
        rf = rf + ease
        rb = rb + ease

    return rx, rf, rb


def _neck_half_width(gender: MannequinGender = "male", garment_offset: float = 0.012) -> float:
    """Exact horizontal radius of the garment torso shell at collar height y=0.422."""
    return float(_torso_profile(np.array([0.422]), gender=gender, extra_offset=garment_offset)[0][0])


def _shoulder_ridge_y(
    abs_x: np.ndarray,
    gender: MannequinGender = "male",
    garment_offset: float = 0.012,
) -> np.ndarray:
    """Return top shoulder ridge height y from collar edge (0.422) to shoulder crest."""
    y_crest = _get_profile(gender).y_crest
    x_neck = _neck_half_width(gender=gender, garment_offset=garment_offset)
    x_crest = float(_torso_profile(np.array([y_crest]), gender=gender, extra_offset=garment_offset)[0][0])
    s = np.clip((abs_x - x_neck) / max(1e-5, x_crest - x_neck), 0.0, 1.0)
    return 0.422 - (0.422 - y_crest) * (s ** 1.1)


def _eval_torso_x_at_yz(
    side: float,
    y: float,
    z: float,
    gender: MannequinGender = "male",
    extra_offset: float = 0.0125,
    waist_ease: float = 0.008,
) -> float:
    """Solve exact x on the torso shell for a given (y, z) so sleeve armhole sits flush on torso."""
    rx_arr, rf_arr, rb_arr = _torso_profile(
        np.array([y]), gender=gender, extra_offset=extra_offset, waist_ease=waist_ease
    )
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
    gender: MannequinGender = "male",
    extra_offset: float = 0.0,
    waist_ease: float = 0.0,
) -> np.ndarray:
    """Evaluate a point (x, y, z) on the smooth torso/garment super-ellipse shell."""
    rx_arr, rf_arr, rb_arr = _torso_profile(
        np.array([y]), gender=gender, extra_offset=extra_offset, waist_ease=waist_ease
    )
    rx = float(rx_arr[0])
    rz = float(rf_arr[0] if is_front else rb_arr[0])

    x_clamped = float(np.clip(x, -rx, rx))
    u = abs(x_clamped) / max(1e-6, rx)

    shoulder_blend = float(_smoothstep(0.18, 0.335, y)) * (1.0 - float(_smoothstep(0.36, 0.43, y)))
    pow_exp = 0.94 - 0.08 * shoulder_blend

    p_inv = 2.0 / pow_exp
    z_mag = rz * (max(0.0, 1.0 - (u ** p_inv)) ** (1.0 / p_inv))
    if is_front and gender == "female":
        z_mag += float(_female_bust_bump(x_clamped, y))
    z = z_mag if is_front else -z_mag

    z_shift = 0.005 * float(np.sin((y - 0.05) * 5.5))
    return np.array([x_clamped, y, z + z_shift], dtype=np.float32)


def _eval_torso_ring(
    y: float,
    theta: np.ndarray,
    gender: MannequinGender = "male",
    extra_offset: float = 0.0,
    waist_ease: float = 0.0,
) -> np.ndarray:
    """Evaluate 3D points (N, 3) on a closed mannequin torso cross-section at height y."""
    rx_arr, rf_arr, rb_arr = _torso_profile(
        np.array([y]), gender=gender, extra_offset=extra_offset, waist_ease=waist_ease
    )
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
    if gender == "female":
        bust = _female_bust_bump(x, y)
        z = np.where(cos_t >= 0.0, z + bust, z)

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


def _weld_seam_normals(prim: MeshPrimitiveData, grid_step: float = 1e-4) -> MeshPrimitiveData:
    """Average vertex normals ONLY for genuinely coincident seam vertices (never interior grid vertices)."""
    keys = np.round(prim.positions / grid_step).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    # Only weld vertices that actually share a position with another vertex
    shared_mask = counts[inv] > 1
    if not np.any(shared_mask):
        return prim
    summed = np.zeros_like(prim.normals)
    np.add.at(summed, inv, prim.normals)
    lengths = np.linalg.norm(summed, axis=1, keepdims=True)
    lengths = np.where(lengths < 1e-8, 1.0, lengths)
    welded = np.where(shared_mask[:, None], summed / lengths, prim.normals).astype(np.float32)
    return MeshPrimitiveData(
        name=prim.name,
        positions=prim.positions,
        normals=welded,
        uvs=prim.uvs,
        indices=prim.indices,
        material_index=prim.material_index,
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


def _arm_centerline(side: float, t_vals: np.ndarray, gender: MannequinGender = "male") -> np.ndarray:
    """Relaxed mannequin arm with subtle outward shoulder curve and forward elbow bend."""
    prof = _get_profile(gender)
    elbow_bend = np.clip((t_vals - 0.35) / 0.65, 0.0, 1.0)
    return np.column_stack(
        [
            side * (prof.arm_x_start + prof.arm_x_span * t_vals + 0.008 * np.sin(np.pi * t_vals)),
            0.278 - 0.415 * t_vals,
            -0.004 + 0.034 * (elbow_bend ** 1.35),
        ]
    )


def _build_mannequin_body(material_index: int, gender: MannequinGender = "male") -> MeshPrimitiveData:
    """Build sculpted male or female mannequin torso, neck dome, and arms."""
    prof = _get_profile(gender)
    ntheta = 64
    thetas = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)

    ys_body = np.linspace(-0.185, 0.495, 42)
    dome_angles = np.linspace(0.0, np.pi / 2.0, 10)[1:]
    ny = len(ys_body) + len(dome_angles)

    grid = np.zeros((ny, ntheta, 3), dtype=np.float32)
    uvs = np.zeros((ny, ntheta, 2), dtype=np.float32)

    for iy, y in enumerate(ys_body):
        grid[iy] = _eval_torso_ring(float(y), thetas, gender=gender)
        uvs[iy, :, 0] = thetas / (2.0 * np.pi)
        uvs[iy, :, 1] = iy / (ny - 1)

    base_ring = grid[len(ys_body) - 1]
    base_center_x = float(np.mean(base_ring[:, 0]))
    base_center_z = float(np.mean(base_ring[:, 2]))
    base_y = float(ys_body[-1])
    dome_height = 0.026

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
        centers = _arm_centerline(side, t_vals, gender=gender)
        radii = prof.arm_r_start - prof.arm_r_taper * t_vals + 0.002 * np.sin(np.pi * t_vals)
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


def _collar_y_at_x(
    abs_x: np.ndarray,
    is_front: bool,
    garment_type: GarmentType,
    neck_half_w: float,
    gender: MannequinGender = "male",
) -> np.ndarray:
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
        dip_y = 0.334 if gender == "female" else 0.342
        return dip_y + (shoulder_neck_y - dip_y) * (t ** 1.85)


def _build_garment_torso_half(
    is_front: bool,
    garment_type: GarmentType,
    material_index: int,
    gender: MannequinGender = "male",
    y_bottom: float = -0.205,
) -> MeshPrimitiveData:
    """Build front or back garment torso+shoulder shell contoured to male or female mannequin."""
    prof = _get_profile(gender)
    ny = 52
    nx = 65
    garment_offset = 0.012
    waist_ease = 0.008
    neck_half_w = _neck_half_width(gender=gender, garment_offset=garment_offset)

    thetas_col = np.linspace(-np.pi / 2.0, np.pi / 2.0, nx)
    u_xs = np.sin(thetas_col)
    x_dir = u_xs if is_front else -u_xs

    y_crest = prof.y_crest
    rx_crest = float(_torso_profile(np.array([y_crest]), gender=gender, extra_offset=garment_offset)[0][0])

    x_top_target = x_dir * rx_crest
    abs_x_top = np.abs(x_top_target)
    in_neck = abs_x_top <= neck_half_w

    y_tops = np.where(
        in_neck,
        _collar_y_at_x(
            abs_x_top, is_front=is_front, garment_type=garment_type, neck_half_w=neck_half_w, gender=gender
        ),
        _shoulder_ridge_y(abs_x_top, gender=gender, garment_offset=garment_offset),
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
            rx_y = float(
                _torso_profile(np.array([y_val]), gender=gender, extra_offset=garment_offset, waist_ease=waist_ease)[0][0]
            )

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

            pt = _eval_torso_point(
                x_val, y_val, is_front=is_front, gender=gender, extra_offset=garment_offset, waist_ease=waist_ease
            )
            if s < 0.55 and ix not in (0, nx - 1):
                drape_fade = (1.0 - float(s) / 0.55) ** 1.6
                drape_z = 0.0020 * float(np.cos(th_col)) * float(np.cos(5.0 * th_col)) * drape_fade
                pt[2] += drape_z if is_front else -drape_z
            if (is_shoulder and iy == ny - 1) or ix == 0 or ix == nx - 1:
                pt[2] = 0.005 * float(np.sin((y_val - 0.05) * 5.5))

            grid[iy, ix] = pt

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


def _build_collar_trim(
    garment_type: GarmentType,
    material_index: int,
    gender: MannequinGender = "male",
) -> MeshPrimitiveData:
    """Build smooth ribbed collar band tapering cleanly at shoulder edges."""
    ntheta = 64
    thetas = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    garment_offset = 0.012
    neck_half_w = _neck_half_width(gender=gender, garment_offset=garment_offset)

    collar_h = 0.020 if garment_type in ("shirt", "polo") else 0.012
    ny = 4
    grid = np.zeros((ny, ntheta, 3), dtype=np.float32)
    uvs = np.zeros((ny, ntheta, 2), dtype=np.float32)

    for ix, theta in enumerate(thetas):
        sin_t = float(np.sin(theta))
        cos_t = float(np.cos(theta))
        is_front = cos_t >= 0.0

        x_rim = sin_t * neck_half_w
        y_base = float(
            _collar_y_at_x(np.array([abs(x_rim)]), is_front, garment_type, neck_half_w, gender=gender)[0]
        )
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

            rx_cur = float(_torso_profile(np.array([y_cur]), gender=gender, extra_offset=off)[0][0])
            x_cur = float(np.clip(x_rim, -rx_cur * 0.99, rx_cur * 0.99))
            pt = _eval_torso_point(
                x_cur, y_cur, is_front=is_front, gender=gender, extra_offset=off, waist_ease=0.0
            )
            grid[iy, ix] = pt
            uvs[iy, ix] = [u_sample, 0.12]

    return _triangulate_grid(grid, uvs, material_index, "garment_collar", wrap_col=True)


def _build_sleeve(
    side: float,
    garment_type: GarmentType,
    material_index: int,
    gender: MannequinGender = "male",
) -> MeshPrimitiveData:
    """Build one garment sleeve anchored flush to the torso shell with upright sleeve UVs."""
    prof = _get_profile(gender)
    is_long = garment_type in ("long_sleeve", "shirt")
    max_t = 0.88 if is_long else 0.40

    ns = 24
    ntheta_half = 24
    s_vals = np.linspace(0.0, max_t, ns)
    s_frac = s_vals / max_t

    arm_pts = _arm_centerline(side, s_vals, gender=gender)
    centers = arm_pts.copy()
    centers[:, 0] = arm_pts[:, 0] + side * 0.022 * (1.0 - s_frac)
    centers[:, 1] = arm_pts[:, 1] + 0.016 * (1.0 - s_frac)

    radii_z = (prof.sleeve_rz - 0.006 * s_frac) if not is_long else (prof.sleeve_rz - 0.018 * s_frac)
    radii_y = (prof.sleeve_ry - 0.006 * s_frac) if not is_long else (prof.sleeve_ry - 0.018 * s_frac)

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
            u_inner, u_outer = (0.415, 0.495) if side > 0 else (0.085, 0.005)
        else:
            u_inner, u_outer = (0.585, 0.505) if side > 0 else (0.915, 0.995)

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

                y_seam = float(np.clip(centers[0, 1] + np.sin(phi) * (radii_y[0] * hole_b[1]), 0.220, prof.y_crest))
                z_seam = float(centers[0, 2] + np.cos(phi) * radii_z[0])
                x_seam = _eval_torso_x_at_yz(
                    side, y_seam, z_seam, gender=gender, extra_offset=0.0125, waist_ease=0.008
                )
                pt_seam = np.array([x_seam, y_seam, z_seam], dtype=np.float32)

                grid[is_idx, ip] = (1.0 - w_tube) * pt_seam + w_tube * pt_tube

                outer_frac = 0.5 - 0.5 * side * float(np.sin(phi))
                u = u_inner + outer_frac * (u_outer - u_inner)
                v = 0.04 + sf * 0.92
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


def _build_lower_mannequin_body(
    material_index: int, gender: MannequinGender = "male"
) -> MeshPrimitiveData:
    """Build sculpted lower-body mannequin (waist, pelvis, and two legs) for trousers/shorts."""
    is_fem = gender == "female"
    y_knots = np.array([0.19, 0.24, 0.30, 0.38, 0.44, 0.50])
    rx_vals = (
        np.array([0.148, 0.172, 0.166, 0.126, 0.120, 0.124])
        if is_fem
        else np.array([0.144, 0.160, 0.158, 0.144, 0.142, 0.144])
    )
    rf_vals = (
        np.array([0.072, 0.100, 0.100, 0.080, 0.078, 0.080])
        if is_fem
        else np.array([0.070, 0.095, 0.096, 0.090, 0.089, 0.090])
    )
    rb_vals = (
        np.array([0.078, 0.110, 0.106, 0.082, 0.079, 0.080])
        if is_fem
        else np.array([0.074, 0.098, 0.098, 0.089, 0.088, 0.089])
    )

    rx_sp = PchipInterpolator(y_knots, rx_vals, extrapolate=True)
    rf_sp = PchipInterpolator(y_knots, rf_vals, extrapolate=True)
    rb_sp = PchipInterpolator(y_knots, rb_vals, extrapolate=True)

    ntheta = 64
    thetas = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    ys_pelvis = np.linspace(0.19, 0.50, 22)
    ny = len(ys_pelvis)
    grid = np.zeros((ny, ntheta, 3), dtype=np.float32)
    uvs = np.zeros((ny, ntheta, 2), dtype=np.float32)

    for iy, y in enumerate(ys_pelvis):
        rx = float(rx_sp(y))
        rf = float(rf_sp(y))
        rb = float(rb_sp(y))
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        rz = np.where(sin_t >= 0.0, rf, rb)
        grid[iy, :, 0] = rx * cos_t
        grid[iy, :, 1] = float(y)
        grid[iy, :, 2] = rz * sin_t
        uvs[iy, :, 0] = thetas / (2.0 * np.pi)
        uvs[iy, :, 1] = iy / max(1, ny - 1)

    pelvis_prim = _triangulate_grid(grid, uvs, material_index, "mannequin_pelvis", wrap_col=True)
    top_cap = _build_cap_disk(grid[-1], np.array([0.0, 0.506, 0.0]), reverse=False, material_index=material_index)
    bot_cap = _build_cap_disk(grid[0], np.array([0.0, 0.192, 0.0]), reverse=True, material_index=material_index)
    parts = [pelvis_prim, top_cap, bot_cap]

    # Left & right mannequin legs
    for side in (-1.0, 1.0):
        t_vals = np.linspace(0.0, 1.0, 32)
        y_leg = 0.22 - t_vals * 0.78  # 0.22 -> -0.56
        x_leg = side * (0.084 + 0.010 * t_vals)
        z_leg = 0.006 * np.sin(np.pi * t_vals) - 0.008 * (t_vals ** 2)
        centers = np.stack([x_leg, y_leg, z_leg], axis=1)

        r_thigh = 0.084 if is_fem else 0.080
        r_knee = 0.056
        r_ankle = 0.041
        radii = np.interp(t_vals, [0.0, 0.45, 0.72, 1.0], [r_thigh, r_knee, 0.052, r_ankle])
        parts.append(
            _build_tube_along_curve(
                centers,
                radii,
                ntheta=32,
                cap_start=True,
                cap_end=True,
                material_index=material_index,
                name="mannequin_leg",
            )
        )

    return _merge_primitives(parts, name="Mannequin", material_index=material_index)


def _build_trousers_mesh(
    is_shorts: bool,
    material_index: int,
    gender: MannequinGender = "male",
) -> MeshPrimitiveData:
    """Build procedural 3D trousers or shorts (hip/waistband shell + left/right leg tubes)."""
    is_fem = gender == "female"
    ease = 0.013

    # Match hip outer width at y=0.16 to leg outer edge (x_center + r_thigh) for a flush join
    r_thigh = (0.084 if is_fem else 0.080) + 0.013
    x_leg_0 = 0.084
    x_hip_leg_outer = x_leg_0 + r_thigh

    # 1. Hip / waistband shell (y = 0.16 .. 0.44), morphing smoothly into the two-leg figure-8 at y=0.16
    y_knots = np.array([0.16, 0.23, 0.30, 0.38, 0.44])
    rx_vals = (
        np.array([x_hip_leg_outer, x_hip_leg_outer * 1.01, 0.172 + ease, 0.130 + ease, 0.123 + ease])
        if is_fem
        else np.array([x_hip_leg_outer, x_hip_leg_outer * 1.01, 0.165 + ease, 0.146 + ease, 0.144 + ease])
    )
    rf_vals = (
        np.array([r_thigh, r_thigh * 1.04, 0.104 + ease, 0.082 + ease, 0.080 + ease])
        if is_fem
        else np.array([r_thigh, r_thigh * 1.03, 0.100 + ease, 0.092 + ease, 0.091 + ease])
    )
    rb_vals = (
        np.array([r_thigh, r_thigh * 1.06, 0.114 + ease, 0.084 + ease, 0.081 + ease])
        if is_fem
        else np.array([r_thigh, r_thigh * 1.04, 0.104 + ease, 0.091 + ease, 0.090 + ease])
    )
    rx_sp = PchipInterpolator(y_knots, rx_vals, extrapolate=True)
    rf_sp = PchipInterpolator(y_knots, rf_vals, extrapolate=True)
    rb_sp = PchipInterpolator(y_knots, rb_vals, extrapolate=True)

    parts: List[MeshPrimitiveData] = []
    ys_hip = np.linspace(0.44, 0.16, 20)
    nx_hip = 48

    for is_front in (True, False):
        u_start, u_end = (0.0, 0.5) if is_front else (0.5, 1.0)
        z_sign = 1.0 if is_front else -1.0
        grid = np.zeros((len(ys_hip), nx_hip, 3), dtype=np.float32)
        uvs = np.zeros((len(ys_hip), nx_hip, 2), dtype=np.float32)

        for iy, y in enumerate(ys_hip):
            rx = float(rx_sp(y))
            rz = float(rf_sp(y) if is_front else rb_sp(y))
            # Distribute x from -rx to +rx for front, +rx to -rx for back (matches leg UV orientation & winding)
            t_x = np.linspace(-1.0, 1.0, nx_hip) if is_front else np.linspace(1.0, -1.0, nx_hip)
            x_cos = rx * np.sin(t_x * (np.pi / 2.0))
            x_lin = t_x * rx
            morph = float(_smoothstep(0.32, 0.16, y))
            x_row = (1.0 - morph) * x_cos + morph * x_lin

            z_ellipse = rz * np.sqrt(np.maximum(0.0, 1.0 - (x_row / max(1e-5, rx)) ** 2))
            # Exact two-leg figure-8 z profile at y=0.16
            dx_leg = np.abs(x_row) - x_leg_0
            z_two_legs = np.sqrt(np.maximum(0.0, r_thigh ** 2 - dx_leg ** 2))
            z_row = z_sign * ((1.0 - morph) * z_ellipse + morph * z_two_legs)

            grid[iy, :, 0] = x_row
            grid[iy, :, 1] = float(y)
            grid[iy, :, 2] = z_row

            x_frac = np.linspace(0.0, 1.0, nx_hip)
            uvs[iy, :, 0] = u_start + x_frac * (u_end - u_start)
            uvs[iy, :, 1] = 0.02 + (iy / max(1, len(ys_hip) - 1)) * 0.26

        parts.append(
            _triangulate_grid(
                grid,
                uvs,
                material_index,
                "trousers_hip_front" if is_front else "trousers_hip_back",
                wrap_col=False,
            )
        )

    # Crotch saddle bridge connecting front crotch edge to back crotch edge (tucked slightly inside)
    nx_sad = 24
    nz_sad = 10
    xs_sad = np.linspace(-x_leg_0, x_leg_0, nx_sad)
    s_sad = np.linspace(0.0, 1.0, nz_sad)
    sad_grid = np.zeros((nz_sad, nx_sad, 3), dtype=np.float32)
    sad_uvs = np.zeros((nz_sad, nx_sad, 2), dtype=np.float32)
    z_edge = np.sqrt(np.maximum(0.0, r_thigh ** 2 - (np.abs(xs_sad) - x_leg_0) ** 2))
    for iz, sv in enumerate(s_sad):
        z_row = (1.0 - 2.0 * sv) * z_edge
        y_row = 0.162 + 0.004 * np.sin(np.pi * sv) * (1.0 - (xs_sad / x_leg_0) ** 2)
        sad_grid[iz, :, 0] = xs_sad
        sad_grid[iz, :, 1] = y_row
        sad_grid[iz, :, 2] = z_row
        sad_uvs[iz, :, 0] = 0.25 + 0.02 * (xs_sad / x_leg_0)
        sad_uvs[iz, :, 1] = 0.34
    saddle_prim = _triangulate_grid(sad_grid, sad_uvs, material_index, "trousers_crotch_saddle", wrap_col=False)

    # 2. Left & right leg tubes starting seamlessly at y=0.16
    y_leg_top = 0.16
    y_leg_bot = -0.04 if is_shorts else -0.50
    ny_leg = 14 if is_shorts else 28
    nx_leg = 24
    t_vals = np.linspace(0.0, 1.0, ny_leg)
    y_leg = y_leg_top + t_vals * (y_leg_bot - y_leg_top)
    depth_frac = (y_leg_top - y_leg) / 0.66

    for side in (-1.0, 1.0):
        x_center = side * (x_leg_0 + 0.010 * depth_frac)
        z_center = 0.005 * np.sin(np.pi * depth_frac) - 0.006 * (depth_frac ** 2)
        r_knee = 0.078
        r_hem = 0.070
        radii = np.interp(depth_frac, [0.0, 0.48, 1.0], [r_thigh, r_knee, r_hem])

        for is_front in (True, False):
            # Front tile: u in [0.0, 0.5]; Back tile: u in [0.5, 1.0]
            # On front (-X is left, +X is right): side=-1 -> [0.0, 0.25], side=+1 -> [0.25, 0.5]
            # On back (+X is viewer-left, -X is viewer-right): side=+1 -> [0.5, 0.75], side=-1 -> [0.75, 1.0]
            if is_front:
                leg_u0 = 0.0 if side < 0 else 0.25
            else:
                leg_u0 = 0.75 if side < 0 else 0.50
            leg_u1 = leg_u0 + 0.25

            thetas = np.linspace(np.pi, 0.0, nx_leg) if is_front else np.linspace(0.0, -np.pi, nx_leg)
            grid = np.zeros((ny_leg, nx_leg, 3), dtype=np.float32)
            uvs = np.zeros((ny_leg, nx_leg, 2), dtype=np.float32)

            for iy in range(ny_leg):
                r = float(radii[iy])
                grid[iy, :, 0] = x_center[iy] + r * np.cos(thetas)
                grid[iy, :, 1] = y_leg[iy]
                grid[iy, :, 2] = z_center[iy] + r * np.sin(thetas)

                x_frac = np.linspace(0.0, 1.0, nx_leg)
                uvs[iy, :, 0] = leg_u0 + x_frac * (leg_u1 - leg_u0)
                uvs[iy, :, 1] = 0.28 + (iy / max(1, ny_leg - 1)) * 0.70

            parts.append(
                _triangulate_grid(
                    grid,
                    uvs,
                    material_index,
                    "trousers_leg",
                    wrap_col=False,
                )
            )

    shell_merged = _weld_seam_normals(_merge_primitives(parts, name="Garment", material_index=material_index))
    return _merge_primitives([shell_merged, saddle_prim], name="Garment", material_index=material_index)


def generate_mannequin_garment_glb(
    front_image_bytes: bytes,
    back_image_bytes: bytes,
    garment_type: Optional[GarmentType] = None,
    mannequin_style: MannequinStyle = "studio_charcoal",
    mannequin_gender: MannequinGender = "male",
    include_stand: bool = False,
) -> bytes:
    """Generate a complete 3D mannequin wearing the uploaded front/back garment as a .glb."""
    is_lower_body = garment_type in ("trousers", "shorts")
    if is_lower_body:
        atlas_png, detected_shorts = build_trousers_atlas_with_meta(front_image_bytes, back_image_bytes)
        is_shorts = True if garment_type == "shorts" else detected_shorts
    else:
        atlas_png, is_long_sleeve, torso_aspect = build_garment_atlas_with_meta(
            front_image_bytes, back_image_bytes
        )
        effective_type: GarmentType = garment_type or ("long_sleeve" if is_long_sleeve else "tshirt")
        # Preserve cropped vs full-length silhouette based on measured torso aspect ratio
        y_bottom = (
            -0.205
            if torso_aspect >= 1.45
            else float(np.interp(torso_aspect, [0.85, 1.45], [0.05, -0.205]))
        )
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

    if is_lower_body:
        mannequin_mesh = _build_lower_mannequin_body(material_index=0, gender=mannequin_gender)
        garment_mesh = _build_trousers_mesh(
            is_shorts=is_shorts, material_index=1, gender=mannequin_gender
        )
    else:
        mannequin_mesh = _build_mannequin_body(material_index=0, gender=mannequin_gender)
        garment_parts = [
            _build_garment_torso_half(
                is_front=True,
                garment_type=effective_type,
                material_index=1,
                gender=mannequin_gender,
                y_bottom=y_bottom,
            ),
            _build_garment_torso_half(
                is_front=False,
                garment_type=effective_type,
                material_index=1,
                gender=mannequin_gender,
                y_bottom=y_bottom,
            ),
            _build_collar_trim(garment_type=effective_type, material_index=1, gender=mannequin_gender),
            _build_sleeve(side=-1.0, garment_type=effective_type, material_index=1, gender=mannequin_gender),
            _build_sleeve(side=1.0, garment_type=effective_type, material_index=1, gender=mannequin_gender),
        ]
        garment_mesh = _merge_primitives(garment_parts, name="Garment", material_index=1)

    primitives = [mannequin_mesh, garment_mesh]
    if include_stand:
        primitives.append(_build_stand(material_index=2))

    return write_glb(primitives=primitives, materials=materials, texture_png=atlas_png)
