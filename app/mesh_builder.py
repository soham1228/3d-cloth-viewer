"""Procedural 3D mannequin + garment mesh builder and glTF (.glb) assembler."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class _GridSurface:
    positions: np.ndarray  # (ny, ntheta, 3)
    uvs: np.ndarray        # (ny, ntheta, 2)
    wrap_u: bool = False


def _smoothstep(edge0: float, edge1: float, x: np.ndarray | float) -> np.ndarray | float:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _torso_profile(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return anatomical (rx, rz_front, rz_back) across vertical height y in [-0.26, 0.58]."""
    # Key anatomical control points (y, rx, rz_front, rz_back)
    knots_y = np.array([-0.26, -0.22, -0.10, 0.02, 0.16, 0.28, 0.36, 0.42, 0.46, 0.57])
    knots_rx = np.array([0.168, 0.174, 0.158, 0.152, 0.184, 0.205, 0.212, 0.115, 0.066, 0.056])
    knots_rf = np.array([0.102, 0.106, 0.098, 0.096, 0.124, 0.118, 0.098, 0.072, 0.060, 0.055])
    knots_rb = np.array([0.098, 0.100, 0.092, 0.088, 0.106, 0.104, 0.092, 0.070, 0.060, 0.055])

    rx = np.interp(y, knots_y, knots_rx)
    rf = np.interp(y, knots_y, knots_rf)
    rb = np.interp(y, knots_y, knots_rb)
    return rx, rf, rb


def _eval_torso_ring(
    y: float,
    theta: np.ndarray,
    extra_offset: float = 0.0,
    waist_ease: float = 0.0,
) -> np.ndarray:
    """Evaluate 3D points (N, 3) on a torso cross-section at height y."""
    rx, rf, rb = _torso_profile(np.array([y]))
    rx_v = float(rx[0]) + extra_offset + waist_ease * float(np.exp(-((y - 0.0) / 0.15) ** 2))
    rf_v = float(rf[0]) + extra_offset + waist_ease * float(np.exp(-((y - 0.0) / 0.15) ** 2))
    rb_v = float(rb[0]) + extra_offset + waist_ease * float(np.exp(-((y - 0.0) / 0.15) ** 2))

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    rz = np.where(cos_t >= 0.0, rf_v, rb_v)

    # Slight squareness at shoulders, smooth oval elsewhere
    shoulder_blend = float(_smoothstep(0.18, 0.36, y)) * (1.0 - float(_smoothstep(0.37, 0.44, y)))
    pow_exp = 0.92 - 0.14 * shoulder_blend

    x = rx_v * np.sign(sin_t) * (np.abs(sin_t) ** pow_exp)
    z = rz * np.sign(cos_t) * (np.abs(cos_t) ** pow_exp)

    # Subtle spine curve (chest slightly forward, upper back slightly back)
    z_shift = 0.008 * float(np.sin((y - 0.05) * 6.0))
    pts = np.stack([x, np.full_like(x, y), z + z_shift], axis=-1)
    return pts


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


def _merge_primitives(parts: List[MeshPrimitiveData], name: str, material_index: int) -> MeshPrimitiveData:
    """Combine multiple primitives sharing a material into one draw call."""
    pos_list = []
    nrm_list = []
    uv_list = []
    idx_list = []
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

        # Parallel transport frame
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


def _build_cap_disk(
    ring: np.ndarray,
    center: np.ndarray,
    reverse: bool,
    material_index: int,
) -> MeshPrimitiveData:
    """Create a flat/domed end cap from a closed boundary ring."""
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


def _build_mannequin_body(material_index: int) -> MeshPrimitiveData:
    """Build sculpted mannequin torso, neck dome, and both arms."""
    ny = 48
    ntheta = 64
    ys = np.linspace(-0.26, 0.565, ny)
    thetas = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)

    grid = np.zeros((ny, ntheta, 3), dtype=np.float32)
    uvs = np.zeros((ny, ntheta, 2), dtype=np.float32)

    for iy, y in enumerate(ys):
        pts = _eval_torso_ring(float(y), thetas, extra_offset=0.0, waist_ease=0.0)
        # Smoothly round the very top of the neck into a dome
        if y > 0.52:
            t = (y - 0.52) / 0.045
            scale = float(np.sqrt(max(0.02, 1.0 - t * t)))
            pts[:, 0] *= scale
            pts[:, 2] *= scale
        grid[iy] = pts
        uvs[iy, :, 0] = thetas / (2.0 * np.pi)
        uvs[iy, :, 1] = iy / (ny - 1)

    torso_prim = _triangulate_grid(grid, uvs, material_index, "mannequin_torso", wrap_col=True)

    # Bottom hip cap & top neck cap
    bottom_cap = _build_cap_disk(grid[0], np.array([0.0, -0.262, 0.0]), reverse=True, material_index=material_index)
    top_cap = _build_cap_disk(grid[-1], np.array([0.0, 0.572, 0.0]), reverse=False, material_index=material_index)

    # Left (+X) and Right (-X) mannequin arms extending to forearm
    arm_parts = [torso_prim, bottom_cap, top_cap]
    for side in (-1.0, 1.0):
        t_vals = np.linspace(0.0, 1.0, 22)
        centers = np.column_stack(
            [
                side * (0.175 + 0.145 * t_vals),
                0.315 - 0.42 * t_vals,
                -0.01 + 0.035 * (t_vals ** 1.4),
            ]
        )
        radii = 0.056 - 0.017 * t_vals
        # Taper wrist smoothly
        radii[-2] *= 0.88
        radii[-1] *= 0.68
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
    # Pole from y = -0.60 up to -0.24
    pole_centers = np.array(
        [
            [0.0, -0.60, 0.0],
            [0.0, -0.42, 0.0],
            [0.0, -0.24, 0.0],
        ]
    )
    pole_radii = np.array([0.017, 0.016, 0.016])
    pole = _build_tube_along_curve(
        pole_centers, pole_radii, ntheta=24, cap_start=True, cap_end=True, material_index=material_index
    )

    # Base disc from y = -0.62 to -0.60
    base_centers = np.array(
        [
            [0.0, -0.622, 0.0],
            [0.0, -0.614, 0.0],
            [0.0, -0.602, 0.0],
        ]
    )
    base_radii = np.array([0.185, 0.185, 0.172])
    base = _build_tube_along_curve(
        base_centers, base_radii, ntheta=48, cap_start=True, cap_end=True, material_index=material_index
    )

    return _merge_primitives([pole, base], name="DisplayStand", material_index=material_index)


def _neckline_height(theta: np.ndarray, garment_type: GarmentType) -> np.ndarray:
    """Compute top edge y(theta) of the garment torso to carve realistic necklines."""
    cos_t = np.cos(theta)
    front_weight = np.clip(cos_t, 0.0, 1.0)
    back_weight = np.clip(-cos_t, 0.0, 1.0)

    shoulder_y = 0.422
    back_dip = 0.022

    if garment_type == "vneck":
        # Deeper V-cut at front center (theta = 0)
        v_factor = front_weight ** 1.35
        front_dip = 0.135 * v_factor
    elif garment_type in ("shirt", "polo"):
        # Open collar plunge at front center
        front_dip = 0.095 * (front_weight ** 1.7)
    else:
        # Classic crew neck
        front_dip = 0.088 * (front_weight ** 1.9)

    return shoulder_y - front_dip - back_dip * (back_weight ** 1.8)


def _build_garment_torso_half(
    is_front: bool,
    garment_type: GarmentType,
    material_index: int,
) -> MeshPrimitiveData:
    """Build either the front or back garment torso shell with clean UV mapping."""
    ny = 44
    ntheta = 40
    y_bottom = -0.235

    if is_front:
        thetas = np.linspace(-np.pi / 2.0, np.pi / 2.0, ntheta)
        u_start, u_end = 0.09, 0.41  # Front torso region inside 2048x1024 atlas
    else:
        thetas = np.linspace(np.pi / 2.0, 1.5 * np.pi, ntheta)
        u_start, u_end = 0.59, 0.91  # Back torso region inside 2048x1024 atlas

    y_tops = _neckline_height(thetas, garment_type)

    grid = np.zeros((ny, ntheta, 3), dtype=np.float32)
    uvs = np.zeros((ny, ntheta, 2), dtype=np.float32)

    s_vals = np.linspace(0.0, 1.0, ny)  # 0 = bottom hem, 1 = neckline/shoulder top
    sin_t = np.sin(thetas)
    x_norm = 0.5 + 0.5 * sin_t if is_front else 0.5 - 0.5 * sin_t

    for iy, s in enumerate(s_vals):
        y_col = y_bottom + s * (y_tops - y_bottom)
        for ix, theta in enumerate(thetas):
            y_val = float(y_col[ix])
            # Slight cloth ease at waist & gentle flare at hem
            hem_flare = 0.008 * float((1.0 - s) ** 2)
            pts = _eval_torso_ring(
                y_val,
                np.array([theta]),
                extra_offset=0.008 + hem_flare,
                waist_ease=0.012,
            )
            grid[iy, ix] = pts[0]

            # Map (x_norm, y_val) onto garment texture coordinates
            u = u_start + float(x_norm[ix]) * (u_end - u_start)
            # Top of garment image (shoulder = 0.422) -> v = 0.02, bottom hem (-0.235) -> v = 0.98
            v_norm = np.clip((0.422 - y_val) / (0.422 - y_bottom), 0.0, 1.0)
            v = 0.02 + float(v_norm) * 0.96
            uvs[iy, ix] = [u, v]

    return _triangulate_grid(
        grid,
        uvs,
        material_index,
        "garment_front" if is_front else "garment_back",
        wrap_col=False,
    )


def _build_collar_trim(garment_type: GarmentType, material_index: int) -> MeshPrimitiveData:
    """Build ribbed collar / folded shirt collar along the neckline."""
    ntheta = 64
    thetas = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    y_tops = _neckline_height(thetas, garment_type)

    collar_height = 0.028 if garment_type in ("shirt", "polo") else 0.016
    collar_offset = 0.014 if garment_type in ("shirt", "polo") else 0.011

    ny = 4
    grid = np.zeros((ny, ntheta, 3), dtype=np.float32)
    uvs = np.zeros((ny, ntheta, 2), dtype=np.float32)

    for ix, theta in enumerate(thetas):
        y_base = float(y_tops[ix]) - 0.004
        is_front = np.cos(theta) >= 0.0
        u_sample = 0.25 if is_front else 0.75  # Sample neckline fabric color
        for iy, t in enumerate(np.linspace(0.0, 1.0, ny)):
            # Outer lip then tucked inward toward neck
            if iy == 0:
                y_cur, off = y_base, 0.008
            elif iy == 1:
                y_cur, off = y_base + collar_height * 0.55, collar_offset
            elif iy == 2:
                y_cur, off = y_base + collar_height, collar_offset * 0.85
            else:
                y_cur, off = y_base + collar_height * 0.65, 0.002
            pts = _eval_torso_ring(y_cur, np.array([theta]), extra_offset=off, waist_ease=0.0)
            grid[iy, ix] = pts[0]
            uvs[iy, ix] = [u_sample, 0.14]

    return _triangulate_grid(grid, uvs, material_index, "garment_collar", wrap_col=True)


def _build_sleeve(
    side: float,  # -1.0 = mannequin right (-X), +1.0 = mannequin left (+X)
    garment_type: GarmentType,
    material_index: int,
) -> MeshPrimitiveData:
    """Build one garment sleeve split cleanly into front & back UV halves."""
    is_long = garment_type in ("long_sleeve", "shirt")
    max_t = 0.90 if is_long else 0.44

    ns = 18
    ntheta_half = 20

    s_vals = np.linspace(0.0, max_t, ns)
    centers = np.column_stack(
        [
            side * (0.165 + 0.145 * s_vals),
            0.315 - 0.42 * s_vals,
            -0.01 + 0.035 * (s_vals ** 1.4),
        ]
    )
    # Sleeve radius wider at shoulder armhole so it blends smoothly into torso
    s_frac = s_vals / max_t
    radii = (0.088 - 0.018 * s_frac) if not is_long else (0.088 - 0.034 * s_frac)

    # Build orthonormal frame along sleeve centerline
    frames_n = []
    frames_b = []
    prev_n = np.array([0.0, 0.0, 1.0])  # +Z front direction
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
        frames_n.append(n_vec)
        frames_b.append(b_vec)

    parts: List[MeshPrimitiveData] = []

    # Front half of sleeve: angle around sleeve from top (-b) through front (+n) to bottom (+b)
    # Back half of sleeve: angle from bottom (+b) through back (-n) to top (-b)
    for is_front in (True, False):
        if is_front:
            # phi = 0 is +n (front); phi in [-pi/2, +pi/2]
            phis = np.linspace(-np.pi / 2.0, np.pi / 2.0, ntheta_half)
        else:
            phis = np.linspace(np.pi / 2.0, 1.5 * np.pi, ntheta_half)

        grid = np.zeros((ns, ntheta_half, 3), dtype=np.float32)
        uvs = np.zeros((ns, ntheta_half, 2), dtype=np.float32)

        # Determine atlas u range for this sleeve side & front/back
        # Front tile is [0.0, 0.5]: left wing [0.00, 0.09], right wing [0.41, 0.50]
        # Back tile is [0.5, 1.0]: left wing [0.50, 0.59], right wing [0.91, 1.00]
        if is_front:
            if side > 0:  # +X sleeve = right side of front image
                u_shoulder, u_cuff = 0.415, 0.495
            else:         # -X sleeve = left side of front image
                u_shoulder, u_cuff = 0.085, 0.005
        else:
            if side > 0:  # +X sleeve = left side of back image
                u_shoulder, u_cuff = 0.585, 0.505
            else:         # -X sleeve = right side of back image
                u_shoulder, u_cuff = 0.915, 0.995

        for is_idx in range(ns):
            c = centers[is_idx]
            r = radii[is_idx]
            n_vec = frames_n[is_idx]
            b_vec = frames_b[is_idx]
            sf = float(s_frac[is_idx])

            for ip, phi in enumerate(phis):
                pt = c + r * (np.cos(phi) * n_vec + np.sin(phi) * b_vec)
                grid[is_idx, ip] = pt
                u = u_shoulder + sf * (u_cuff - u_shoulder)
                # Map top-to-bottom of sleeve onto vertical [0.12 .. 0.85]
                # Make sure top (+Y) maps to smaller v and underarm (-Y) maps to larger v
                y_dir = float(np.dot(np.cos(phi) * n_vec + np.sin(phi) * b_vec, np.array([0.0, 1.0, 0.0])))
                v = 0.48 - 0.35 * y_dir
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
