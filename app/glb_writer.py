"""Minimal, spec-compliant glTF 2.0 binary (.glb) writer."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MeshPrimitiveData:
    """Single triangulated mesh primitive with positions, normals, UVs, and material."""

    name: str
    positions: np.ndarray  # (N, 3) float32
    normals: np.ndarray    # (N, 3) float32
    uvs: np.ndarray        # (N, 2) float32
    indices: np.ndarray    # (M,) uint32
    material_index: int


@dataclass
class MaterialSpec:
    """PBR metallic-roughness material specification."""

    name: str
    base_color_factor: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    metallic_factor: float = 0.0
    roughness_factor: float = 0.85
    double_sided: bool = False
    use_base_color_texture: bool = False


def _pad4(data: bytearray, pad_byte: int = 0) -> None:
    rem = len(data) % 4
    if rem:
        data.extend([pad_byte] * (4 - rem))


def write_glb(
    primitives: List[MeshPrimitiveData],
    materials: List[MaterialSpec],
    texture_png: Optional[bytes] = None,
) -> bytes:
    """Serialize meshes, materials, and optional PNG texture into a glTF 2.0 .glb."""
    bin_buf = bytearray()
    buffer_views: List[Dict[str, Any]] = []
    accessors: List[Dict[str, Any]] = []

    def append_buffer(raw: bytes, target: Optional[int] = None) -> int:
        _pad4(bin_buf, 0)
        offset = len(bin_buf)
        bin_buf.extend(raw)
        bv: Dict[str, Any] = {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(raw),
        }
        if target is not None:
            bv["target"] = target
        idx = len(buffer_views)
        buffer_views.append(bv)
        return idx

    gltf_meshes: List[Dict[str, Any]] = []
    gltf_nodes: List[Dict[str, Any]] = []

    for prim in primitives:
        pos = np.ascontiguousarray(prim.positions, dtype=np.float32)
        nrm = np.ascontiguousarray(prim.normals, dtype=np.float32)
        uv = np.ascontiguousarray(prim.uvs, dtype=np.float32)
        idx = np.ascontiguousarray(prim.indices, dtype=np.uint32)

        pos_bv = append_buffer(pos.tobytes(), target=34962)
        nrm_bv = append_buffer(nrm.tobytes(), target=34962)
        uv_bv = append_buffer(uv.tobytes(), target=34962)
        idx_bv = append_buffer(idx.tobytes(), target=34963)

        pos_min = pos.min(axis=0).astype(float).tolist()
        pos_max = pos.max(axis=0).astype(float).tolist()

        pos_acc = len(accessors)
        accessors.append(
            {
                "bufferView": pos_bv,
                "componentType": 5126,  # FLOAT
                "count": int(len(pos)),
                "type": "VEC3",
                "min": pos_min,
                "max": pos_max,
            }
        )

        nrm_acc = len(accessors)
        accessors.append(
            {
                "bufferView": nrm_bv,
                "componentType": 5126,
                "count": int(len(nrm)),
                "type": "VEC3",
            }
        )

        uv_acc = len(accessors)
        accessors.append(
            {
                "bufferView": uv_bv,
                "componentType": 5126,
                "count": int(len(uv)),
                "type": "VEC2",
            }
        )

        idx_acc = len(accessors)
        accessors.append(
            {
                "bufferView": idx_bv,
                "componentType": 5125,  # UNSIGNED_INT
                "count": int(len(idx)),
                "type": "SCALAR",
            }
        )

        mesh_idx = len(gltf_meshes)
        gltf_meshes.append(
            {
                "name": prim.name,
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": pos_acc,
                            "NORMAL": nrm_acc,
                            "TEXCOORD_0": uv_acc,
                        },
                        "indices": idx_acc,
                        "material": prim.material_index,
                    }
                ],
            }
        )
        gltf_nodes.append({"name": prim.name, "mesh": mesh_idx})

    gltf_materials: List[Dict[str, Any]] = []
    for mat in materials:
        pbr: Dict[str, Any] = {
            "baseColorFactor": list(mat.base_color_factor),
            "metallicFactor": mat.metallic_factor,
            "roughnessFactor": mat.roughness_factor,
        }
        if mat.use_base_color_texture and texture_png is not None:
            pbr["baseColorTexture"] = {"index": 0}
        mat_dict: Dict[str, Any] = {
            "name": mat.name,
            "pbrMetallicRoughness": pbr,
        }
        if mat.double_sided:
            mat_dict["doubleSided"] = True
        gltf_materials.append(mat_dict)

    gltf: Dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "py-3d-viewer"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(gltf_nodes)))}],
        "nodes": gltf_nodes,
        "meshes": gltf_meshes,
        "materials": gltf_materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
    }

    if texture_png is not None:
        img_bv = append_buffer(texture_png, target=None)
        gltf["samplers"] = [
            {
                "magFilter": 9729,  # LINEAR
                "minFilter": 9987,  # LINEAR_MIPMAP_LINEAR
                "wrapS": 33071,     # CLAMP_TO_EDGE
                "wrapT": 33071,     # CLAMP_TO_EDGE
            }
        ]
        gltf["images"] = [{"bufferView": img_bv, "mimeType": "image/png"}]
        gltf["textures"] = [{"sampler": 0, "source": 0}]

    _pad4(bin_buf, 0)
    gltf["buffers"] = [{"byteLength": len(bin_buf)}]

    json_bytes = bytearray(json.dumps(gltf, separators=(",", ":")).encode("utf-8"))
    _pad4(json_bytes, 0x20)

    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_buf)
    header = struct.pack("<III", 0x46546C67, 2, total_length)
    json_chunk = struct.pack("<II", len(json_bytes), 0x4E4F534A) + bytes(json_bytes)
    bin_chunk = struct.pack("<II", len(bin_buf), 0x004E4942) + bytes(bin_buf)

    return header + json_chunk + bin_chunk
