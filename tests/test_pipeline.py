"""Tests for texture processing, glTF .glb validity, FastAPI endpoints, and mocked S3 upload."""

from __future__ import annotations

import io
import json
import struct
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.main import app
from app.mesh_builder import generate_mannequin_garment_glb
from app.s3_service import S3UploadResult
from app.texture_processor import ATLAS_HEIGHT, ATLAS_WIDTH, build_garment_atlas


def _make_sample_shirt_png(color=(30, 90, 180), label="FRONT") -> bytes:
    """Create a synthetic flat-lay t-shirt image on a white studio background."""
    img = Image.new("RGB", (600, 700), (248, 248, 248))
    draw = ImageDraw.Draw(img)
    # Torso
    draw.rectangle([160, 100, 440, 620], fill=color)
    # Left & right sleeves
    draw.polygon([(160, 100), (50, 220), (105, 290), (160, 230)], fill=color)
    draw.polygon([(440, 100), (550, 220), (495, 290), (440, 230)], fill=color)
    # Collar cutout
    draw.ellipse([245, 75, 355, 145], fill=(248, 248, 248))
    # Chest/back stripe + accent
    draw.rectangle([220, 230, 380, 310], fill=(245, 200, 60))
    draw.text((260, 260), label, fill=(20, 20, 20))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _validate_glb_bytes(glb_bytes: bytes) -> dict:
    """Validate glTF 2.0 binary container and return parsed JSON chunk."""
    assert glb_bytes[:4] == b"glTF"
    magic, version, total_len = struct.unpack("<III", glb_bytes[:12])
    assert magic == 0x46546C67
    assert version == 2
    assert total_len == len(glb_bytes)

    json_len, json_type = struct.unpack("<II", glb_bytes[12:20])
    assert json_type == 0x4E4F534A
    json_end = 20 + json_len
    gltf = json.loads(glb_bytes[20:json_end].decode("utf-8"))

    bin_len, bin_type = struct.unpack("<II", glb_bytes[json_end : json_end + 8])
    assert bin_type == 0x004E4942
    bin_data = glb_bytes[json_end + 8 :]
    assert len(bin_data) == bin_len
    assert gltf["buffers"][0]["byteLength"] == bin_len

    # Validate accessors & bufferViews
    for bv in gltf["bufferViews"]:
        assert bv["byteOffset"] % 4 == 0
        assert bv["byteOffset"] + bv["byteLength"] <= bin_len

    for mesh in gltf["meshes"]:
        for prim in mesh["primitives"]:
            pos_acc = gltf["accessors"][prim["attributes"]["POSITION"]]
            assert "min" in pos_acc and "max" in pos_acc
            assert len(pos_acc["min"]) == 3 and len(pos_acc["max"]) == 3
            idx_acc = gltf["accessors"][prim["indices"]]
            idx_bv = gltf["bufferViews"][idx_acc["bufferView"]]
            idx_raw = bin_data[idx_bv["byteOffset"] : idx_bv["byteOffset"] + idx_bv["byteLength"]]
            indices = np.frombuffer(idx_raw, dtype=np.uint32)
            assert len(indices) == idx_acc["count"]
            assert int(indices.max()) < pos_acc["count"]

    # Validate embedded PNG image
    img_bv = gltf["bufferViews"][gltf["images"][0]["bufferView"]]
    png_bytes = bin_data[img_bv["byteOffset"] : img_bv["byteOffset"] + img_bv["byteLength"]]
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    return gltf


def test_atlas_dimensions():
    front = _make_sample_shirt_png(color=(30, 90, 180), label="FRONT")
    back = _make_sample_shirt_png(color=(30, 90, 180), label="BACK")
    atlas_bytes = build_garment_atlas(front, back)
    img = Image.open(io.BytesIO(atlas_bytes))
    assert img.size == (ATLAS_WIDTH, ATLAS_HEIGHT)


def test_glb_generation_all_styles():
    front = _make_sample_shirt_png(color=(25, 110, 95), label="FRONT")
    back = _make_sample_shirt_png(color=(25, 110, 95), label="BACK")
    for gender in ("male", "female"):
        for gtype in ("tshirt", "vneck", "polo", "long_sleeve", "shirt"):
            glb = generate_mannequin_garment_glb(
                front, back, garment_type=gtype, mannequin_gender=gender
            )
            gltf = _validate_glb_bytes(glb)
            assert len(gltf["meshes"]) == 2  # Mannequin + Garment (no stand pole by default)


def test_api_endpoints_and_mocked_s3():
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/viewer").status_code == 200

    front = _make_sample_shirt_png(label="FRONT")
    back = _make_sample_shirt_png(label="BACK")

    # Local generation without S3
    resp = client.post(
        "/api/garments/generate-3d",
        files={
            "front_image": ("front.png", front, "image/png"),
            "back_image": ("back.png", back, "image/png"),
        },
        data={
            "product_id": "test-sku-01",
            "mannequin_style": "matte_white",
            "upload_to_s3": "false",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["filename"] == "test-sku-01.glb"
    assert payload["s3"]["uploaded"] is False

    # Verify served local .glb
    glb_resp = client.get(payload["local_url"])
    assert glb_resp.status_code == 200
    _validate_glb_bytes(glb_resp.content)

    # Mocked S3 upload
    fake_s3 = S3UploadResult(
        bucket="my-shop-bucket",
        key="garments/3d-models/test-sku-02.glb",
        s3_uri="s3://my-shop-bucket/garments/3d-models/test-sku-02.glb",
        model_url="https://my-shop-bucket.s3.ap-south-1.amazonaws.com/garments/3d-models/test-sku-02.glb",
    )
    with patch("app.main.upload_glb_to_s3", return_value=fake_s3) as mock_up:
        resp_s3 = client.post(
            "/api/garments/generate-3d",
            files={
                "front_image": ("front.png", front, "image/png"),
                "back_image": ("back.png", back, "image/png"),
            },
            data={
                "product_id": "test-sku-02",
                "upload_to_s3": "true",
            },
        )
        assert resp_s3.status_code == 200
        mock_up.assert_called_once()
        s3_payload = resp_s3.json()
        assert s3_payload["s3"]["uploaded"] is True
        assert s3_payload["s3"]["model_url"] == fake_s3.model_url

    # Raw endpoint
    raw_resp = client.post(
        "/api/garments/generate-3d/raw",
        files={
            "front_image": ("front.png", front, "image/png"),
            "back_image": ("back.png", back, "image/png"),
        },
    )
    assert raw_resp.status_code == 200
    _validate_glb_bytes(raw_resp.content)
