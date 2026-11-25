"""FastAPI microservice for converting front/back shirt images into 3D mannequin .glb models."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response

from app.config import settings
from app.mesh_builder import MannequinGender, MannequinStyle, generate_mannequin_garment_glb
from app.s3_service import is_s3_configured, upload_glb_to_s3

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="3D Mannequin Garment Service",
    description="Drapes uploaded front & back shirt/t-shirt images onto a 3D mannequin (.glb) and uploads to S3.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _fetch_image_bytes(url: str) -> bytes:
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        return resp.content
    except requests.RequestException as exc:
        raise HTTPException(status_code=400, detail=f"Failed to download image from URL '{url}': {exc}") from exc


@app.get("/health")
def health():
    return {
        "status": "ok",
        "s3_configured": is_s3_configured(),
        "s3_bucket": settings.s3_bucket_name,
    }


@app.get("/", response_class=HTMLResponse)
def studio_page():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return index_path.read_text(encoding="utf-8")


@app.get("/viewer", response_class=HTMLResponse)
def viewer_page():
    viewer_path = STATIC_DIR / "viewer.html"
    if not viewer_path.exists():
        raise HTTPException(status_code=404, detail="viewer.html not found")
    return viewer_path.read_text(encoding="utf-8")


@app.get("/models/{filename}")
def serve_local_model(filename: str):
    safe_name = Path(filename).name
    model_path = settings.local_output_dir / safe_name
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    return FileResponse(
        path=model_path,
        media_type="model/gltf-binary",
        filename=safe_name,
        headers={"Access-Control-Allow-Origin": "*"},
    )


@app.post("/api/garments/generate-3d")
async def generate_3d_garment(
    front_image: Optional[UploadFile] = File(default=None),
    back_image: Optional[UploadFile] = File(default=None),
    front_image_url: Optional[str] = Form(default=None),
    back_image_url: Optional[str] = Form(default=None),
    product_id: Optional[str] = Form(default=None),
    mannequin_gender: MannequinGender = Form(default="male"),
    mannequin_style: MannequinStyle = Form(default="studio_charcoal"),
    include_stand: bool = Form(default=False),
    upload_to_s3: Optional[bool] = Form(default=None),
    save_local: bool = Form(default=True),
):
    """Generate a 3D mannequin .glb wearing the uploaded front & back shirt images."""
    if front_image is not None:
        front_bytes = await front_image.read()
    elif front_image_url:
        front_bytes = _fetch_image_bytes(front_image_url)
    else:
        raise HTTPException(status_code=422, detail="Provide either front_image file or front_image_url.")

    if back_image is not None:
        back_bytes = await back_image.read()
    elif back_image_url:
        back_bytes = _fetch_image_bytes(back_image_url)
    else:
        raise HTTPException(status_code=422, detail="Provide either back_image file or back_image_url.")

    try:
        glb_bytes = generate_mannequin_garment_glb(
            front_image_bytes=front_bytes,
            back_image_bytes=back_bytes,
            mannequin_style=mannequin_style,
            mannequin_gender=mannequin_gender,
            include_stand=include_stand,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to build 3D garment model: {exc}") from exc

    model_id = (product_id or uuid.uuid4().hex[:12]).strip()
    safe_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in model_id)
    filename = f"{safe_id}.glb"

    local_url: Optional[str] = None
    if save_local:
        out_path = settings.local_output_dir / filename
        out_path.write_bytes(glb_bytes)
        local_url = f"/models/{filename}"

    should_upload = is_s3_configured() if upload_to_s3 is None else upload_to_s3
    s3_info = {"uploaded": False, "bucket": None, "key": None, "s3_uri": None, "model_url": None}

    if should_upload:
        try:
            gender_prefix = f"{settings.s3_prefix.strip('/')}/{mannequin_gender}".strip("/")
            res = upload_glb_to_s3(glb_bytes, filename=filename, prefix=gender_prefix)
            s3_info = {
                "uploaded": True,
                "bucket": res.bucket,
                "key": res.key,
                "s3_uri": res.s3_uri,
                "model_url": res.model_url,
            }
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"3D model built, but S3 upload failed: {exc}") from exc

    active_model_url = s3_info["model_url"] or local_url
    return {
        "model_id": safe_id,
        "filename": filename,
        "mannequin_gender": mannequin_gender,
        "mannequin_style": mannequin_style,
        "file_size_bytes": len(glb_bytes),
        "local_url": local_url,
        "viewer_url": f"/viewer?model={active_model_url}" if active_model_url else None,
        "s3": s3_info,
    }


@app.post("/api/garments/generate-3d/raw")
async def generate_3d_garment_raw(
    front_image: UploadFile = File(...),
    back_image: UploadFile = File(...),
    mannequin_gender: MannequinGender = Form(default="male"),
    mannequin_style: MannequinStyle = Form(default="studio_charcoal"),
    include_stand: bool = Form(default=False),
):
    """Return raw .glb binary directly (useful if another backend handles S3 upload itself)."""
    front_bytes = await front_image.read()
    back_bytes = await back_image.read()
    glb_bytes = generate_mannequin_garment_glb(
        front_image_bytes=front_bytes,
        back_image_bytes=back_bytes,
        mannequin_style=mannequin_style,
        mannequin_gender=mannequin_gender,
        include_stand=include_stand,
    )
    return Response(
        content=glb_bytes,
        media_type="model/gltf-binary",
        headers={"Content-Disposition": 'attachment; filename="garment.glb"'},
    )
