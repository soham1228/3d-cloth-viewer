# Py 3D Garment Mannequin Service

Converts shopkeeper-uploaded **front** and **back** shirt / t-shirt images into a 3D mannequin wearing the garment (`.glb`), uploads it to **AWS S3**, and provides a **3D web viewer** so customers can rotate and inspect the product in 360°.

## Architecture

```
Shopkeeper uploads Front + Back images
        │
        ▼
Python Service (FastAPI / CLI)
  1. Segments garment & removes studio background
  2. Packs Front + Back UV texture atlas (2048×1024)
  3. Generates 3D mannequin + garment mesh (.glb)
  4. Uploads .glb to S3 (Content-Type: model/gltf-binary)
        │
        ▼
Frontend loads the S3 .glb URL in <model-viewer> (or /viewer?model=<S3_URL>)
```

## Quick Start

1. **Configure S3** (copy `.env.example` to `.env` and fill in your bucket & AWS credentials):

```bash
cp .env.example .env
```

2. **Run the FastAPI service**:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- Open `http://localhost:8000/` for the interactive **Shopkeeper Upload & 3D Studio**.
- API Docs: `http://localhost:8000/docs`.

3. **Or run as a standalone Python script**:

```bash
python generate_3d.py --front path/to/front.png --back path/to/back.png --output out.glb --upload-s3
```

## API Endpoints

### `POST /api/garments/generate-3d`
Accepts `multipart/form-data`:
- `front_image` (file) or `front_image_url` (string)
- `back_image` (file) or `back_image_url` (string)
- `product_id` (optional string)
- `mannequin_gender`: `male` | `female` (default: `male`)
- `mannequin_style`: `studio_charcoal` | `matte_white` | `warm_beige` | `matte_black`
- `include_stand`: `false` | `true` (default: `false`)
- `upload_to_s3`: `true` | `false`

Returns JSON with `s3.model_url` and `viewer_url`.

### `POST /api/garments/generate-3d/raw`
Returns the raw `.glb` binary directly (`model/gltf-binary`).

## Frontend Integration (Loading from S3)

In your React / Next.js / HTML product page, load `<model-viewer>` with the S3 URL returned by the Python service:

```html
<script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>

<model-viewer
  src="https://your-bucket.s3.ap-south-1.amazonaws.com/garments/3d-models/sku123_tshirt.glb"
  alt="3D Shirt Preview"
  camera-controls
  auto-rotate
  shadow-intensity="1"
  style="width: 100%; height: 500px;"
></model-viewer>
```

> **Important S3 CORS Note**: Ensure your S3 bucket CORS policy allows `GET` requests from your website domain so browsers can fetch the `.glb` file:
> ```json
> [
>   {
>     "AllowedHeaders": ["*"],
>     "AllowedMethods": ["GET", "HEAD"],
>     "AllowedOrigins": ["*"],
>     "ExposeHeaders": ["ETag"]
>   }
> ]
> ```
