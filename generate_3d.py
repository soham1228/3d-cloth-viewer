"""CLI script to generate a 3D mannequin .glb from front & back shirt images and optionally upload to S3.

Example usage:
    python generate_3d.py --front front.png --back back.png --output out.glb
    python generate_3d.py --front front.jpg --back back.jpg --type polo --mannequin matte_white --upload-s3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.mesh_builder import generate_mannequin_garment_glb
from app.s3_service import upload_glb_to_s3


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 3D mannequin .glb from front and back shirt images.")
    parser.add_argument("--front", required=True, type=Path, help="Path to front shirt/t-shirt image.")
    parser.add_argument("--back", required=True, type=Path, help="Path to back shirt/t-shirt image.")
    parser.add_argument(
        "--type",
        dest="garment_type",
        default="tshirt",
        choices=["tshirt", "vneck", "long_sleeve", "shirt", "polo"],
        help="Garment silhouette style (default: tshirt).",
    )
    parser.add_argument(
        "--mannequin",
        dest="mannequin_style",
        default="studio_charcoal",
        choices=["studio_charcoal", "matte_white", "warm_beige", "matte_black"],
        help="Mannequin surface finish (default: studio_charcoal).",
    )
    parser.add_argument("--no-stand", action="store_true", help="Omit the display stand base and pole.")
    parser.add_argument("--output", "-o", type=Path, default=Path("output_models/garment_3d.glb"), help="Output .glb path.")
    parser.add_argument("--upload-s3", action="store_true", help="Upload the generated .glb to S3 using .env config.")
    parser.add_argument("--s3-name", default=None, help="Filename to use in S3 (defaults to output filename).")
    args = parser.parse_args()

    front_bytes = args.front.read_bytes()
    back_bytes = args.back.read_bytes()

    glb_bytes = generate_mannequin_garment_glb(
        front_image_bytes=front_bytes,
        back_image_bytes=back_bytes,
        garment_type=args.garment_type,
        mannequin_style=args.mannequin_style,
        include_stand=not args.no_stand,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(glb_bytes)
    print(f"Generated 3D model: {args.output} ({len(glb_bytes):,} bytes)")

    if args.upload_s3:
        s3_filename = args.s3_name or args.output.name
        res = upload_glb_to_s3(glb_bytes, filename=s3_filename)
        print(f"Uploaded to S3: {res.s3_uri}")
        print(f"Frontend 3D URL: {res.model_url}")


if __name__ == "__main__":
    main()
