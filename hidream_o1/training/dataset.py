from __future__ import annotations

import json
from pathlib import Path

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def create_image_caption_jsonl(image_directory: str, output_filename: str = "train.jsonl") -> str:
    root = Path(image_directory).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_directory}")

    image_paths = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No supported images found in {root}. Use jpg, jpeg, or png files.")

    output_path = root / output_filename
    valid = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for image_path in image_paths:
            caption_path = image_path.with_suffix(".txt")
            if not caption_path.is_file():
                continue
            caption = caption_path.read_text(encoding="utf-8").strip()
            if not caption:
                continue
            row = {
                "image": str(image_path.resolve()),
                "caption": caption,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            valid += 1

    if valid == 0:
        raise RuntimeError(
            f"No valid image/caption pairs found in {root}. Each image needs a matching .txt caption."
        )

    return str(output_path.resolve())


def load_image_caption_manifest(dataset_path: str) -> list[dict[str, str]]:
    manifest_path = Path(dataset_path).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {dataset_path}")

    samples: list[dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {manifest_path}:{line_no}: {exc}") from exc
            image = str(row.get("image") or row.get("image_path") or "").strip()
            caption = str(row.get("caption") or row.get("text") or "").strip()
            if not image or not caption:
                raise ValueError(f"Manifest row {line_no} must contain image and caption fields.")
            if not Path(image).expanduser().is_file():
                raise FileNotFoundError(f"Image from manifest row {line_no} not found: {image}")
            samples.append({"image": image, "caption": caption})

    if not samples:
        raise RuntimeError(f"No samples found in dataset manifest: {manifest_path}")
    return samples
