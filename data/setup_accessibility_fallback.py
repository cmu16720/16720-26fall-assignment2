"""Generate the course-authored PA2 accessibility fallback collection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


SEED = 16720
CATEGORIES = ("orbit_tile", "chevron_tile", "window_tile")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _background(rng: np.random.Generator, size: int) -> Image.Image:
    yy, xx = np.mgrid[:size, :size]
    base = 225 + 12 * np.sin(xx / 18.0) + 8 * np.cos(yy / 23.0)
    noise = rng.normal(0.0, 3.0, (size, size))
    gray = np.clip(base + noise, 190, 250).astype(np.uint8)
    array = np.stack([gray, np.clip(gray + 3, 0, 255), np.clip(gray - 4, 0, 255)], axis=-1)
    return Image.fromarray(array, mode="RGB")


def _draw_orbit(draw: ImageDraw.ImageDraw, center: tuple[int, int], scale: float, color) -> None:
    cx, cy = center
    radius = int(50 * scale)
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=color, width=10)
    draw.ellipse((cx - 15, cy - 15, cx + 15, cy + 15), fill=color)
    for angle in np.linspace(0, 2 * np.pi, 4, endpoint=False):
        x = cx + int(76 * scale * np.cos(angle))
        y = cy + int(76 * scale * np.sin(angle))
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=color)


def _draw_chevron(draw: ImageDraw.ImageDraw, center: tuple[int, int], scale: float, color) -> None:
    cx, cy = center
    width = int(72 * scale)
    height = int(55 * scale)
    for offset in (-34, 0, 34):
        points = [(cx - width, cy + offset - height), (cx, cy + offset),
                  (cx - width, cy + offset + height)]
        draw.line(points, fill=color, width=14, joint="curve")
        points = [(cx, cy + offset - height), (cx + width, cy + offset),
                  (cx, cy + offset + height)]
        draw.line(points, fill=color, width=14, joint="curve")


def _draw_window(draw: ImageDraw.ImageDraw, center: tuple[int, int], scale: float, color) -> None:
    cx, cy = center
    radius = int(78 * scale)
    draw.rounded_rectangle(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        radius=14, outline=color, width=10,
    )
    draw.line((cx, cy - radius, cx, cy + radius), fill=color, width=8)
    draw.line((cx - radius, cy, cx + radius, cy), fill=color, width=8)
    inset = int(20 * scale)
    draw.rectangle((cx + inset, cy - radius + inset, cx + radius - inset, cy - inset), fill=color)


DRAWERS = {
    "orbit_tile": _draw_orbit,
    "chevron_tile": _draw_chevron,
    "window_tile": _draw_window,
}


def generate(output_root: Path) -> list[Path]:
    """Write 24 deterministic, EXIF-free PNGs plus CSV/JSON records."""

    image_dir = output_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    written: list[Path] = []
    palette = ((28, 96, 164), (175, 64, 54), (40, 126, 83))
    for category_index, category in enumerate(CATEGORIES):
        for image_index in range(8):
            rng = np.random.default_rng(SEED + 100 * category_index + image_index)
            image = _background(rng, 224)
            draw = ImageDraw.Draw(image)
            center = (112 + int(rng.integers(-10, 11)), 112 + int(rng.integers(-10, 11)))
            scale = float(rng.uniform(0.82, 1.04))
            color = tuple(int(np.clip(value + rng.integers(-12, 13), 0, 255)) for value in palette[category_index])
            DRAWERS[category](draw, center, scale, color)
            angle = float(rng.uniform(-9.0, 9.0))
            image = image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=(232, 232, 228))
            filename = f"{category}_{image_index + 1:02d}.png"
            destination = image_dir / filename
            image.save(destination, format="PNG", optimize=True)
            written.append(destination)
            rows.append({
                "filename": filename,
                "category": category,
                "split": "development" if image_index < 6 else "held-out",
                "own_photo": "false",
                "source_url_or_creator": "16-720 course staff deterministic generator",
                "license_or_permission": "course-authored accommodation asset; course use and modification permitted",
                "consent_privacy_note": "software-generated; no people, EXIF, or private data",
            })

    manifest_path = output_root / "fallback_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    written.append(manifest_path)

    checksum_path = output_root / "SHA256_MANIFEST.json"
    checksum_payload = {
        "generator": "data/setup_accessibility_fallback.py",
        "seed": SEED,
        "files": {
            str(path.relative_to(output_root)): sha256(path)
            for path in written
        },
    }
    checksum_path.write_text(
        json.dumps(checksum_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    written.append(checksum_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "accessibility_fallback",
    )
    args = parser.parse_args()
    written = generate(args.output)
    print(f"wrote {len(written)} fallback files beneath {args.output}")


if __name__ == "__main__":
    main()
