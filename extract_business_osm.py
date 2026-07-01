#!/usr/bin/env python3
"""Extract deduplicated OpenStreetMap business tags from one file or a folder of photos."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import mimetypes
import os
import sys
from pathlib import Path
from typing import Iterable

import exifread
from openrouter import OpenRouter
from dotenv import load_dotenv

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}

PROMPT = """Extract all business information from the photos provided
Extract it in openstreetmap key value format
Ensure there are no duplicates
If there are several businesses in photos, extract all of them separately

Example
amenity=restaurant
name=Taste of China
cuisine=chinese
addr:housenumber=123
addr:street=Main Street
addr:city=Toronto
addr:province=Ontario
addr:country=CA
addr:postcode=A1A 1A1
phone=+1-555-555-5555
website=https://example.com/
opening_hours=Mo-Su 11:00-23:00

shop=supermarket
name=Fresh Market

Output requirements:
- Return only key=value lines.
- No markdown, bullets, numbering, or commentary.
- Format for phone: +1-555-555-5555 for North American numbers, or +44 20 1234 5678 for international numbers.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract deduplicated OSM business tags from an image file or all images in a directory using OpenRouter."
    )
    parser.add_argument(
        "image_input",
        type=Path,
        help="Image file path or directory containing photos",
    )
    parser.add_argument(
        "--model",
        default="google/gemma-4-31b-it",
        help="OpenRouter model slug (default: google/gemma-4-31b-it)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENROUTER_API_KEY", ""),
        help="OpenRouter API key (default: OPENROUTER_API_KEY env var)",
    )
    return parser.parse_args()


def find_images(image_input: Path) -> list[Path]:
    if not image_input.exists():
        raise ValueError(f"Path does not exist: {image_input}")

    if image_input.is_file():
        if image_input.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image file type: {image_input}")
        return [image_input]

    if image_input.is_dir():
        files = [
            path
            for path in sorted(image_input.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not files:
            raise ValueError(f"No supported image files found in: {image_input}")
        return files

    raise ValueError(f"Path is neither a file nor directory: {image_input}")


def to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def response_text(response: object) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content

    if isinstance(content, Iterable):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = getattr(item, "text", None)
            if isinstance(text, str):
                chunks.append(text)
        return "\n".join(chunks)

    return str(content)


def exif_taken_timestamp(image: Path) -> float | None:
    try:
        with image.open("rb") as file:
            tags = exifread.process_file(file, details=False)
    except OSError:
        return None

    for tag_name in ("EXIF DateTimeOriginal", "EXIF DateTimeDigitized", "Image DateTime"):
        raw_value = tags.get(tag_name)
        if not raw_value:
            continue

        try:
            parsed = dt.datetime.strptime(str(raw_value), "%Y:%m:%d %H:%M:%S")
            return parsed.timestamp()
        except ValueError:
            continue

    return None


def image_timestamp_for_processing(image: Path) -> tuple[float, str]:
    taken_ts = exif_taken_timestamp(image)
    if taken_ts is not None:
        return (taken_ts, "exif")
    return (image.stat().st_mtime, "filesystem_mtime")


def sort_images_for_processing(images: list[Path]) -> list[tuple[Path, float, str]]:
    image_info: list[tuple[Path, float, str]] = []
    for image in images:
        timestamp, source = image_timestamp_for_processing(image)
        image_info.append((image, timestamp, source))

    # For multi-image runs, process oldest capture first; fall back to mtime then filename.
    return sorted(image_info, key=lambda item: (item[1], item[0].name.casefold()))


def extract_from_image(client: OpenRouter, model: str, image: Path) -> str:
    content = [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": to_data_url(image)}},
    ]
    response = client.chat.send(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0,
    )
    return response_text(response)


def main() -> int:
    load_dotenv()
    args = parse_args()

    if not args.api_key:
        print("Missing API key. Set OPENROUTER_API_KEY or pass --api-key.", file=sys.stderr)
        return 1

    try:
        images = find_images(args.image_input)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    with OpenRouter(api_key=args.api_key) as client:
        if len(images) == 1:
            ts, source = image_timestamp_for_processing(images[0])
            when = dt.datetime.fromtimestamp(ts).isoformat(sep=" ", timespec="seconds")
            print(
                f"[1/1] Processing {images[0]} (time={when}, source={source})"
            )
            print(extract_from_image(client, args.model, images[0]))
            return 0

        image_info = sort_images_for_processing(images)
        print(
            f"Processing {len(image_info)} images sequentially with model {args.model}."
        )

        for index, (image, ts, source) in enumerate(image_info, start=1):
            when = dt.datetime.fromtimestamp(ts).isoformat(sep=" ", timespec="seconds")
            print(
                f"[{index}/{len(image_info)}] Processing {image} (time={when}, source={source})"
            )
            output = extract_from_image(client, args.model, image).strip()
            if output:
                print(output, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
