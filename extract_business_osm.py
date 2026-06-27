#!/usr/bin/env python3
"""Extract deduplicated OpenStreetMap business tags from one file or a folder of photos."""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import re
import sys
from pathlib import Path
from typing import Iterable

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
- Deduplicate repeated lines across all photos.
- Do not guess the address or other information if it is not present in the photos.
- Do not make up addr:street if it is not present in the photos.
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


def dedupe_osm_lines(text: str) -> list[str]:
    seen: set[str] = set()
    lines: list[str] = []

    for raw in text.splitlines():
        line = raw.strip().lstrip("-* ").strip()
        if not line or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            continue

        normalized = f"{key.lower()}={re.sub(r'\\s+', ' ', value).strip().lower()}"
        if normalized in seen:
            continue

        seen.add(normalized)
        lines.append(f"{key}={value}")

    return lines


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

    content = [{"type": "text", "text": PROMPT}]
    for image in images:
        content.append({"type": "image_url", "image_url": {"url": to_data_url(image)}})

    with OpenRouter(api_key=args.api_key) as client:
        response = client.chat.send(
            model=args.model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
        )

    print(response_text(response))

    #output_lines = dedupe_osm_lines(response_text(response))
    #if not output_lines:
    #    print("No OSM key=value output detected from model response.", file=sys.stderr)
    #    return 2

    #print("\n".join(output_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
