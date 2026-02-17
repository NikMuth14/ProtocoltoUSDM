"""Standalone test script for Claude Vision SOA extraction.

Usage:
    python test_vision_extraction.py
    python test_vision_extraction.py --folder image_outputs/CDISC_Pilot_Study_20260217_145329
    python test_vision_extraction.py --images image_outputs/CDISC_Pilot_Study_20260217_145329/page_53.png image_outputs/CDISC_Pilot_Study_20260217_145329/page_54.png

Input is taken from image_outputs/ — no PDF required.
Output JSON and README are saved to json_outputs/ with a test_ prefix.
"""

import argparse
import base64
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Allow importing from app.py in the same directory
sys.path.insert(0, os.path.dirname(__file__))
from app import (
    IMAGE_SOA_SYSTEM_PROMPT,
    JSON_OUTPUT_DIR,
    IMAGE_OUTPUT_DIR,
    LLM_MODEL_ID,
    get_bedrock,
    _generate_readme_summary,
)


def load_images_from_folder(folder_path: str) -> list[dict]:
    """Load all PNG images from a folder as base64 dicts, sorted by page number."""
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    png_files = sorted(folder.glob("*.png"), key=lambda p: _extract_page_num(p.stem))
    if not png_files:
        raise ValueError(f"No PNG files found in: {folder_path}")

    images = []
    for png_path in png_files:
        page_num = _extract_page_num(png_path.stem)
        with open(png_path, "rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode("utf-8")
        images.append({
            "page_num": page_num,
            "base64_data": b64,
            "media_type": "image/png",
            "file_path": str(png_path),
        })
        print(f"  [LOAD] {png_path.name} (page {page_num}, {len(raw) // 1024} KB)")

    return images


def load_images_from_paths(image_paths: list[str]) -> list[dict]:
    """Load specific image files as base64 dicts."""
    images = []
    for path_str in image_paths:
        png_path = Path(path_str)
        if not png_path.exists():
            raise FileNotFoundError(f"Image not found: {path_str}")
        page_num = _extract_page_num(png_path.stem)
        with open(png_path, "rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode("utf-8")
        images.append({
            "page_num": page_num,
            "base64_data": b64,
            "media_type": "image/png",
            "file_path": str(png_path),
        })
        print(f"  [LOAD] {png_path.name} (page {page_num}, {len(raw) // 1024} KB)")
    return images


def _extract_page_num(stem: str) -> int:
    """Extract page number from filename like 'page_53' -> 53."""
    match = re.search(r"(\d+)$", stem)
    return int(match.group(1)) if match else 0


def call_vision_for_soa_test(page_images: list[dict]) -> dict:
    """Send images to Claude Vision and return parsed USDM JSON.
    Mirrors app.py call_vision_for_soa() exactly.
    """
    content_parts = []
    for img in page_images:
        content_parts.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": img["media_type"],
                "data": img["base64_data"],
            },
        })

    page_list = ", ".join(str(img["page_num"]) for img in page_images)
    content_parts.append({
        "type": "text",
        "text": (
            f"These are pages {page_list} from a clinical trial protocol PDF. "
            "They contain the Schedule of Assessments / Schedule of Events / "
            "Schedule of Activities table(s). "
            "Extract the COMPLETE table(s) into the USDM JSON format as specified. "
            "Each visit column becomes an Encounter, each row becomes an Activity, "
            "and the SOA grid mapping (which activities are marked at which visits) "
            "is captured via ScheduledActivityInstance objects in the scheduleTimelines. "
            "Infer epochs from the column groupings. "
            "Preserve every row, every cell value, and every footnote exactly as shown. "
            "If the table spans multiple pages, combine them into one coherent structure. "
            "Do not omit or summarize anything."
        ),
    })

    client = get_bedrock()
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 64000,
        "system": IMAGE_SOA_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": content_parts}],
    })

    print(f"\n[VISION] Sending {len(page_images)} image(s) to Claude Vision (pages: {page_list})...")
    response = client.invoke_model(
        modelId=LLM_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )

    result = json.loads(response["body"].read())
    raw_text = result["content"][0]["text"]

    json_clean = re.sub(r'^```(?:json)?\s*', '', raw_text.strip())
    json_clean = re.sub(r'\s*```$', '', json_clean)

    try:
        soa_json = json.loads(json_clean)
        print("[VISION] JSON parsed successfully.")
        return soa_json
    except json.JSONDecodeError as e:
        print(f"[VISION] JSON parse failed: {e}")
        print(f"[VISION] Raw response (first 500 chars):\n{raw_text[:500]}")
        return {"error": str(e), "raw_response": raw_text}


def save_outputs(soa_json: dict, label: str) -> tuple[str, str]:
    """Save JSON and README to json_outputs/ with a test_ prefix."""
    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = re.sub(r'[^\w\-]', '_', label)
    json_filename = f"test_{safe_label}_{timestamp}.json"
    json_path = os.path.join(JSON_OUTPUT_DIR, json_filename)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(soa_json, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVE] JSON  → {json_path}")

    readme_path = _generate_readme_summary(soa_json, json_path)
    print(f"[SAVE] README → {readme_path}")

    return json_path, readme_path


def list_available_folders() -> list[str]:
    """List all available image output folders."""
    base = Path(IMAGE_OUTPUT_DIR)
    if not base.exists():
        return []
    return sorted([str(f) for f in base.iterdir() if f.is_dir()])


def main():
    parser = argparse.ArgumentParser(
        description="Test Claude Vision SOA extraction from saved page images."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--folder", "-f",
        help="Path to an image_outputs subfolder (all PNGs loaded in order). "
             "Relative to backend/ or absolute.",
    )
    group.add_argument(
        "--images", "-i",
        nargs="+",
        help="One or more specific PNG file paths to send.",
    )
    parser.add_argument(
        "--label", "-l",
        default=None,
        help="Label for the output filename (defaults to folder name).",
    )
    args = parser.parse_args()

    # Resolve folder/image paths relative to backend/ if not absolute
    backend_dir = Path(__file__).parent

    if args.folder:
        folder_path = Path(args.folder)
        if not folder_path.is_absolute():
            folder_path = backend_dir / folder_path
        label = args.label or folder_path.name
        print(f"\n=== Test Vision Extraction ===")
        print(f"Folder : {folder_path}")
        print(f"Label  : {label}\n")
        images = load_images_from_folder(str(folder_path))

    elif args.images:
        resolved = [
            str(backend_dir / p) if not Path(p).is_absolute() else p
            for p in args.images
        ]
        label = args.label or Path(resolved[0]).parent.name
        print(f"\n=== Test Vision Extraction ===")
        print(f"Images : {resolved}")
        print(f"Label  : {label}\n")
        images = load_images_from_paths(resolved)

    else:
        # No args — show available folders and use the most recent one
        folders = list_available_folders()
        if not folders:
            print(f"No image output folders found in: {IMAGE_OUTPUT_DIR}")
            print("Run a PDF extraction first to generate images.")
            sys.exit(1)

        print("\n=== Available image output folders ===")
        for i, f in enumerate(folders):
            png_count = len(list(Path(f).glob("*.png")))
            print(f"  [{i}] {Path(f).name}  ({png_count} pages)")

        print(f"\nNo --folder specified. Using most recent: {Path(folders[-1]).name}")
        folder_path = Path(folders[-1])
        label = args.label or folder_path.name
        images = load_images_from_folder(str(folder_path))

    if not images:
        print("No images loaded. Exiting.")
        sys.exit(1)

    print(f"\nLoaded {len(images)} image(s). Sending to Claude Vision...")

    # Call Vision
    soa_json = call_vision_for_soa_test(images)

    if "error" in soa_json:
        print(f"\n[ERROR] Extraction failed: {soa_json['error']}")
        sys.exit(1)

    # Quick summary
    sd = soa_json.get("studyDesign", {})
    encounters = sd.get("encounters", [])
    activities = sd.get("activities", [])
    timelines = sd.get("scheduleTimelines", [])
    all_instances = [inst for tl in timelines for inst in tl.get("instances", [])]
    print(f"\n=== Extraction Summary ===")
    print(f"  Study      : {sd.get('name', 'N/A')}")
    print(f"  Encounters : {len(encounters)}")
    print(f"  Activities : {len(activities)}")
    print(f"  Instances  : {len(all_instances)}")
    print(f"  Epochs     : {len(sd.get('epochs', []))}")
    print(f"  Footnotes  : {len(soa_json.get('footnotes', []))}")

    # Save outputs
    json_path, readme_path = save_outputs(soa_json, label)
    print(f"\nDone. Open the README for a quick validation:")
    print(f"  {readme_path}")


if __name__ == "__main__":
    main()
