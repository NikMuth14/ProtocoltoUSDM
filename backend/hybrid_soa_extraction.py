"""Hybrid SOA Extraction Pipeline: PDF → SOA Detection → pdfplumber + Vision → Claude LLM → USDM JSON.

Flow:
  1. Accept a PDF file as input
  2. Detect SOA pages using pdfplumber heuristics
  3. Render SOA pages to images (for Vision cross-check)
  4. Extract table cell grids using pdfplumber native parser (reads PDF vector data — no OCR errors)
  5. Cross-check with Claude Vision to fill any gaps pdfplumber misses
  6. Merge multi-page grids into one coherent table
  7. Feed the merged grid (as structured text) to Claude LLM to:
     - Normalize visit headers
     - Interpret markers (X, P, Xa, Xb, etc.)
     - Output USDM-compliant JSON
  8. Validate and save JSON + README

Usage:
    python hybrid_soa_extraction.py path/to/protocol.pdf
    python hybrid_soa_extraction.py path/to/protocol.pdf --label "MyStudy"
"""

import argparse
import base64
import io
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import boto3
import pdfplumber
from botocore.config import Config as BotoConfig
from dotenv import load_dotenv

# ── Setup ────────────────────────────────────────────────────────────────────
BASEDIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASEDIR, ".env"))

JSON_OUTPUT_DIR = os.path.join(BASEDIR, "json_outputs")
os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

IMAGE_OUTPUT_DIR = os.path.join(BASEDIR, "image_outputs")
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

LLM_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

ALLOWED_MARKERS = {"X", "x", "P", "Xa", "Xb", "Xc", "Xd", "Xe", "✓", "•", "Yes"}

SOA_KEYWORDS = [
    "schedule of assessments", "schedule of events", "schedule of activities",
    "schedule of evaluations", "schedule of procedures",
    "schedule of study procedures", "study procedures table", "time and events",
]

print(f"[STARTUP] AWS key loaded: {'YES' if AWS_ACCESS_KEY_ID else 'NO'}")

# ── AWS Clients ──────────────────────────────────────────────────────────────
_bedrock_client = None


def get_bedrock():
    """Lazy-init AWS Bedrock Runtime client."""
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            config=BotoConfig(
                read_timeout=300,
                connect_timeout=10,
                retries={"max_attempts": 2},
            ),
        )
    return _bedrock_client


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Detect SOA pages in PDF
# ══════════════════════════════════════════════════════════════════════════════

def find_soa_pages(pdf_path: str) -> list[int]:
    """Scan a PDF to identify pages containing SOA tables."""
    soa_pages = set()

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

        for page_num, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").lower()
            tables = page.extract_tables()

            has_keyword = any(kw in text for kw in SOA_KEYWORDS)
            has_table = bool(tables and any(len(t) > 2 for t in tables))

            if has_keyword and has_table:
                soa_pages.add(page_num)
                continue

            if has_table:
                for table in tables:
                    if len(table) < 4:
                        continue
                    table_text = " ".join(
                        str(cell).lower() for row in table for cell in row if cell
                    )
                    soa_indicators = [
                        "screening", "baseline", "follow-up", "follow up",
                        "washout", "period 1", "period 2", "day 1", "day -1",
                        "procedure", "assessment", "visit",
                    ]
                    matches = sum(1 for ind in soa_indicators if ind in table_text)
                    if matches >= 4:
                        soa_pages.add(page_num)
                        break

        # Iterative continuation page detection
        while True:
            continuation_pages = set()
            for pg in sorted(soa_pages):
                next_pg = pg + 1
                if next_pg <= total_pages and next_pg not in soa_pages:
                    next_page = pdf.pages[next_pg - 1]
                    next_text = (next_page.extract_text() or "").lower()
                    next_tables = next_page.extract_tables()
                    has_big_table = next_tables and any(len(t) > 3 for t in next_tables)
                    is_new_section = any(
                        heading in next_text
                        for heading in ["table of contents", "list of tables", "synopsis",
                                        "appendix", "references", "abbreviations"]
                    )
                    if has_big_table and not is_new_section:
                        continuation_pages.add(next_pg)
            if not continuation_pages:
                break
            soa_pages.update(continuation_pages)

        # Trailing page scan
        if soa_pages:
            last_page = max(soa_pages)
            for check_pg in range(last_page + 1, total_pages + 1):
                if check_pg not in soa_pages:
                    check_page = pdf.pages[check_pg - 1]
                    check_text = (check_page.extract_text() or "").lower()
                    check_tables = check_page.extract_tables()
                    is_new_section = any(
                        heading in check_text
                        for heading in ["table of contents", "list of tables", "synopsis",
                                        "appendix", "references", "abbreviations", "signature"]
                    )
                    if is_new_section:
                        break
                    has_table = check_tables and any(len(t) > 1 for t in check_tables)
                    if has_table:
                        soa_pages.add(check_pg)

    result = sorted(soa_pages)
    print(f"  [SOA DETECT] Found SOA content on pages: {result}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Render PDF pages to images
# ══════════════════════════════════════════════════════════════════════════════

def render_pdf_pages(pdf_path: str, page_numbers: list[int],
                     resolution: int = 200, save_dir: str = None) -> list[dict]:
    """Convert PDF pages to PNG byte arrays using pdfplumber."""
    images = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num in page_numbers:
            if page_num < 1 or page_num > len(pdf.pages):
                print(f"  [PDF->IMG] Skipping invalid page {page_num}")
                continue

            page = pdf.pages[page_num - 1]
            page_image = page.to_image(resolution=resolution)

            buf = io.BytesIO()
            page_image.original.save(buf, format="PNG")
            png_bytes = buf.getvalue()

            images.append({
                "page_num": page_num,
                "raw_bytes": png_bytes,
            })
            print(f"  [PDF->IMG] Page {page_num} rendered ({len(png_bytes) // 1024} KB)")

            if save_dir:
                try:
                    img_path = os.path.join(save_dir, f"page_{page_num}.png")
                    with open(img_path, "wb") as f:
                        f.write(png_bytes)
                    print(f"  [PDF->IMG] Saved {img_path}")
                except Exception as e:
                    print(f"  [WARN] Failed to save image for page {page_num}: {e}")

    return images


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: pdfplumber native table extraction + Claude Vision cross-check
# ══════════════════════════════════════════════════════════════════════════════

# pdfplumber table settings tuned for dense SOA tables
_PLUMBER_TABLE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 3,
    "min_words_vertical": 1,
    "min_words_horizontal": 1,
    "intersection_tolerance": 3,
}

# Fallback settings when lines strategy finds nothing (text-based tables)
_PLUMBER_TABLE_SETTINGS_TEXT = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "snap_tolerance": 5,
    "join_tolerance": 5,
    "min_words_vertical": 1,
    "min_words_horizontal": 1,
}


def _clean_cell(val) -> str:
    """Normalize a pdfplumber cell value to a clean string.

    Preserves newlines as '\\n' so callers can detect multi-line merged cells.
    """
    if val is None:
        return ""
    text = str(val).strip()
    # Normalize carriage returns but keep newlines as separators
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\r", "\n", text)
    # Collapse multiple consecutive newlines into one
    text = re.sub(r"\n{2,}", "\n", text)
    # Collapse multiple spaces on a single line
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _expand_merged_rows(table: list[list[str]]) -> list[list[str]]:
    """Handle col0 cells that contain multiple lines of text.

    pdfplumber concatenates newlines within a single cell. This joins them
    with a space so the activity name reads naturally. Vision will see the
    actual image and output the correct name regardless.
    """
    expanded = []
    for row in table:
        if not row:
            expanded.append(row)
            continue
        col0 = row[0]
        if "\n" in col0:
            # Join all lines with a space — correct for both wrapped text and merged cells
            parts = [p.strip() for p in col0.split("\n") if p.strip()]
            combined_name = " ".join(parts)
            expanded.append([combined_name] + row[1:])
        else:
            expanded.append(row)
    return expanded


def pdfplumber_extract_tables(pdf_path: str, page_num: int) -> list[list[list[str]]]:
    """Extract all tables from a single PDF page using pdfplumber.

    Tries line-based detection first; falls back to text-based if nothing found.
    Returns a list of tables, each table is a list of rows of cell strings.
    """
    with pdfplumber.open(pdf_path) as pdf:
        if page_num < 1 or page_num > len(pdf.pages):
            return []
        page = pdf.pages[page_num - 1]

        # Try line-based first
        raw_tables = page.extract_tables(_PLUMBER_TABLE_SETTINGS)
        if not raw_tables or all(len(t) < 2 for t in raw_tables):
            # Fallback to text-based
            raw_tables = page.extract_tables(_PLUMBER_TABLE_SETTINGS_TEXT)

        tables = []
        for raw in raw_tables:
            if not raw:
                continue
            # Clean cells (preserving newlines in col0 for merged-row detection)
            cleaned = [[_clean_cell(cell) for cell in row] for row in raw]
            # Expand merged col0 cells back into individual rows
            cleaned = _expand_merged_rows(cleaned)
            # Drop completely empty rows
            cleaned = [row for row in cleaned if any(c for c in row)]
            if cleaned:
                tables.append(cleaned)

        return tables


VISION_CROSSCHECK_PROMPT = """You are given a page image from a clinical trial protocol's Schedule of Assessments (SOA) table.

A pdfplumber parser has already extracted the table below. Your job is to produce the COMPLETE, VERBATIM table from the image.

STRICT RULES — violations are not acceptable:
1. COUNT every visible row in the image. Your output MUST have exactly that many rows. Do NOT skip any row.
2. COUNT every visible column in the image. Every row in your output MUST have exactly that many columns.
3. Column 0 of each row = the FULL activity/procedure name exactly as printed.
   - If a cell's text wraps across multiple lines but is ONE activity (e.g. "CT Scan (if not within last year and patient passes all other screens)"), join the lines with a space into ONE name.
   - If a cell is a VERTICALLY MERGED cell containing MULTIPLE DISTINCT activity names (e.g. "Study drug record", "Medications dispensed", "Medications returned" stacked in one cell with shared borders), output them as ONE row with the names joined by " / " (e.g. "Study drug record / Medications dispensed / Medications returned"). Do NOT split them into separate rows.
   - NEVER truncate or shorten activity names.
4. Other cells = the marker exactly as printed ("X", "P", "Xa", "Xb", "Xc", etc.) or empty string "" if the cell is blank.
5. Row 0 = header row with visit/column labels exactly as printed.
6. Sub-header rows (e.g. a WEEK row under the VISIT row) must be included as their own row.
7. Do NOT omit rows with no markers — they are still activities.
8. Do NOT add any explanation, commentary, or markdown. Output ONLY the raw JSON array of arrays.

pdfplumber extracted table (use as a starting reference — correct and complete it from the image):
{plumber_grid_text}
"""


def vision_crosscheck(image_bytes: bytes, plumber_grid: list[list[str]]) -> list[list[str]]:
    """Send page image + pdfplumber grid to Claude Vision for gap-filling.

    Returns the corrected/completed grid.
    """
    client = get_bedrock()

    # Format the pdfplumber grid as readable text for the prompt
    if plumber_grid:
        grid_lines = []
        for r, row in enumerate(plumber_grid):
            grid_lines.append(f"Row {r}: " + " | ".join(f"[{c}] {cell}" for c, cell in enumerate(row)))
        plumber_grid_text = "\n".join(grid_lines)
    else:
        plumber_grid_text = "(pdfplumber found no table on this page)"

    prompt_text = VISION_CROSSCHECK_PROMPT.format(plumber_grid_text=plumber_grid_text)

    img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 16000,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64,
                    },
                },
                {"type": "text", "text": prompt_text},
            ],
        }],
    })

    response = client.invoke_model(
        modelId=LLM_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(response["body"].read())
    raw_text = result["content"][0]["text"].strip()

    # Strip markdown fences
    raw_text = re.sub(r'^```(?:json)?\s*', '', raw_text)
    raw_text = re.sub(r'\s*```$', '', raw_text)

    try:
        corrected = json.loads(raw_text)
        if isinstance(corrected, list) and corrected:
            # Ensure all cells are strings
            corrected = [[str(cell) if cell is not None else "" for cell in row] for row in corrected]
            return corrected
    except json.JSONDecodeError:
        pass

    # If Vision parse fails, return the original pdfplumber grid unchanged
    print("  [VISION] Cross-check parse failed — keeping pdfplumber grid")
    return plumber_grid


def extract_all_tables_from_pages(pdf_path: str, images: list[dict]) -> list[dict]:
    """Extract tables from each SOA page using pdfplumber + Vision cross-check.

    Returns:
        list of dicts: [{"page_num": int, "tables": [grid, ...]}]
    """
    results = []
    for img in images:
        page_num = img["page_num"]
        raw_bytes = img["raw_bytes"]

        print(f"  [PLUMBER] Extracting tables from page {page_num}...")
        tables = pdfplumber_extract_tables(pdf_path, page_num)
        print(f"  [PLUMBER] Page {page_num}: found {len(tables)} table(s), "
              f"sizes: {[f'{len(t)}x{len(t[0]) if t else 0}' for t in tables]}")

        # Pick the largest table as the SOA candidate
        best_table = max(tables, key=lambda t: len(t) * len(t[0]) if t and t[0] else 0) if tables else []

        print(f"  [VISION] Cross-checking page {page_num} with Claude Vision...")
        corrected = vision_crosscheck(raw_bytes, best_table)

        if corrected and len(corrected) > len(best_table):
            print(f"  [VISION] Vision added {len(corrected) - len(best_table)} row(s) vs pdfplumber")
        elif corrected and corrected != best_table:
            print(f"  [VISION] Vision corrected cell values on page {page_num}")
        else:
            print(f"  [VISION] Vision confirmed pdfplumber grid on page {page_num}")

        final_table = corrected if corrected else best_table
        results.append({"page_num": page_num, "tables": [final_table] if final_table else []})

    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Merge multi-page tables into one coherent grid
# ══════════════════════════════════════════════════════════════════════════════

def merge_table_grids(page_results: list[dict]) -> list[list[str]]:
    """Merge tables from multiple pages into a single grid.

    Heuristic: if the first row of a subsequent page looks like a header
    (matches the first page's header), skip it. Otherwise append all rows.
    """
    if not page_results:
        return []

    # Pick the largest table from each page
    page_tables = []
    for pr in page_results:
        if pr["tables"]:
            # Take the table with the most rows
            biggest = max(pr["tables"], key=lambda t: len(t))
            page_tables.append({"page_num": pr["page_num"], "grid": biggest})

    if not page_tables:
        return []

    # Start with the first page's table
    merged = list(page_tables[0]["grid"])
    header_row = page_tables[0]["grid"][0] if page_tables[0]["grid"] else []

    for pt in page_tables[1:]:
        grid = pt["grid"]
        if not grid:
            continue

        # Check if first row is a repeated header
        first_row = grid[0]
        if _rows_similar(header_row, first_row):
            # Skip the header row, append the rest
            print(f"  [MERGE] Page {pt['page_num']}: skipping repeated header row")
            merged.extend(grid[1:])
        else:
            merged.extend(grid)

    print(f"  [MERGE] Final merged grid: {len(merged)} rows x {len(merged[0]) if merged else 0} cols")
    return merged


def _rows_similar(row_a: list[str], row_b: list[str], threshold: float = 0.6) -> bool:
    """Check if two rows are similar enough to be considered the same header."""
    if not row_a or not row_b:
        return False
    min_len = min(len(row_a), len(row_b))
    if min_len == 0:
        return False
    matches = sum(
        1 for i in range(min_len)
        if row_a[i].strip().lower() == row_b[i].strip().lower() and row_a[i].strip()
    )
    non_empty = sum(1 for i in range(min_len) if row_a[i].strip() or row_b[i].strip())
    if non_empty == 0:
        return False
    return (matches / non_empty) >= threshold


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Validate OCR grid
# ══════════════════════════════════════════════════════════════════════════════

def validate_grid(grid: list[list[str]]) -> dict:
    """Validate the OCR-extracted grid and report statistics."""
    if not grid:
        return {"valid": False, "error": "Empty grid"}

    num_rows = len(grid)
    num_cols = len(grid[0]) if grid else 0

    # Count markers in the body (skip header row)
    marker_counts = {}
    unknown_markers = set()
    empty_cells = 0
    total_body_cells = 0

    # Sub-header row patterns to skip (WEEK/VISIT label rows, repeated column headers)
    _SUBHEADER_PATTERNS = re.compile(
        r'^(week|visit|activity|procedure|assessment|day|period|epoch)$', re.IGNORECASE
    )

    for r in range(1, num_rows):  # skip header
        row = grid[r]
        # Skip sub-header rows: col0 is empty or a structural label, AND col1 is a timing/visit label
        col0 = row[0].strip() if row else ""
        col1 = row[1].strip() if len(row) > 1 else ""
        if not col0 or _SUBHEADER_PATTERNS.match(col0) or _SUBHEADER_PATTERNS.match(col1):
            continue

        for c in range(1, num_cols):  # skip first col (activity names)
            if c >= len(row):
                continue
            cell = row[c].strip()
            total_body_cells += 1
            if not cell:
                empty_cells += 1
            elif cell in ALLOWED_MARKERS:
                marker_counts[cell] = marker_counts.get(cell, 0) + 1
            else:
                # Check if it's a known marker with extra whitespace/case
                normalized = cell.strip()
                if normalized.upper() in {m.upper() for m in ALLOWED_MARKERS}:
                    marker_counts[normalized] = marker_counts.get(normalized, 0) + 1
                else:
                    unknown_markers.add(cell)

    return {
        "valid": True,
        "rows": num_rows,
        "cols": num_cols,
        "body_cells": total_body_cells,
        "empty_cells": empty_cells,
        "marker_counts": marker_counts,
        "unknown_markers": list(unknown_markers),
        "header_row": grid[0] if grid else [],
        "activity_column": [grid[r][0] for r in range(num_rows)] if grid else [],
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Format grid as structured text for LLM
# ══════════════════════════════════════════════════════════════════════════════

def grid_to_text(grid: list[list[str]]) -> str:
    """Convert a 2D grid to a structured text representation for the LLM.

    Format: pipe-delimited rows with row/col indices for clarity.
    """
    if not grid:
        return "(empty grid)"

    lines = []
    lines.append(f"TABLE DIMENSIONS: {len(grid)} rows x {len(grid[0])} columns")
    lines.append("")

    # Header row
    lines.append("HEADER ROW (row 0):")
    lines.append(" | ".join(f"[col{c}] {grid[0][c]}" for c in range(len(grid[0]))))
    lines.append("")

    # Body rows
    lines.append("DATA ROWS:")
    for r in range(1, len(grid)):
        row = grid[r]
        activity_name = row[0] if row else ""
        cells = []
        for c in range(len(row)):
            cell_val = row[c].strip()
            cells.append(f"[col{c}] {cell_val if cell_val else '(empty)'}")
        lines.append(f"  Row {r}: {' | '.join(cells)}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Claude LLM — normalize OCR grid → USDM JSON
# ══════════════════════════════════════════════════════════════════════════════

HYBRID_LLM_SYSTEM_PROMPT = """You are an expert clinical protocol analyst. You will receive a structured OCR-extracted table grid from a Schedule of Assessments (SOA) / Schedule of Events (SOE) from a clinical trial protocol.

The grid has been extracted by pdfplumber (native PDF vector parser) and cross-checked by Claude Vision. Your job is to:
1. Interpret the header row to identify visit/encounter columns and their timing (Week -2, Day 1, Week 2, etc.)
2. Interpret each body row as an activity/procedure
3. For each cell, determine if the activity is scheduled at that visit:
   - "X", "x", "✓", "•" = scheduled
   - "P" = partially scheduled or conditional
   - "Xa", "Xb", "Xc" etc. = scheduled with footnote reference
   - Empty or blank = not scheduled
4. Normalize visit headers into consistent labels
5. Detect activity categories/grouping (bold section headers)
6. Output the result as USDM-compliant JSON

Output ONLY valid JSON — no markdown, no explanation, no code fences.

The JSON structure MUST follow this USDM format exactly:
{
  "studyDesign": {
    "id": "StudyDesign_1",
    "name": "<protocol/study identifier>",
    "label": "<SOA table title>",
    "description": "<brief description>",
    "encounters": [
      {
        "id": "Encounter_<N>",
        "extensionAttributes": [],
        "name": "<short code, e.g. V1, V2>",
        "label": "<visit label, e.g. Visit 1, Visit 2>",
        "description": "<timing, e.g. Week -2, Day 1>",
        "type": {
          "id": "Code_<N>",
          "extensionAttributes": [],
          "code": "C25716",
          "codeSystem": "http://www.cdisc.org",
          "codeSystemVersion": "2024-09-27",
          "decode": "Visit",
          "instanceType": "Code"
        },
        "previousId": "<id of previous Encounter or null>",
        "nextId": "<id of next Encounter or null>",
        "scheduledAtId": "<Timing_N or null>",
        "environmentalSettings": [
          {
            "id": "Code_<N>",
            "extensionAttributes": [],
            "code": "C51282",
            "codeSystem": "http://www.cdisc.org",
            "codeSystemVersion": "2024-09-27",
            "decode": "Clinic",
            "instanceType": "Code"
          }
        ],
        "contactModes": [
          {
            "id": "Code_<N>",
            "extensionAttributes": [],
            "code": "C175574",
            "codeSystem": "http://www.cdisc.org",
            "codeSystemVersion": "2024-09-27",
            "decode": "In Person",
            "instanceType": "Code"
          }
        ],
        "transitionStartRule": null,
        "transitionEndRule": null,
        "notes": [],
        "instanceType": "Encounter"
      }
    ],
    "activities": [
      {
        "id": "Activity_<N>",
        "extensionAttributes": [],
        "name": "<exact activity name from OCR>",
        "label": "<activity label>",
        "description": "<footnote or empty string>",
        "previousId": "<previous Activity id or null>",
        "nextId": "<next Activity id or null>",
        "childIds": [],
        "definedProcedures": [],
        "biomedicalConceptIds": [],
        "bcCategoryIds": [],
        "bcSurrogateIds": [],
        "timelineId": null,
        "notes": [],
        "instanceType": "Activity"
      }
    ],
    "epochs": [
      {
        "id": "StudyEpoch_<N>",
        "extensionAttributes": [],
        "name": "<epoch name>",
        "label": "<epoch label>",
        "description": "<epoch description>",
        "type": {
          "id": "Code_<N>",
          "extensionAttributes": [],
          "code": "<C202487 for Screening, C101526 for Treatment, C202578 for Follow-Up>",
          "codeSystem": "http://www.cdisc.org",
          "codeSystemVersion": "2024-09-27",
          "decode": "<Screening Epoch, Treatment Epoch, or Follow-Up Epoch>",
          "instanceType": "Code"
        },
        "previousId": "<previous epoch id or null>",
        "nextId": "<next epoch id or null>",
        "notes": [],
        "instanceType": "StudyEpoch"
      }
    ],
    "scheduleTimelines": [
      {
        "id": "ScheduleTimeline_1",
        "extensionAttributes": [],
        "name": "Main Timeline",
        "label": "Main Timeline",
        "description": "Main schedule timeline for the study design.",
        "mainTimeline": true,
        "entryCondition": "Potential subject identified",
        "entryId": "<id of first ScheduledActivityInstance>",
        "exits": [
          {
            "id": "ScheduleTimelineExit_1",
            "extensionAttributes": [],
            "instanceType": "ScheduleTimelineExit"
          }
        ],
        "timings": [
          {
            "id": "Timing_<N>",
            "extensionAttributes": [],
            "name": "TIM<N>",
            "label": "<timing label>",
            "description": "<timing description>",
            "type": {
              "id": "Code_<N>",
              "extensionAttributes": [],
              "code": "<C201357 for Before, C201356 for After>",
              "codeSystem": "http://www.cdisc.org",
              "codeSystemVersion": "2024-09-27",
              "decode": "<Before or After>",
              "instanceType": "Code"
            },
            "value": "<ISO 8601 duration, e.g. P2W, P14D>",
            "valueLabel": "<human-readable, e.g. 2 weeks>",
            "relativeToFrom": {
              "id": "Code_<N>",
              "extensionAttributes": [],
              "code": "C201355",
              "codeSystem": "http://www.cdisc.org",
              "codeSystemVersion": "2024-09-27",
              "decode": "Start to Start",
              "instanceType": "Code"
            },
            "relativeFromScheduledInstanceId": "<from instance>",
            "relativeToScheduledInstanceId": "<to instance>",
            "windowLower": null,
            "windowUpper": null,
            "windowLabel": "",
            "instanceType": "Timing"
          }
        ],
        "instances": [
          {
            "id": "ScheduledActivityInstance_<N>",
            "extensionAttributes": [],
            "name": "<short code>",
            "label": "<label>",
            "description": "-",
            "defaultConditionId": "<next instance id or null>",
            "epochId": "<StudyEpoch id>",
            "instanceType": "ScheduledActivityInstance",
            "timelineId": null,
            "timelineExitId": "<exit id if last, else null>",
            "activityIds": [
              "<actual activity names that are marked at this visit>"
            ],
            "encounterId": "<Encounter id>"
          }
        ],
        "plannedDuration": null,
        "instanceType": "ScheduleTimeline"
      }
    ]
  },
  "footnotes": [
    "<verbatim footnote text>"
  ]
}

CRITICAL RULES — ALL ARE MANDATORY, NO EXCEPTIONS:
1. ZERO TRUNCATION: Every activity name in the `activities` array and in every `activityIds` list MUST be the FULL verbatim name from column 0 of the grid. Never shorten, abbreviate, or paraphrase. If the name is "CT Scan (if not within last year and patient passes all other screens)" that is the exact string to use everywhere.
2. ZERO OMISSION: Every data row in the grid becomes an Activity. Even rows with no markers (e.g. "Medications dispensed", "Medications returned") MUST appear in the activities array. They simply won't appear in any activityIds.
3. EVERY VISIT COLUMN: For each visit/encounter column, scan EVERY row. If the cell contains ANY mark (X, x, P, Xa, Xb, Xc, Xd, Xe, ✓, •, Yes, or any non-empty value), include that activity's FULL name in that visit's activityIds.
4. activityIds must contain the actual activity FULL NAMES (not Activity_N IDs, not shortened names).
5. ACTIVITY COUNT: The number of entries in the `activities` array must equal the number of data rows in the grid (excluding header rows and sub-header rows like WEEK/VISIT rows). If the grid has 30 data rows, you must output 30 activities.
6. ENCOUNTER COUNT: The number of entries in `encounters` must equal the number of visit columns in the grid (excluding column 0 which is activity names, and excluding sub-header columns).
7. If a row appears to be a category/section header (no marks in any visit column, text looks like a grouping label such as "Laboratory Tests", "Efficacy Assessments"), model it as a grouping activity with childIds listing the activities that follow it in that group. Grouping activities are NOT included in activityIds — only leaf activities.
8. Encounters linked via previousId/nextId. Activities linked via previousId/nextId.
9. Infer epochs from column groupings (Screening, Treatment, Follow-Up).
10. Use ISO 8601 durations for timings.
11. Output raw JSON only. No markdown, no code fences, no explanation."""


def call_llm_for_usdm(grid_text: str, validation_info: dict) -> dict:
    """Send the OCR grid text to Claude LLM for normalization into USDM JSON."""
    client = get_bedrock()

    num_rows = validation_info.get('rows', '?')
    num_cols = validation_info.get('cols', '?')
    marker_counts = validation_info.get('marker_counts', {})
    total_markers = sum(marker_counts.values()) if marker_counts else '?'

    user_message = (
        "Below is the COMPLETE table grid from a clinical trial protocol's "
        "Schedule of Assessments (SOA). It was extracted by pdfplumber (native PDF parser) "
        "and cross-checked by Claude Vision.\n\n"
        f"GRID DIMENSIONS: {num_rows} rows x {num_cols} columns\n"
        f"TOTAL MARKERS IN GRID: {total_markers} (X/P/Xa/Xb etc.)\n"
        f"MARKER BREAKDOWN: {json.dumps(marker_counts)}\n\n"
        "MANDATORY COMPLETENESS CHECKS before outputting JSON:\n"
        f"  1. Your `activities` array MUST contain exactly {num_rows} entries "
        f"(one per data row, excluding header/sub-header rows).\n"
        f"  2. Your `encounters` array MUST contain exactly {num_cols - 1} entries "
        f"(one per visit column, excluding column 0).\n"
        f"  3. The total number of activity name entries across ALL activityIds lists "
        f"MUST equal {total_markers} (matching the marker count above).\n"
        "  4. Every activity name in activityIds MUST be the FULL verbatim name from column 0 — "
        "no truncation, no shortening.\n\n"
        f"OCR TABLE GRID:\n"
        f"{'=' * 80}\n"
        f"{grid_text}\n"
        f"{'=' * 80}\n\n"
        "Convert this grid into the USDM JSON format as specified in your instructions. "
        "Do not omit any activity, any visit, or any marker."
    )

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 64000,
        "system": HYBRID_LLM_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_message}],
    })

    print(f"\n[LLM] Sending OCR grid ({validation_info.get('rows', '?')} rows x "
          f"{validation_info.get('cols', '?')} cols) to Claude...")

    response = client.invoke_model(
        modelId=LLM_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )

    result = json.loads(response["body"].read())
    raw_text = result["content"][0]["text"]

    # Strip markdown fences if present
    json_clean = re.sub(r'^```(?:json)?\s*', '', raw_text.strip())
    json_clean = re.sub(r'\s*```$', '', json_clean)

    try:
        soa_json = json.loads(json_clean)
        print("[LLM] JSON parsed successfully.")
        return soa_json
    except json.JSONDecodeError as e:
        print(f"[LLM] JSON parse failed: {e}")
        print(f"[LLM] Raw response (first 500 chars):\n{raw_text[:500]}")
        return {"error": str(e), "raw_response": raw_text}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Generate README validation summary (longform)
# ══════════════════════════════════════════════════════════════════════════════

def generate_readme_summary(soa_json: dict, json_path: str) -> str:
    """Generate a longform markdown README from the USDM JSON."""
    sd = soa_json.get("studyDesign", {})
    encounters = sd.get("encounters", [])
    activities = sd.get("activities", [])
    timelines = sd.get("scheduleTimelines", [])
    epochs = sd.get("epochs", [])
    footnotes = soa_json.get("footnotes", [])

    epoch_lookup = {ep["id"]: ep for ep in epochs}

    enc_to_instance = {}
    for tl in timelines:
        for inst in tl.get("instances", []):
            enc_to_instance[inst.get("encounterId", "")] = inst

    all_activity_names = []
    seen = set()
    for act in activities:
        if not act.get("childIds") and act.get("name") and act["name"] not in seen:
            all_activity_names.append(act["name"])
            seen.add(act["name"])

    child_to_category = {}
    for act in activities:
        if act.get("childIds"):
            cat_name = act.get("name", "")
            for cid in act["childIds"]:
                child = next((a for a in activities if a["id"] == cid), None)
                if child:
                    child_to_category[child.get("name", cid)] = cat_name

    epoch_to_encounters = {}
    for enc in encounters:
        inst = enc_to_instance.get(enc["id"], {})
        epoch_to_encounters.setdefault(inst.get("epochId", ""), []).append(enc)

    lines = []
    study_name = sd.get("name", sd.get("label", "Unknown Study"))
    lines.append("# SOA Validation Summary (Hybrid OCR→LLM)")
    lines.append(f"## {study_name}")
    lines.append("")
    lines.append(f"**Source file:** `{os.path.basename(json_path) if json_path else 'N/A'}`  ")
    lines.append(f"**Pipeline:** AWS Textract OCR → Claude LLM → USDM JSON  ")
    lines.append(f"**Total Visits/Encounters:** {len(encounters)}  ")
    lines.append(f"**Total Activities:** {len(all_activity_names)} leaf activities across {len(activities)} total  ")
    lines.append(f"**Epochs:** {len(epochs)}  ")
    lines.append("")
    lines.append("---")
    lines.append("")

    if epochs:
        lines.append("## Study Epochs")
        lines.append("")
        for ep in epochs:
            enc_count = len(epoch_to_encounters.get(ep["id"], []))
            lines.append(f"### {ep.get('label', ep.get('name', ''))}")
            lines.append(f"- **Name:** {ep.get('name', '')}")
            lines.append(f"- **Description:** {ep.get('description', '-')}")
            lines.append(f"- **Visits in this epoch:** {enc_count}")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Visit-by-Visit Activity Listing")
    lines.append("")

    for enc in encounters:
        inst = enc_to_instance.get(enc["id"], {})
        epoch = epoch_lookup.get(inst.get("epochId", ""), {})
        epoch_label = epoch.get("label", epoch.get("name", "Unknown Epoch"))
        enc_label = enc.get("label", enc.get("name", ""))
        enc_desc = enc.get("description", "")
        act_names = inst.get("activityIds", [])

        heading = f"### {enc_label}"
        if enc_desc and enc_desc != "-":
            heading += f" — {enc_desc}"
        lines.append(heading)
        lines.append(f"**Epoch:** {epoch_label}  ")
        lines.append(f"**Activities scheduled ({len(act_names)}):**  ")
        lines.append("")

        if act_names:
            categorised = {}
            uncategorised = []
            for name in act_names:
                cat = child_to_category.get(name)
                if cat:
                    categorised.setdefault(cat, []).append(name)
                else:
                    uncategorised.append(name)

            if categorised:
                for cat_name, cat_acts in categorised.items():
                    lines.append(f"**{cat_name}**")
                    for a in cat_acts:
                        lines.append(f"- {a}")
                    lines.append("")
                if uncategorised:
                    lines.append("**Other**")
                    for a in uncategorised:
                        lines.append(f"- {a}")
                    lines.append("")
            else:
                for a in act_names:
                    lines.append(f"- {a}")
                lines.append("")
        else:
            lines.append("_No activities recorded for this visit._")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## All Activities Reference")
    lines.append("")
    if child_to_category:
        current_cat = None
        for name in all_activity_names:
            cat = child_to_category.get(name, "Uncategorised")
            if cat != current_cat:
                lines.append(f"### {cat}")
                current_cat = cat
            lines.append(f"- {name}")
        lines.append("")
    else:
        for name in all_activity_names:
            lines.append(f"- {name}")
        lines.append("")

    if footnotes:
        lines.append("---")
        lines.append("")
        lines.append("## Footnotes")
        lines.append("")
        for i, fn in enumerate(footnotes, 1):
            lines.append(f"{i}. {fn}")
        lines.append("")

    readme_content = "\n".join(lines)
    readme_path = json_path.replace(".json", "_README.md") if json_path else os.path.join(JSON_OUTPUT_DIR, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    return readme_path


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: Save outputs
# ══════════════════════════════════════════════════════════════════════════════

def save_outputs(soa_json: dict, label: str, grid: list[list[str]] = None) -> tuple[str, str]:
    """Save JSON, README, and optionally the raw OCR grid."""
    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = re.sub(r'[^\w\-]', '_', label)

    # Save JSON
    json_filename = f"hybrid_{safe_label}_{timestamp}.json"
    json_path = os.path.join(JSON_OUTPUT_DIR, json_filename)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(soa_json, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVE] JSON   → {json_path}")

    # Save README
    readme_path = generate_readme_summary(soa_json, json_path)
    print(f"[SAVE] README → {readme_path}")

    # Save raw OCR grid for debugging
    if grid:
        grid_filename = f"hybrid_{safe_label}_{timestamp}_ocr_grid.txt"
        grid_path = os.path.join(JSON_OUTPUT_DIR, grid_filename)
        with open(grid_path, "w", encoding="utf-8") as f:
            f.write(grid_to_text(grid))
        print(f"[SAVE] GRID   → {grid_path}")

    return json_path, readme_path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN: Full hybrid pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_hybrid_pipeline(pdf_path: str, label: str = None):
    """Execute the full hybrid pipeline: PDF → SOA detect → images → pdfplumber+Vision → LLM → USDM JSON."""

    pdf_path = os.path.abspath(pdf_path)
    if not os.path.isfile(pdf_path):
        print(f"[ERROR] PDF file not found: {pdf_path}")
        sys.exit(1)

    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
    label = label or pdf_basename

    print(f"\n{'#' * 60}")
    print(f"  HYBRID SOA EXTRACTION PIPELINE")
    print(f"  PDF  : {pdf_path}")
    print(f"  Label: {label}")
    print(f"{'#' * 60}")

    # ── Step 1: Detect SOA pages ──
    print(f"\n{'=' * 60}")
    print("STEP 1: Detecting SOA pages in PDF")
    print(f"{'=' * 60}")
    soa_pages = find_soa_pages(pdf_path)
    if not soa_pages:
        print("[ERROR] No SOA pages detected in this PDF.")
        sys.exit(1)
    print(f"  Found {len(soa_pages)} SOA page(s): {soa_pages}")

    # ── Step 2: Render pages to images ──
    print(f"\n{'=' * 60}")
    print("STEP 2: Rendering SOA pages to images")
    print(f"{'=' * 60}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_save_dir = os.path.join(IMAGE_OUTPUT_DIR, f"hybrid_{pdf_basename}_{timestamp}")
    os.makedirs(image_save_dir, exist_ok=True)

    images = render_pdf_pages(pdf_path, soa_pages, resolution=200, save_dir=image_save_dir)
    if not images:
        print("[ERROR] Failed to render PDF pages to images.")
        sys.exit(1)
    print(f"  Rendered {len(images)} image(s). Saved to: {image_save_dir}")

    # ── Step 3: pdfplumber + Vision extraction ──
    print(f"\n{'=' * 60}")
    print("STEP 3: pdfplumber native extraction + Claude Vision cross-check")
    print(f"{'=' * 60}")
    page_results = extract_all_tables_from_pages(pdf_path, images)

    # ── Step 4: Merge grids ──
    print(f"\n{'=' * 60}")
    print("STEP 4: Merging multi-page tables")
    print(f"{'=' * 60}")
    merged_grid = merge_table_grids(page_results)

    if not merged_grid:
        print("[ERROR] No tables found in any page. Exiting.")
        sys.exit(1)

    # ── Step 5: Validate ──
    print(f"\n{'=' * 60}")
    print("STEP 5: Validating OCR grid")
    print(f"{'=' * 60}")
    validation = validate_grid(merged_grid)
    print(f"  Rows: {validation['rows']}")
    print(f"  Cols: {validation['cols']}")
    print(f"  Body cells: {validation['body_cells']}")
    print(f"  Empty cells: {validation['empty_cells']}")
    print(f"  Marker counts: {validation['marker_counts']}")
    if validation['unknown_markers']:
        print(f"  ⚠ Unknown markers: {validation['unknown_markers']}")

    # ── Step 6: Format grid as text ──
    grid_text = grid_to_text(merged_grid)

    # ── Step 7: LLM normalization ──
    print(f"\n{'=' * 60}")
    print("STEP 6: Claude LLM — normalizing OCR grid → USDM JSON")
    print(f"{'=' * 60}")
    soa_json = call_llm_for_usdm(grid_text, validation)

    if "error" in soa_json:
        print(f"\n[ERROR] LLM extraction failed: {soa_json['error']}")
        sys.exit(1)

    # ── Summary ──
    sd = soa_json.get("studyDesign", {})
    encounters = sd.get("encounters", [])
    activities = sd.get("activities", [])
    timelines = sd.get("scheduleTimelines", [])
    all_instances = [inst for tl in timelines for inst in tl.get("instances", [])]
    print(f"\n{'=' * 60}")
    print("EXTRACTION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Study      : {sd.get('name', 'N/A')}")
    print(f"  Encounters : {len(encounters)}")
    print(f"  Activities : {len(activities)}")
    print(f"  Instances  : {len(all_instances)}")
    print(f"  Epochs     : {len(sd.get('epochs', []))}")
    print(f"  Footnotes  : {len(soa_json.get('footnotes', []))}")

    # ── Step 8: Save ──
    print(f"\n{'=' * 60}")
    print("STEP 7: Saving outputs")
    print(f"{'=' * 60}")
    json_path, readme_path = save_outputs(soa_json, label, merged_grid)

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")
    print(f"  JSON   : {json_path}")
    print(f"  README : {readme_path}")
    print(f"  Images : {image_save_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid SOA Extraction: PDF → Textract OCR → Claude LLM → USDM JSON"
    )
    parser.add_argument(
        "pdf",
        help="Path to the clinical trial protocol PDF file.",
    )
    parser.add_argument(
        "--label", "-l",
        default=None,
        help="Label for the output filename (defaults to PDF filename).",
    )
    args = parser.parse_args()

    # Resolve relative paths from backend dir
    pdf_path = args.pdf
    if not Path(pdf_path).is_absolute():
        pdf_path = str(Path(__file__).parent / pdf_path)

    run_hybrid_pipeline(pdf_path, args.label)


if __name__ == "__main__":
    main()
