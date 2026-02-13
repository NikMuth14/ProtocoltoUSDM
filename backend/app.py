import os
import io
import re
import json
import base64
import tempfile
from datetime import datetime
import pdfplumber
import boto3
from botocore.config import Config as BotoConfig
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

BASEDIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASEDIR, ".env"))

JSON_OUTPUT_DIR = os.path.join(BASEDIR, "json_outputs")
os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

IMAGE_OUTPUT_DIR = os.path.join(BASEDIR, "image_outputs")
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(BASEDIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB limit

# ── Config ──────────────────────────────────────────────────────────────────
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

print(f"[STARTUP] AWS key loaded: {'YES' if AWS_ACCESS_KEY_ID else 'NO'}")

LLM_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# ── Clients ─────────────────────────────────────────────────────────────────
bedrock_client = None


def get_bedrock():
    """Lazy-init AWS Bedrock Runtime client with extended timeout for Vision calls."""
    global bedrock_client
    if bedrock_client is None:
        bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            config=BotoConfig(
                read_timeout=300,  # 5 min for large Vision payloads
                connect_timeout=10,
                retries={"max_attempts": 2},
            ),
        )
    return bedrock_client



# ── Routes ──────────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ── Automated SOA Extraction from PDF ──────────────────────────────────────
SOA_KEYWORDS = [
    "schedule of assessments",
    "schedule of events",
    "schedule of activities",
    "schedule of evaluations",
    "schedule of procedures",
    "schedule of study procedures",
    "study procedures table",
    "time and events",
]

IMAGE_SOA_SYSTEM_PROMPT = """You are an expert clinical protocol analyst specializing in extracting Schedule of Assessments (SOA) / Schedule of Events (SOE) tables from clinical trial protocol page images and converting them into CDISC Unified Study Definitions Model (USDM) JSON format.

You will receive one or more page images from a clinical trial protocol PDF. Your task is to extract ALL SOA/SOE table content and produce a USDM-compliant JSON structure.

Output ONLY valid JSON — no markdown, no explanation, no code fences.

The JSON structure MUST follow this USDM format exactly:
{
  "studyDesign": {
    "id": "StudyDesign_1",
    "name": "<protocol/study identifier visible in the document>",
    "label": "<document title or SOA table title>",
    "description": "<brief description of the study design if visible>",
    "encounters": [
      {
        "id": "Encounter_<N>",
        "extensionAttributes": [],
        "name": "<short encounter code, e.g. E1, E2>",
        "label": "<visit label, e.g. Screening 1, Baseline, Week 2>",
        "description": "<description of the encounter, e.g. Day 14>",
        "type": {
          "id": "Code_<N>",
          "extensionAttributes": [],
          "code": "C25716",
          "codeSystem": "http://www.cdisc.org",
          "codeSystemVersion": "2024-09-27",
          "decode": "Visit",
          "instanceType": "Code"
        },
        "previousId": "<id of previous Encounter or null if first>",
        "nextId": "<id of next Encounter or null if last>",
        "scheduledAtId": "<id of the Timing object for this encounter, or null>",
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
        "name": "<category/section header name, e.g. Eligibility, Safety Assessments>",
        "label": "<category label>",
        "description": "<category name> grouping activity",
        "previousId": "<id of previous Activity or null if first>",
        "nextId": "<id of first child Activity>",
        "childIds": [
          "<Activity IDs of all child activities under this category>"
        ],
        "definedProcedures": [],
        "biomedicalConceptIds": [],
        "bcCategoryIds": [],
        "bcSurrogateIds": [],
        "timelineId": null,
        "notes": [],
        "instanceType": "Activity"
      },
      {
        "id": "Activity_<N>",
        "extensionAttributes": [],
        "name": "<exact procedure/activity name from the SOA table row>",
        "label": "<activity label, same as name or more descriptive>",
        "description": "<any footnote or comment associated with this activity, empty string if none>",
        "previousId": "<id of parent grouping Activity if first child, else id of previous sibling Activity>",
        "nextId": "<id of next Activity or null if last>",
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
        "name": "<epoch name, e.g. Screening, Treatment, Follow-Up>",
        "label": "<epoch label>",
        "description": "<epoch description>",
        "type": {
          "id": "Code_<N>",
          "extensionAttributes": [],
          "code": "<CDISC code: C202487 for Screening, C101526 for Treatment, C202578 for Follow-Up>",
          "codeSystem": "http://www.cdisc.org",
          "codeSystemVersion": "2024-09-27",
          "decode": "<Screening Epoch, Treatment Epoch, or Follow-Up Epoch>",
          "instanceType": "Code"
        },
        "previousId": "<id of previous epoch or null>",
        "nextId": "<id of next epoch or null>",
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
        "entryId": "<id of the first ScheduledActivityInstance>",
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
            "label": "<timing label, e.g. Screening, Week 2, Week 4>",
            "description": "<timing description>",
            "type": {
              "id": "Code_<N>",
              "extensionAttributes": [],
              "code": "<C201357 for Before, C201356 for After, C201358 for Fixed Reference>",
              "codeSystem": "http://www.cdisc.org",
              "codeSystemVersion": "2024-09-27",
              "decode": "<Before, After, or Fixed Reference>",
              "instanceType": "Code"
            },
            "value": "<ISO 8601 duration, e.g. P2W, P14D, P4W>",
            "valueLabel": "<human-readable label, e.g. 2 weeks, 14 days>",
            "relativeToFrom": {
              "id": "Code_<N>",
              "extensionAttributes": [],
              "code": "C201355",
              "codeSystem": "http://www.cdisc.org",
              "codeSystemVersion": "2024-09-27",
              "decode": "Start to Start",
              "instanceType": "Code"
            },
            "relativeFromScheduledInstanceId": "<ScheduledActivityInstance this timing is FROM>",
            "relativeToScheduledInstanceId": "<ScheduledActivityInstance this timing is relative TO (usually the dosing/baseline anchor)>",
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
            "name": "<short code, e.g. SCREEN1, DOSE, WK2, WK4>",
            "label": "<label, e.g. Screen One, Dose, Week 2>",
            "description": "-",
            "defaultConditionId": "<id of next ScheduledActivityInstance or null if last>",
            "epochId": "<id of the StudyEpoch this instance belongs to>",
            "instanceType": "ScheduledActivityInstance",
            "timelineId": null,
            "timelineExitId": "<ScheduleTimelineExit id if this is the last instance, else null>",
            "activityIds": [
              "<Activity_N ids for ALL activities/procedures marked (X) at this visit>"
            ],
            "encounterId": "<Encounter_N id for the visit/encounter this instance maps to>"
          }
        ],
        "plannedDuration": null,
        "instanceType": "ScheduleTimeline"
      }
    ]
  },
  "footnotes": [
    "<verbatim footnote text from the SOA table>"
  ]
}

CRITICAL RULES:
- Extract EVERY row (activity/procedure) and EVERY column (visit/encounter) from the SOA table.
- Each visit column becomes an Encounter. Each row becomes an Activity.
- The ScheduledActivityInstance is the SOA grid cell mapping: its activityIds list contains ONLY the Activity IDs that are marked (X or equivalent) for that encounter/visit.
- Encounters must be linked sequentially via previousId/nextId.
- Activities must be linked sequentially via previousId/nextId.
- ACTIVITY CATEGORIES: SOA tables often have bold section headers or category rows (e.g. "Eligibility", "Study Administration", "Safety Assessments", "Laboratory Analyses", "Other") that group multiple child activities beneath them. These MUST be modeled as grouping activities:
  * A grouping activity has a non-empty "childIds" array listing the IDs of all child activities under it.
  * Its "description" should be "<category name> grouping activity".
  * Its "nextId" points to its first child activity.
  * Child activities have empty "childIds": [].
  * The first child's "previousId" points back to the parent grouping activity.
  * The last child's "nextId" points to the next grouping activity (or null if last).
  * Grouping activities are NOT included in ScheduledActivityInstance activityIds — only leaf/child activities appear there.
  * If the SOA table has NO visible section headers or categories, then all activities should have empty childIds.
- Epochs should be inferred from the column groupings (e.g. Screening columns → Screening epoch, Treatment columns → Treatment epoch, Follow-up → Follow-Up epoch).
- Timings should capture the visit timing relative to a baseline/dosing anchor (e.g. Day -14, Day 1, Week 2, Week 4).
- Use ISO 8601 durations for timing values (P2W = 2 weeks, P14D = 14 days, P4W = 4 weeks, etc.).
- Preserve ALL footnotes/comments verbatim in the footnotes array.
- If the table spans multiple pages, combine into one coherent structure.
- All IDs must be unique and sequentially numbered (Encounter_1, Encounter_2, Activity_1, Activity_2, etc.).
- Output raw JSON only. No markdown formatting."""


def find_soa_pages(pdf_path):
    """Scan a PDF to identify pages that contain Schedule of Activities/Events/Assessments tables.

    Uses a two-pass approach:
    - Pass 1: Find pages with SOA keywords that also contain tables.
    - Pass 2: Include immediate continuation pages (next page has a table
      and shares similar column structure).

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        list[int]: 1-indexed page numbers containing SOA tables.
    """
    soa_pages = set()

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

        for page_num, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").lower()
            tables = page.extract_tables()

            # Primary detection: page has an SOA keyword AND contains a table
            has_keyword = any(kw in text for kw in SOA_KEYWORDS)
            has_table = bool(tables and any(len(t) > 2 for t in tables))

            if has_keyword and has_table:
                soa_pages.add(page_num)
                continue

            # Secondary detection: table content looks like an SOA
            # (must have a large table with many SOA-specific indicators)
            if has_table:
                for table in tables:
                    if len(table) < 4:  # SOA tables have many rows
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
                    if matches >= 4:  # stricter threshold
                        soa_pages.add(page_num)
                        break

        # Include continuation pages: only if the next page is consecutive
        # and has a table (likely a multi-page SOA table)
        continuation_pages = set()
        for pg in sorted(soa_pages):
            next_pg = pg + 1
            if next_pg <= total_pages and next_pg not in soa_pages:
                next_page = pdf.pages[next_pg - 1]
                next_text = (next_page.extract_text() or "").lower()
                next_tables = next_page.extract_tables()
                # Only add if it has a substantial table and no new section heading
                has_big_table = next_tables and any(len(t) > 3 for t in next_tables)
                is_new_section = any(
                    heading in next_text
                    for heading in ["table of contents", "list of tables", "synopsis",
                                    "appendix", "references", "abbreviations"]
                )
                if has_big_table and not is_new_section:
                    continuation_pages.add(next_pg)

        soa_pages.update(continuation_pages)

    result = sorted(soa_pages)
    print(f"[SOA DETECT] Found SOA content on pages: {result}")
    return result


def pdf_pages_to_base64_images(pdf_path, page_numbers, resolution=200, save_dir=None):
    """Convert specific PDF pages to base64-encoded PNG images using pdfplumber.

    Uses pdfplumber's built-in .to_image() — no Poppler or other system
    dependencies required.

    Args:
        pdf_path: Path to the PDF file.
        page_numbers: List of 1-indexed page numbers to convert.
        resolution: Resolution for rendering (higher = better OCR, larger payload).
        save_dir: Optional directory to save rendered PNG images to disk.

    Returns:
        list[dict]: Each dict has 'page_num', 'base64_data', 'media_type'.
    """
    images = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num in page_numbers:
            if page_num < 1 or page_num > len(pdf.pages):
                print(f"[PDF→IMG] Skipping invalid page {page_num}")
                continue

            page = pdf.pages[page_num - 1]  # 0-indexed
            page_image = page.to_image(resolution=resolution)

            buf = io.BytesIO()
            page_image.original.save(buf, format="PNG")
            png_bytes = buf.getvalue()
            b64 = base64.standard_b64encode(png_bytes).decode("utf-8")
            images.append({
                "page_num": page_num,
                "base64_data": b64,
                "media_type": "image/png",
            })
            print(f"[PDF→IMG] Page {page_num} rendered ({len(b64)} bytes base64)")

            # Save image to disk if save_dir is provided
            if save_dir:
                try:
                    img_filename = f"page_{page_num}.png"
                    img_path = os.path.join(save_dir, img_filename)
                    with open(img_path, "wb") as img_file:
                        img_file.write(png_bytes)
                    print(f"[PDF→IMG] Saved {img_path}")
                except Exception as save_err:
                    print(f"[WARN] Failed to save image for page {page_num}: {save_err}")

    return images


def call_vision_for_soa(page_images):
    """Send page images to Claude Vision to extract SOA table as structured JSON.

    Args:
        page_images: List of dicts from pdf_pages_to_base64_images().

    Returns:
        dict: Parsed JSON with full table structure, or error dict.
    """
    # Build multi-image message content
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
        "messages": [
            {"role": "user", "content": content_parts}
        ],
    })

    print(f"[VISION] Sending {len(page_images)} page image(s) to Claude Vision")

    response = client.invoke_model(
        modelId=LLM_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )

    result = json.loads(response["body"].read())
    raw_text = result["content"][0]["text"]

    # Strip any accidental markdown fences
    json_clean = re.sub(r'^```(?:json)?\s*', '', raw_text.strip())
    json_clean = re.sub(r'\s*```$', '', json_clean)

    try:
        soa_json = json.loads(json_clean)
    except json.JSONDecodeError as e:
        print(f"[VISION] JSON parse failed: {e}")
        print(f"[VISION] Raw response (first 500 chars): {raw_text[:500]}")
        return {"error": f"Failed to parse LLM response as JSON: {str(e)}", "raw_response": raw_text}

    return soa_json


MAX_PAGES_PER_VISION_CALL = 5  # Claude Vision limit for reliable processing


def extract_soa_from_pdf(pdf_path):
    """Full pipeline: PDF → detect SOA pages → render as images → Claude Vision → structured JSON.

    This is the main entry point for automated SOA extraction. It:
    1. Scans the PDF to find pages containing SOA/SOE tables
    2. Converts those pages to high-resolution images
    3. Sends images to Claude Vision (batched if >5 pages)
    4. Parses and saves the resulting structured JSON

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        dict: Extraction result with 'soa_json', 'pages_found', etc., or error dict.
    """
    # Step 1: Find SOA pages
    soa_page_nums = find_soa_pages(pdf_path)
    if not soa_page_nums:
        return {
            "error": "No Schedule of Assessments/Events/Activities pages detected in this PDF.",
            "suggestion": "The PDF may not contain an SOA table, or the table format was not recognized.",
        }

    print(f"[EXTRACT SOA] Step 1 complete: {len(soa_page_nums)} SOA page(s) found: {soa_page_nums}")

    # Step 2: Convert pages to images and save to disk
    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_save_dir = os.path.join(IMAGE_OUTPUT_DIR, f"{pdf_basename}_{timestamp}")
    os.makedirs(image_save_dir, exist_ok=True)

    page_images = pdf_pages_to_base64_images(pdf_path, soa_page_nums, save_dir=image_save_dir)
    if not page_images:
        return {"error": "Failed to convert PDF pages to images."}

    print(f"[EXTRACT SOA] Step 2 complete: {len(page_images)} page image(s) rendered")

    # Step 3: Send to Claude Vision — batch if too many pages
    if len(page_images) <= MAX_PAGES_PER_VISION_CALL:
        soa_json = call_vision_for_soa(page_images)
        if isinstance(soa_json, dict) and "error" in soa_json:
            return soa_json
    else:
        # Process in batches and merge USDM results
        all_encounters = []
        all_activities = []
        all_epochs = []
        all_timelines = []
        all_footnotes = []
        study_design_base = None

        for i in range(0, len(page_images), MAX_PAGES_PER_VISION_CALL):
            batch = page_images[i:i + MAX_PAGES_PER_VISION_CALL]
            batch_pages = [img["page_num"] for img in batch]
            print(f"[EXTRACT SOA] Processing batch: pages {batch_pages}")

            batch_result = call_vision_for_soa(batch)
            if isinstance(batch_result, dict) and "error" in batch_result:
                print(f"[WARN] Batch failed for pages {batch_pages}: {batch_result.get('error')}")
                continue

            if isinstance(batch_result, dict):
                sd = batch_result.get("studyDesign", {})
                if study_design_base is None:
                    study_design_base = {
                        "id": sd.get("id", "StudyDesign_1"),
                        "name": sd.get("name", ""),
                        "label": sd.get("label", ""),
                        "description": sd.get("description", ""),
                    }
                all_encounters.extend(sd.get("encounters", []))
                all_activities.extend(sd.get("activities", []))
                all_epochs.extend(sd.get("epochs", []))
                all_timelines.extend(sd.get("scheduleTimelines", []))
                all_footnotes.extend(batch_result.get("footnotes", []))

        if not all_encounters and not all_activities:
            return {"error": "Failed to extract SOA content from any page batch."}

        # Re-number IDs sequentially across merged batches
        for idx, enc in enumerate(all_encounters, start=1):
            enc["id"] = f"Encounter_{idx}"
        for idx, act in enumerate(all_activities, start=1):
            act["id"] = f"Activity_{idx}"

        soa_json = {
            "studyDesign": {
                **(study_design_base or {}),
                "encounters": all_encounters,
                "activities": all_activities,
                "epochs": all_epochs,
                "scheduleTimelines": all_timelines,
            },
            "footnotes": all_footnotes,
        }

    print(f"[EXTRACT SOA] Step 3 complete: JSON extracted successfully")

    # Step 4: Save JSON file
    saved_path = None
    try:
        protocol_id = "UnknownProtocol"
        if isinstance(soa_json, dict):
            sd = soa_json.get("studyDesign", {})
            protocol_id = sd.get("name") or sd.get("id", "UnknownProtocol")
        safe_name = re.sub(r'[^\w\-.]', '_', str(protocol_id))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_soa_{timestamp}.json"
        saved_path = os.path.join(JSON_OUTPUT_DIR, filename)
        with open(saved_path, "w", encoding="utf-8") as f:
            json.dump(soa_json, f, indent=2, ensure_ascii=False)
        print(f"[EXTRACT SOA] Step 4 complete: JSON saved to {saved_path}")
    except Exception as save_err:
        print(f"[WARN] Failed to save JSON file: {save_err}")

    # Build lightweight image list for API response
    page_images_for_response = [
        {
            "page_num": img["page_num"],
            "base64_data": img["base64_data"],
            "media_type": img["media_type"],
        }
        for img in page_images
    ]

    return {
        "soa_json": soa_json,
        "pages_found": soa_page_nums,
        "total_soa_pages": len(soa_page_nums),
        "saved_to": saved_path,
        "images_saved_to": image_save_dir,
        "page_images": page_images_for_response,
    }


@app.route("/api/extract-soa", methods=["POST"])
def extract_soa():
    """Upload a PDF and automatically find + extract SOA tables to structured JSON."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        result = extract_soa_from_pdf(filepath)
        if "error" in result:
            return jsonify(result), 500

        return jsonify({
            "message": (
                f"Successfully extracted SOA table from '{filename}'. "
                f"Found {result['total_soa_pages']} SOA page(s): {result['pages_found']}."
            ),
            "soa_json": result["soa_json"],
            "pages_found": result["pages_found"],
            "total_soa_pages": result["total_soa_pages"],
            "saved_to": result["saved_to"],
            "page_images": result.get("page_images", []),
        })
    except Exception as e:
        return jsonify({"error": f"Failed to extract SOA from PDF: {str(e)}"}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
