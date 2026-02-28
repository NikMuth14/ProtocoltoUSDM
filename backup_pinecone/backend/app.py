import os
import io
import re
import json
import base64
import hashlib
import tempfile
from datetime import datetime
import pdfplumber
import boto3
from botocore.config import Config as BotoConfig
from pinecone import Pinecone, ServerlessSpec
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

BASEDIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASEDIR, ".env"))

JSON_OUTPUT_DIR = os.path.join(BASEDIR, "json_outputs")
os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(BASEDIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB limit

# ── Config ──────────────────────────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

print(f"[STARTUP] Pinecone key loaded: {'YES' if PINECONE_API_KEY else 'NO'} (len={len(PINECONE_API_KEY) if PINECONE_API_KEY else 0})")
print(f"[STARTUP] AWS key loaded: {'YES' if AWS_ACCESS_KEY_ID else 'NO'}")

PINECONE_INDEX_NAME = "soa"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBEDDING_DIMENSION = 1024  # Titan V2 outputs 1024-dim vectors
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 100  # overlap between chunks
LLM_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# ── Clients ─────────────────────────────────────────────────────────────────
pc = None
bedrock_client = None


def get_pinecone():
    """Lazy-init Pinecone client."""
    global pc
    if pc is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc


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


def ensure_index_exists():
    """Create the Pinecone index if it doesn't already exist."""
    pinecone = get_pinecone()
    existing = [idx.name for idx in pinecone.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        pinecone.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pinecone.Index(PINECONE_INDEX_NAME)


# ── PDF Processing ──────────────────────────────────────────────────────────
def extract_text_from_pdf(filepath):
    """Extract text and table data from each page of a PDF."""
    pages = []
    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""

            # Also extract tables as structured text
            tables = page.extract_tables()
            table_texts = []
            for table in (tables or []):
                rows = []
                for row in table:
                    cells = [str(c).strip() if c else "" for c in row]
                    rows.append(" | ".join(cells))
                table_texts.append("\n".join(rows))

            combined = text
            if table_texts:
                combined += "\n\n[TABLE]\n" + "\n[TABLE]\n".join(table_texts)

            pages.append({
                "page_num": page_num,
                "text": combined,
            })
    return pages


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks


def get_embedding(text):
    """Get embedding vector from AWS Bedrock Titan Embeddings V2."""
    client = get_bedrock()
    body = json.dumps({
        "inputText": text[:10000],  # Titan V2 max input
    })
    response = client.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


def get_embeddings_batch(texts):
    """Get embeddings for a batch of texts (sequential calls to Bedrock)."""
    embeddings = []
    for text in texts:
        emb = get_embedding(text)
        embeddings.append(emb)
    return embeddings


def ingest_pdf_to_pinecone(filepath, doc_name):
    """Full pipeline: PDF → text → chunks → embeddings → Pinecone upsert."""
    index = ensure_index_exists()

    # Generate a stable document ID from filename
    doc_id = hashlib.md5(doc_name.encode()).hexdigest()[:12]

    # Delete any existing vectors for this document
    try:
        index.delete(filter={"doc_id": doc_id})
    except Exception:
        pass  # Index might be empty

    # Extract text from PDF
    pages = extract_text_from_pdf(filepath)
    if not pages:
        return {"error": "No text could be extracted from the PDF."}

    # Chunk and prepare vectors
    vectors_to_upsert = []
    total_chunks = 0

    for page_data in pages:
        page_num = page_data["page_num"]
        text = page_data["text"]

        if not text.strip():
            continue

        chunks = chunk_text(text)
        for chunk_idx, chunk in enumerate(chunks):
            vector_id = f"{doc_id}_p{page_num}_c{chunk_idx}"
            embedding = get_embedding(chunk)

            vectors_to_upsert.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "page_num": page_num,
                    "chunk_index": chunk_idx,
                    "text": chunk,
                },
            })
            total_chunks += 1

            # Upsert in batches of 50
            if len(vectors_to_upsert) >= 50:
                index.upsert(vectors=vectors_to_upsert)
                vectors_to_upsert = []

    # Upsert remaining vectors
    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert)

    return {
        "doc_id": doc_id,
        "doc_name": doc_name,
        "total_pages": len(pages),
        "total_chunks": total_chunks,
    }


# ── Routes ──────────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/ingest", methods=["POST"])
def ingest():
    """Upload a PDF and store it in the Pinecone 'soa' index."""
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
        result = ingest_pdf_to_pinecone(filepath, filename)
        if "error" in result:
            return jsonify(result), 500

        return jsonify({
            "message": f"Successfully ingested '{filename}' into Pinecone index '{PINECONE_INDEX_NAME}'.",
            **result,
        })
    except Exception as e:
        return jsonify({"error": f"Failed to ingest PDF: {str(e)}"}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/api/query", methods=["POST"])
def query_soa():
    """Query Pinecone for Schedule of Assessments / Events / Activities content."""
    data = request.get_json() or {}
    query_text = data.get("query", "Schedule of Assessments Schedule of Events Schedule of Activities SOA SOE")
    top_k = data.get("top_k", 50)

    try:
        index = ensure_index_exists()

        # Embed the query
        query_embedding = get_embedding(query_text)

        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
        )

        if not results.matches:
            return jsonify({
                "chunks": [],
                "message": "No matching content found in the Pinecone index.",
            })

        # Organize results by page
        pages_map = {}
        for match in results.matches:
            meta = match.metadata
            page_num = meta.get("page_num", 0)
            if page_num not in pages_map:
                pages_map[page_num] = {
                    "page_num": page_num,
                    "doc_name": meta.get("doc_name", ""),
                    "chunks": [],
                    "max_score": 0,
                }
            pages_map[page_num]["chunks"].append({
                "text": meta.get("text", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "score": round(match.score, 4),
            })
            if match.score > pages_map[page_num]["max_score"]:
                pages_map[page_num]["max_score"] = round(match.score, 4)

        # Sort pages by max relevance score, then chunks within each page by chunk_index
        pages = sorted(pages_map.values(), key=lambda p: -p["max_score"])
        for page in pages:
            page["chunks"] = sorted(page["chunks"], key=lambda c: c["chunk_index"])

        return jsonify({
            "pages": pages,
            "total_matches": len(results.matches),
            "query_used": query_text,
        })

    except Exception as e:
        return jsonify({"error": f"Failed to query: {str(e)}"}), 500


def retrieve_soa_context(query_text, top_k=100):
    """Two-phase retrieval: find SOA pages, then fetch ALL chunks from those pages."""
    index = ensure_index_exists()
    query_embedding = get_embedding(query_text)

    # Phase 1: Semantic search to identify which pages contain SOA content
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    if not results.matches:
        return ""

    # Identify the pages that matched
    soa_pages = set()
    for match in results.matches:
        page_num = match.metadata.get("page_num", 0)
        soa_pages.add(int(page_num))

    print(f"[RETRIEVAL] Phase 1: Found {len(soa_pages)} SOA-related pages: {sorted(soa_pages)}")

    # Phase 2: Fetch ALL chunks from those pages using metadata filter
    all_chunks = []
    for page_num in sorted(soa_pages):
        page_results = index.query(
            vector=query_embedding,
            top_k=200,
            include_metadata=True,
            filter={"page_num": {"$eq": page_num}},
        )
        for match in page_results.matches:
            all_chunks.append(match)

    print(f"[RETRIEVAL] Phase 2: Retrieved {len(all_chunks)} total chunks from {len(soa_pages)} pages")

    # Group by page, sort by page then chunk_index for coherent context
    pages_map = {}
    for match in all_chunks:
        meta = match.metadata
        page_num = meta.get("page_num", 0)
        chunk_index = meta.get("chunk_index", 0)
        text = meta.get("text", "")
        key = (page_num, chunk_index)
        if key not in pages_map:
            pages_map[key] = {
                "page_num": page_num,
                "chunk_index": chunk_index,
                "text": text,
            }

    # Assemble context ordered by page then chunk
    sorted_chunks = sorted(pages_map.values(), key=lambda c: (c["page_num"], c["chunk_index"]))

    context_parts = []
    current_page = None
    for chunk in sorted_chunks:
        if chunk["page_num"] != current_page:
            current_page = chunk["page_num"]
            context_parts.append(f"\n[Page {current_page}]")
        context_parts.append(chunk["text"])

    return "\n".join(context_parts)


def call_llm(system_prompt, user_message):
    """Call Claude 3.5 Sonnet via AWS Bedrock."""
    client = get_bedrock()

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 64000,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_message}
        ],
    })

    response = client.invoke_model(
        modelId=LLM_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )

    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


SYSTEM_PROMPT = """You are an expert clinical protocol analyst. You have access to content extracted from a clinical trial protocol PDF.

When the user asks you to print or show the Schedule of Assessments (SOA), Schedule of Events (SOE), Schedule of Activities, or any similar table:

1. Reproduce the table EXACTLY as it appears in the source document.
2. Use an HTML table format with proper <table>, <thead>, <tbody>, <tr>, <th>, <td> tags.
3. Preserve ALL columns, ALL rows, ALL cell values, ALL merged cells (use colspan/rowspan as needed).
4. Preserve the exact text in each cell — do not paraphrase, summarize, or omit anything.
5. If the table spans multiple pages, combine all parts into one complete table.
6. Use clean, readable HTML table styling with borders.
7. If you find multiple SOA tables (e.g., for different study periods), include all of them.

For any other questions about the protocol, answer based on the provided context.

CRITICAL RULES:
- NEVER summarize, abbreviate, or truncate any part of a table.
- NEVER write "The table continues with..." or "Additional assessments include..." or any similar summary.
- NEVER skip rows. Every single row from the source must appear in your output.
- NEVER add a key/legend unless it literally appears in the source.
- If the table is very large, output the COMPLETE table no matter how long it is.
- You have enough output tokens. Use them all if needed. Completeness is mandatory."""

JSON_SYSTEM_PROMPT = """You are an expert clinical protocol analyst. Convert the provided SOA/SOE table content into a structured JSON format.

Output ONLY valid JSON — no markdown, no explanation, no code fences.

The JSON structure MUST follow this exact format:
{
  "study_id": "<protocol or study identifier from the document>",
  "visits": [
    {
      "visit_id": 1,
      "week": <week number or day number as a number, use null if not available>,
      "name": "<visit name, e.g. Screening, Baseline, Week 4, Follow-up>",
      "activities": [
        "<activity/assessment performed at this visit>",
        "<another activity>",
        ...
      ]
    },
    ...
  ]
}

Each visit column in the SOA table becomes one object in the "visits" array.
The "activities" array should list ONLY the activities/assessments that are marked (e.g. with X, checkmark, or any indicator) for that visit.
Do NOT include activities that are not marked for a given visit.

CRITICAL RULES:
- Include EVERY visit column from the table. Do not skip or summarize.
- Include EVERY marked activity for each visit. Do not skip any.
- Preserve exact activity names as they appear in the table.
- The visit_id should be sequential starting from 1.
- Extract the week/day number from the column header if available.
- Extract the study_id from the document context (protocol number, study ID, etc.).
- If there are multiple SOA tables (e.g. different study periods), include all visits from all tables.
- Output raw JSON only. No markdown formatting."""


@app.route("/api/chat", methods=["POST"])
def chat():
    """Chatbot endpoint: retrieves context from Pinecone, sends to Claude for response."""
    data = request.get_json() or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Retrieve relevant context from Pinecone
        context = retrieve_soa_context(user_message, top_k=100)

        if not context:
            return jsonify({
                "response": "No protocol content found in the database. Please ingest a PDF first.",
                "has_context": False,
            })

        # Build the prompt with context
        full_message = f"""Here is the relevant content extracted from the clinical trial protocol:

<protocol_content>
{context}
</protocol_content>

User question: {user_message}"""

        # Call Claude for HTML table
        llm_response = call_llm(SYSTEM_PROMPT, full_message)

        # Check if response contains a table — if so, also generate JSON
        soa_json = None
        has_table = bool(re.search(r'<table[\s>]', llm_response, re.IGNORECASE))
        if has_table:
            try:
                json_message = f"""Here is the relevant content extracted from the clinical trial protocol:

<protocol_content>
{context}
</protocol_content>

Convert the Schedule of Assessments / Schedule of Events table into JSON format."""
                json_raw = call_llm(JSON_SYSTEM_PROMPT, json_message)
                # Strip any accidental markdown fences
                json_clean = re.sub(r'^```(?:json)?\s*', '', json_raw.strip())
                json_clean = re.sub(r'\s*```$', '', json_clean)
                soa_json = json.loads(json_clean)

                # Save JSON file locally with ProtocolName_DateTime
                try:
                    study_id = "UnknownProtocol"
                    if isinstance(soa_json, dict):
                        study_id = soa_json.get("study_id", "UnknownProtocol")
                    elif isinstance(soa_json, list) and len(soa_json) > 0:
                        study_id = soa_json[0].get("study_id", "UnknownProtocol")
                    # Sanitize filename
                    safe_name = re.sub(r'[^\w\-.]', '_', str(study_id))
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{safe_name}_{timestamp}.json"
                    filepath = os.path.join(JSON_OUTPUT_DIR, filename)
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(soa_json, f, indent=2, ensure_ascii=False)
                    print(f"[JSON SAVED] {filepath}")
                except Exception as save_err:
                    print(f"[WARN] Failed to save JSON file: {save_err}")

            except Exception as json_err:
                print(f"[WARN] Failed to generate JSON: {json_err}")
                soa_json = None

        return jsonify({
            "response": llm_response,
            "has_context": True,
            "soa_json": soa_json,
        })

    except Exception as e:
        return jsonify({"error": f"Failed to process: {str(e)}"}), 500


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

IMAGE_SOA_SYSTEM_PROMPT = """You are an expert clinical protocol analyst specializing in extracting Schedule of Assessments (SOA) / Schedule of Events (SOE) tables from images of clinical trial protocol pages.

You will receive one or more page images from a clinical trial protocol PDF. Your task is to extract ALL SOA/SOE table content into a structured JSON that preserves the FULL semantic meaning, relationships, and hierarchy.

Output ONLY valid JSON — no markdown, no explanation, no code fences.

The JSON structure MUST follow this exact format:
{
  "document_info": {
    "protocol_id": "<protocol/study identifier visible in the document>",
    "title": "<document title>",
    "source_pages": [<list of page numbers these tables came from>]
  },
  "tables": [
    {
      "table_title": "<title of this specific table if visible, e.g. Schedule of Activities - Period 1>",
      "source_page": <page number>,
      "column_headers": {
        "levels": [
          {
            "level": 1,
            "headers": [
              {
                "text": "<top-level header text>",
                "spans_columns": ["<list of sub-column names this header spans>"]
              }
            ]
          },
          {
            "level": 2,
            "headers": [
              {
                "text": "<sub-header text, e.g. Days -28 to -2, Day -1, Day 1>",
                "parent": "<parent header from level 1>"
              }
            ]
          }
        ]
      },
      "rows": [
        {
          "procedure": "<exact procedure/activity name>",
          "values": {
            "<column_name>": "<cell value: X, empty string, or any text found in the cell>"
          },
          "comments": "<full comment text for this row, preserving all detail. null if no comment>"
        }
      ]
    }
  ],
  "visits": [
    {
      "visit_id": 1,
      "phase": "<study phase: Screening, Period 1, Washout, Period 2, Follow-up/ED, etc.>",
      "timing": "<timing label from column header>",
      "activities": [
        {
          "procedure": "<exact procedure name>",
          "marked": true,
          "comment": "<associated comment if any, else null>"
        }
      ]
    }
  ]
}

CRITICAL RULES:
- Extract EVERY row, EVERY column, EVERY cell value exactly as shown.
- If the table spans multiple pages, combine all parts into one coherent table in the "tables" array.
- Preserve multi-level column headers and their parent-child relationships.
- Preserve ALL comments/notes/footnotes verbatim — do not summarize or truncate them.
- If a cell contains "X", record it as "X". If empty, record as "".
- Capture merged/spanning headers accurately.
- The "visits" array should list each visit column with ONLY the procedures marked (X) for that visit.
- Include the full comment text associated with each procedure.
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


def pdf_pages_to_base64_images(pdf_path, page_numbers, resolution=200):
    """Convert specific PDF pages to base64-encoded PNG images using pdfplumber.

    Uses pdfplumber's built-in .to_image() — no Poppler or other system
    dependencies required.

    Args:
        pdf_path: Path to the PDF file.
        page_numbers: List of 1-indexed page numbers to convert.
        resolution: Resolution for rendering (higher = better OCR, larger payload).

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
            b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
            images.append({
                "page_num": page_num,
                "base64_data": b64,
                "media_type": "image/png",
            })
            print(f"[PDF→IMG] Page {page_num} rendered ({len(b64)} bytes base64)")

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
            "Extract the COMPLETE table(s) into the structured JSON format. "
            "Preserve every header level, every row, every cell value, "
            "and every comment exactly as shown. "
            "If the table spans multiple pages, combine them into one coherent table. "
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

    # Step 2: Convert pages to images
    page_images = pdf_pages_to_base64_images(pdf_path, soa_page_nums)
    if not page_images:
        return {"error": "Failed to convert PDF pages to images."}

    print(f"[EXTRACT SOA] Step 2 complete: {len(page_images)} page image(s) rendered")

    # Step 3: Send to Claude Vision — batch if too many pages
    if len(page_images) <= MAX_PAGES_PER_VISION_CALL:
        soa_json = call_vision_for_soa(page_images)
        if isinstance(soa_json, dict) and "error" in soa_json:
            return soa_json
    else:
        # Process in batches and merge results
        all_tables = []
        all_visits = []
        doc_info = None

        for i in range(0, len(page_images), MAX_PAGES_PER_VISION_CALL):
            batch = page_images[i:i + MAX_PAGES_PER_VISION_CALL]
            batch_pages = [img["page_num"] for img in batch]
            print(f"[EXTRACT SOA] Processing batch: pages {batch_pages}")

            batch_result = call_vision_for_soa(batch)
            if isinstance(batch_result, dict) and "error" in batch_result:
                print(f"[WARN] Batch failed for pages {batch_pages}: {batch_result.get('error')}")
                continue

            if isinstance(batch_result, dict):
                if doc_info is None:
                    doc_info = batch_result.get("document_info")
                all_tables.extend(batch_result.get("tables", []))
                all_visits.extend(batch_result.get("visits", []))

        if not all_tables and not all_visits:
            return {"error": "Failed to extract SOA content from any page batch."}

        # Re-number visit IDs sequentially
        for idx, visit in enumerate(all_visits, start=1):
            visit["visit_id"] = idx

        soa_json = {
            "document_info": doc_info or {},
            "tables": all_tables,
            "visits": all_visits,
        }

    print(f"[EXTRACT SOA] Step 3 complete: JSON extracted successfully")

    # Step 4: Save JSON file
    saved_path = None
    try:
        protocol_id = "UnknownProtocol"
        if isinstance(soa_json, dict):
            doc_info = soa_json.get("document_info", {})
            protocol_id = doc_info.get("protocol_id") or soa_json.get("study_id", "UnknownProtocol")
        safe_name = re.sub(r'[^\w\-.]', '_', str(protocol_id))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_soa_{timestamp}.json"
        saved_path = os.path.join(JSON_OUTPUT_DIR, filename)
        with open(saved_path, "w", encoding="utf-8") as f:
            json.dump(soa_json, f, indent=2, ensure_ascii=False)
        print(f"[EXTRACT SOA] Step 4 complete: JSON saved to {saved_path}")
    except Exception as save_err:
        print(f"[WARN] Failed to save JSON file: {save_err}")

    return {
        "soa_json": soa_json,
        "pages_found": soa_page_nums,
        "total_soa_pages": len(soa_page_nums),
        "saved_to": saved_path,
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
        })
    except Exception as e:
        return jsonify({"error": f"Failed to extract SOA from PDF: {str(e)}"}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
