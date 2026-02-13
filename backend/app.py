import os
import re
import json
import hashlib
import pdfplumber
import boto3
from pinecone import Pinecone, ServerlessSpec
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

BASEDIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASEDIR, ".env"))

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
    """Lazy-init AWS Bedrock Runtime client."""
    global bedrock_client
    if bedrock_client is None:
        bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
