# SOA Extractor

Extract Schedule of Assessments (SOA), Schedule of Events (SOE), and Schedule of Activities tables from clinical trial protocol PDFs — preserving the exact table structure, semantic relationships, and all cell-level detail.

---

## Extraction Methods

### 1. Automated Vision Extraction (Recommended)

Fully automated pipeline that detects SOA tables inside a PDF, renders them as images, and uses Claude Vision to extract structured JSON.

#### How It Works

1. **SOA Page Detection** — The PDF is scanned page-by-page using `pdfplumber`. Pages are identified as SOA content when they contain both an SOA keyword (e.g., "schedule of assessments", "schedule of events") and a table with 3+ rows. A secondary check catches pages with tables containing 4+ SOA-specific indicators (e.g., "screening", "baseline", "visit").
2. **Continuation Page Detection** — Pages immediately following a detected SOA page are included if they contain a substantial table (likely a multi-page SOA).
3. **Page-to-Image Rendering** — Detected pages are rendered to high-resolution PNG images (200 DPI) using `pdfplumber`'s built-in `.to_image()` — no external system dependencies like Poppler required.
4. **Batched Vision Extraction** — Page images are sent to Claude Vision (via AWS Bedrock) in batches of up to 5 pages per call. A detailed system prompt instructs the model to:
   - Extract every row, column, and cell exactly as shown
   - Preserve multi-level column headers with parent-child relationships
   - Mark cells with `"X"` or `""` exactly as they appear
   - Capture all footnotes, comments, and annotations verbatim
   - Combine multi-page tables into one coherent JSON structure
5. **Result Merging** — If multiple batches are processed, tables and visits are merged and visit IDs are re-numbered sequentially.
6. **JSON Output** — The structured JSON is saved to `backend/json_outputs/` and returned to the frontend.

#### Output JSON Structure

```json
{
  "document_info": {
    "protocol_id": "H2Q-MC-LZZT(c)",
    "title": "Schedule of Activities",
    "source_pages": [53, 54]
  },
  "tables": [
    {
      "title": "Schedule of Activities",
      "columns": ["Procedure", "Visit 1", "Visit 2", ...],
      "rows": [
        {"procedure": "Vital signs", "values": ["X", "", "X", ...]}
      ],
      "footnotes": ["a = Note text...", "b = Note text..."]
    }
  ],
  "visits": [
    {
      "visit_id": 1,
      "phase": "Screening",
      "timing": "Visit 1",
      "activities": [
        {"procedure": "Vital signs", "marked": true, "comment": null},
        {"procedure": "ADAS-Cog", "marked": true, "comment": "P = Practice only..."}
      ]
    }
  ]
}
```

#### Usage

1. Open http://localhost:3000
2. Navigate to the **Image Extract** tab
3. Upload a clinical trial protocol PDF (drag & drop or browse)
4. Click **"Extract SOA from PDF"**
5. View the extracted tables, visit details, and full JSON
6. Use **Copy JSON** or **Download JSON** to export

---

### 2. RAG-Based Chat Extraction

Uses vector search and LLM reconstruction for interactive SOA extraction via chat.

#### How It Works

1. **PDF Text Extraction** — The uploaded PDF is parsed page-by-page using `pdfplumber` to extract all text content.
2. **Chunking** — Extracted text is split into overlapping chunks (500 characters, 100 overlap) to maintain context across chunk boundaries.
3. **Embedding** — Each chunk is embedded into a 1024-dimensional vector using AWS Bedrock's Titan Embed Text V2 model.
4. **Vector Storage** — Embeddings and metadata are stored in a Pinecone serverless index.
5. **Two-Phase Retrieval** — Semantic search identifies SOA pages, then all chunks from those pages are fetched.
6. **LLM Reconstruction** — Retrieved chunks are passed to Claude, which reconstructs the SOA table in HTML format.

#### Usage

1. Open http://localhost:3000
2. Upload a PDF via the **Ingest** tab
3. Navigate to the **Chat** tab
4. Ask the chatbot to print the SOA table (e.g., "Print the Schedule of Assessments")
5. The full table is rendered in HTML — use the **Copy** button to grab the content

---

## Setup

### Backend (Python)
```bash
cd backend
pip install -r requirements.txt
python app.py
```
Backend runs on http://localhost:5000

### Frontend (React)
```bash
cd frontend
npm install
npm start
```
Frontend runs on http://localhost:3000

### Environment Variables

Create a `backend/.env` file with:
```
PINECONE_API_KEY=your_pinecone_api_key
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Health check |
| `/api/ingest` | POST | Upload and ingest a PDF into Pinecone |
| `/api/query` | POST | Query raw chunks from Pinecone |
| `/api/chat` | POST | Chat with the LLM to extract tables (RAG) |
| `/api/extract-soa` | POST | Upload a PDF → auto-detect SOA → extract JSON (Vision) |

## Tech Stack
- **Backend**: Python, Flask, pdfplumber, Pillow
- **Vector DB**: Pinecone (serverless, cosine similarity)
- **Embeddings**: AWS Bedrock — Titan Embed Text V2 (1024 dimensions)
- **LLM**: AWS Bedrock — Claude Sonnet 4.5 (text + vision)
- **Frontend**: React, Axios, React Router
