# SOA Extractor

Extract Schedule of Assessments (SOA), Schedule of Events (SOE), and Schedule of Activities tables from clinical trial protocol PDFs — preserving the exact table structure.

## How It Works

1. **PDF Text Extraction** — The uploaded PDF is parsed page-by-page using `pdfplumber` to extract all text content.
2. **Chunking** — Extracted text is split into overlapping chunks (500 characters, 100 overlap) to maintain context across chunk boundaries.
3. **Embedding** — Each chunk is embedded into a 1024-dimensional vector using AWS Bedrock's Titan Embed Text V2 model (`amazon.titan-embed-text-v2:0`).
4. **Vector Storage** — Embeddings and their metadata (page number, chunk index, text) are stored in a Pinecone serverless index for fast similarity search.
5. **Two-Phase Retrieval** — When a user asks for the SOA table:
   - **Phase 1**: A semantic search identifies which pages contain SOA-related content.
   - **Phase 2**: All chunks from those pages are fetched using metadata filters, ensuring complete coverage.
6. **LLM Reconstruction** — The retrieved chunks are passed as context to Claude (via AWS Bedrock), which reconstructs the full SOA table in HTML format — preserving every row, column, and cell value exactly as it appears in the PDF.

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

## Usage
1. Open http://localhost:3000
2. Upload a clinical trial protocol PDF
3. Navigate to the **Chat** tab
4. Ask the chatbot to print the SOA table (e.g., "Print the Schedule of Assessments")
5. The full table is rendered in HTML — use the **Copy** button to grab the content

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Health check |
| `/api/ingest` | POST | Upload and process a PDF |
| `/api/query` | POST | Query raw chunks from Pinecone |
| `/api/chat` | POST | Chat with the LLM to extract tables |

## Tech Stack
- **Backend**: Python, Flask, pdfplumber
- **Vector DB**: Pinecone (serverless, cosine similarity)
- **Embeddings**: AWS Bedrock — Titan Embed Text V2 (1024 dimensions)
- **LLM**: AWS Bedrock — Claude Sonnet 4.5
- **Frontend**: React, Axios, React Router
