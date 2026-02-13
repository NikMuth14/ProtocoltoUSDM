import React, { useState } from 'react';
import axios from 'axios';

const API_URL = 'http://localhost:5000/api';

function ExtractPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [results, setResults] = useState(null);

  const handleExtract = async () => {
    setLoading(true);
    setError('');
    setMessage('');
    setResults(null);

    try {
      const response = await axios.post(`${API_URL}/query`, {
        query: "Schedule of Assessments Schedule of Events Schedule of Activities SOA table",
        top_k: 100,
      }, {
        timeout: 120000,
      });

      const data = response.data;
      if (data.message) {
        setMessage(data.message);
      } else {
        setResults(data);
        setMessage(`Retrieved ${data.total_matches} chunks across ${data.pages.length} page(s).`);
      }
    } catch (err) {
      if (err.response && err.response.data && err.response.data.error) {
        setError(err.response.data.error);
      } else {
        setError('Failed to connect to the server. Make sure the backend is running.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setError('');
    setMessage('');
    setResults(null);
  };

  const copyPageText = (page) => {
    const fullText = page.chunks.map(c => c.text).join('\n');
    navigator.clipboard.writeText(fullText);
  };

  const copyAllText = () => {
    if (!results) return;
    const allText = results.pages.map(page =>
      `--- Page ${page.page_num} ---\n` + page.chunks.map(c => c.text).join('\n')
    ).join('\n\n');
    navigator.clipboard.writeText(allText);
  };

  return (
    <div>
      <section className="upload-section">
        <div className="extract-prompt">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#6366f1" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <h2 className="extract-title">Retrieve SOA Table</h2>
          <p className="extract-desc">
            Search the ingested protocol in Pinecone for the Schedule of Assessments / Events / Activities table.
          </p>
        </div>

        <div className="actions">
          <button
            className="btn btn-primary"
            onClick={handleExtract}
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="spinner"></span>
                Searching Pinecone...
              </>
            ) : (
              <>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="11" cy="11" r="8" />
                  <line x1="21" y1="21" x2="16.65" y2="16.65" />
                </svg>
                Extract SOA Table
              </>
            )}
          </button>
          {results && (
            <>
              <button className="btn btn-secondary" onClick={copyAllText}>
                Copy All
              </button>
              <button className="btn btn-secondary" onClick={handleReset}>
                Reset
              </button>
            </>
          )}
        </div>
      </section>

      {error && (
        <div className="alert alert-error">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <line x1="15" y1="9" x2="9" y2="15" />
            <line x1="9" y1="9" x2="15" y2="15" />
          </svg>
          {error}
        </div>
      )}
      {message && !error && (
        <div className="alert alert-success">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
            <polyline points="22 4 12 14.01 9 11.01" />
          </svg>
          {message}
        </div>
      )}

      {results && results.pages && (
        <section className="tables-section">
          {results.pages.map((page, idx) => (
            <div key={idx} className="table-card">
              <div className="table-header">
                <div className="table-meta">
                  <h3>Page {page.page_num}</h3>
                  <span className="badge">{page.doc_name}</span>
                  <span className="badge badge-dim">
                    {page.chunks.length} chunk(s) &middot; Score: {page.max_score}
                  </span>
                </div>
                <div className="table-actions">
                  <button
                    className="btn-icon"
                    title="Copy page text"
                    onClick={() => copyPageText(page)}
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                    </svg>
                    Copy
                  </button>
                </div>
              </div>
              <div className="page-content">
                {page.chunks.map((chunk, ci) => (
                  <div key={ci} className="chunk-block">
                    <div className="chunk-header">
                      <span className="chunk-label">Chunk {chunk.chunk_index}</span>
                      <span className="chunk-score">Relevance: {chunk.score}</span>
                    </div>
                    <pre className="chunk-text">{chunk.text}</pre>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </section>
      )}
    </div>
  );
}

export default ExtractPage;
