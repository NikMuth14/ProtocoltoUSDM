import React, { useState, useRef, useCallback } from 'react';
import axios from 'axios';

const API_URL = 'http://localhost:5000/api';

function ImageExtractPage() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [soaJson, setSoaJson] = useState(null);
  const [pagesFound, setPagesFound] = useState([]);
  const [dragActive, setDragActive] = useState(false);
  const [showJson, setShowJson] = useState(false);
  const [pageImages, setPageImages] = useState([]);
  const fileInputRef = useRef(null);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type === 'application/pdf') {
        setFile(droppedFile);
        setError('');
      } else {
        setError('Please upload a PDF file.');
      }
    }
  }, []);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError('');
    }
  };

  const handleExtract = async () => {
    if (!file) {
      setError('Please select a PDF file first.');
      return;
    }

    setLoading(true);
    setError('');
    setMessage('');
    setSoaJson(null);
    setPagesFound([]);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_URL}/extract-soa`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 600000,
      });

      const data = response.data;
      setSoaJson(data.soa_json);
      setPagesFound(data.pages_found || []);
      setPageImages(data.page_images || []);
      setMessage(data.message || 'SOA table extracted successfully.');
    } catch (err) {
      if (err.response && err.response.data && err.response.data.error) {
        setError(err.response.data.error);
        if (err.response.data.suggestion) {
          setError(prev => prev + ' ' + err.response.data.suggestion);
        }
      } else {
        setError('Failed to connect to the server. Make sure the backend is running.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setError('');
    setMessage('');
    setSoaJson(null);
    setPagesFound([]);
    setPageImages([]);
    setShowJson(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const copyJson = () => {
    if (soaJson) {
      navigator.clipboard.writeText(JSON.stringify(soaJson, null, 2));
    }
  };

  const downloadJson = () => {
    if (!soaJson) return;
    const blob = new Blob([JSON.stringify(soaJson, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const protocolId = soaJson?.document_info?.protocol_id || 'soa_extract';
    a.download = `${protocolId.replace(/[^\w\-.]/g, '_')}_soa_extract.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const renderVisitsTable = () => {
    if (!soaJson || !soaJson.visits || soaJson.visits.length === 0) return null;

    return (
      <div className="table-card" style={{ marginTop: '1.5rem' }}>
        <div className="table-header">
          <div className="table-meta">
            <h3>Extracted Visits</h3>
            <span className="badge">{soaJson.visits.length} visit(s)</span>
          </div>
        </div>
        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Visit</th>
                <th>Phase</th>
                <th>Timing</th>
                <th>Activities</th>
              </tr>
            </thead>
            <tbody>
              {soaJson.visits.map((visit, idx) => (
                <tr key={idx}>
                  <td style={{ fontWeight: 600 }}>{visit.visit_id}</td>
                  <td>{visit.phase || '—'}</td>
                  <td>{visit.timing || '—'}</td>
                  <td>
                    {visit.activities && visit.activities.length > 0 ? (
                      <ul style={{ margin: 0, paddingLeft: '1.2rem' }}>
                        {visit.activities.map((act, ai) => (
                          <li key={ai} style={{ fontSize: '0.85rem', marginBottom: '0.25rem' }}>
                            {act.procedure || act}
                            {act.comment && (
                              <span style={{ color: '#64748b', fontSize: '0.8rem', display: 'block' }}>
                                {act.comment}
                              </span>
                            )}
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <span style={{ color: '#94a3b8' }}>None</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  const renderRowsTable = (tableData, tableIndex) => {
    if (!tableData || !tableData.rows || tableData.rows.length === 0) return null;

    const allColumns = new Set();
    tableData.rows.forEach(row => {
      if (row.values) {
        Object.keys(row.values).forEach(k => allColumns.add(k));
      }
    });
    const columns = Array.from(allColumns);

    return (
      <div key={tableIndex} className="table-card" style={{ marginTop: '1.5rem' }}>
        <div className="table-header">
          <div className="table-meta">
            <h3>{tableData.table_title || `Table ${tableIndex + 1}`}</h3>
            {tableData.source_page && (
              <span className="badge">Page {tableData.source_page}</span>
            )}
            <span className="badge badge-dim">
              {tableData.rows.length} procedure(s) &middot; {columns.length} column(s)
            </span>
          </div>
        </div>
        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Procedure</th>
                {columns.map((col, i) => (
                  <th key={i}>{col}</th>
                ))}
                <th>Comments</th>
              </tr>
            </thead>
            <tbody>
              {tableData.rows.map((row, idx) => (
                <tr key={idx}>
                  <td style={{ fontWeight: 600 }}>{row.procedure}</td>
                  {columns.map((col, i) => (
                    <td key={i} style={{ textAlign: 'center' }}>
                      {row.values?.[col] || ''}
                    </td>
                  ))}
                  <td style={{ fontSize: '0.8rem', color: '#475569', maxWidth: '300px' }}>
                    {row.comments || '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  const renderAllTables = () => {
    if (!soaJson) return null;

    // New format: soaJson.tables is an array
    if (soaJson.tables && Array.isArray(soaJson.tables)) {
      return soaJson.tables.map((t, i) => renderRowsTable(t, i));
    }

    // Fallback: single table at soaJson.table
    if (soaJson.table && soaJson.table.rows) {
      return renderRowsTable(soaJson.table, 0);
    }

    return null;
  };

  return (
    <div>
      <section className="upload-section">
        <div
          className={`drop-zone ${dragActive ? 'drag-active' : ''} ${file ? 'has-file' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            onChange={handleFileChange}
            className="file-input"
          />
          {file ? (
            <div className="file-info">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                <polyline points="22 4 12 14.01 9 11.01" />
              </svg>
              <p className="file-name">{file.name}</p>
              <p className="file-size">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
            </div>
          ) : (
            <div className="upload-prompt">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
                <line x1="16" y1="13" x2="8" y2="13" />
                <line x1="16" y1="17" x2="8" y2="17" />
              </svg>
              <p className="upload-text">Drag & drop your protocol PDF here</p>
              <p className="upload-subtext">The SOA table will be automatically detected and extracted</p>
            </div>
          )}
        </div>

        <div className="actions">
          <button
            className="btn btn-primary"
            onClick={handleExtract}
            disabled={!file || loading}
          >
            {loading ? (
              <>
                <span className="spinner"></span>
                Detecting & extracting SOA...
              </>
            ) : (
              <>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="11" cy="11" r="8" />
                  <line x1="21" y1="21" x2="16.65" y2="16.65" />
                </svg>
                Extract SOA from PDF
              </>
            )}
          </button>
          {(file || soaJson) && (
            <button className="btn btn-secondary" onClick={handleReset}>
              Reset
            </button>
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

      {soaJson && (
        <section className="tables-section">
          {/* Document Info */}
          <div className="table-card">
            <div className="table-header">
              <div className="table-meta">
                <h3>Extraction Result</h3>
                {pagesFound.length > 0 && (
                  <span className="badge">
                    SOA found on page(s): {pagesFound.join(', ')}
                  </span>
                )}
              </div>
              <div className="table-actions">
                <button className="btn-icon btn-json" onClick={() => setShowJson(!showJson)}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="16 18 22 12 16 6" />
                    <polyline points="8 6 2 12 8 18" />
                  </svg>
                  {showJson ? 'Hide JSON' : 'View JSON'}
                </button>
                <button className="btn-icon" onClick={copyJson} title="Copy JSON">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                  </svg>
                  Copy
                </button>
                <button className="btn-icon" onClick={downloadJson} title="Download JSON">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="7 10 12 15 17 10" />
                    <line x1="12" y1="15" x2="12" y2="3" />
                  </svg>
                  Download
                </button>
              </div>
            </div>
            {soaJson.document_info && (
              <div className="stats-grid">
                <div className="stat-item">
                  <span className="stat-value">{soaJson.document_info.protocol_id || '—'}</span>
                  <span className="stat-label">Protocol ID</span>
                </div>
                <div className="stat-item">
                  <span className="stat-value">{soaJson.document_info.title || '—'}</span>
                  <span className="stat-label">Title</span>
                </div>
                <div className="stat-item">
                  <span className="stat-value">
                    {soaJson.document_info.source_pages
                      ? soaJson.document_info.source_pages.join(', ')
                      : '—'}
                  </span>
                  <span className="stat-label">Source Pages</span>
                </div>
              </div>
            )}
          </div>

          {/* Raw JSON View */}
          {showJson && (
            <div className="json-modal-overlay" onClick={() => setShowJson(false)}>
              <div className="json-modal" onClick={(e) => e.stopPropagation()}>
                <div className="json-modal-header">
                  <h3>Extracted JSON</h3>
                  <div className="json-modal-actions">
                    <button className="btn-icon" onClick={copyJson}>
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                      </svg>
                      Copy
                    </button>
                    <button className="btn-icon btn-close-modal" onClick={() => setShowJson(false)}>
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="18" y1="6" x2="6" y2="18" />
                        <line x1="6" y1="6" x2="18" y2="18" />
                      </svg>
                    </button>
                  </div>
                </div>
                <pre className="json-modal-body">
                  {JSON.stringify(soaJson, null, 2)}
                </pre>
              </div>
            </div>
          )}

          {/* Extracted Page Images */}
          {pageImages.length > 0 && (
            <div className="table-card" style={{ marginTop: '1.5rem' }}>
              <div className="table-header">
                <div className="table-meta">
                  <h3>Extracted Page Images</h3>
                  <span className="badge">{pageImages.length} page(s)</span>
                </div>
              </div>
              <div style={{ padding: '1rem', display: 'flex', flexDirection: 'column', gap: '1.5rem', alignItems: 'center' }}>
                {pageImages.map((img, idx) => (
                  <div key={idx} style={{ width: '100%', textAlign: 'center' }}>
                    <p style={{ margin: '0 0 0.5rem 0', fontWeight: 600, fontSize: '0.9rem', color: '#334155' }}>
                      Page {img.page_num}
                    </p>
                    <img
                      src={`data:${img.media_type};base64,${img.base64_data}`}
                      alt={`SOA page ${img.page_num}`}
                      style={{
                        maxWidth: '100%',
                        border: '1px solid #e2e8f0',
                        borderRadius: '8px',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
                      }}
                    />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Full Table View(s) */}
          {renderAllTables()}

          {/* Visits View */}
          {renderVisitsTable()}
        </section>
      )}
    </div>
  );
}

export default ImageExtractPage;
