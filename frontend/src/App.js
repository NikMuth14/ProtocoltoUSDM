import React from 'react';
import ImageExtractPage from './pages/ImageExtractPage';
import './App.css';

function App() {
  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="16" y1="13" x2="8" y2="13" />
              <line x1="16" y1="17" x2="8" y2="17" />
              <polyline points="10 9 9 9 8 9" />
            </svg>
          </div>
          <div>
            <h1>SOA Table Extractor</h1>
            <p className="subtitle">Upload a protocol PDF to automatically extract the SOA table as structured JSON</p>
          </div>
        </div>
      </header>

      <main className="main">
        <ImageExtractPage />
      </main>
    </div>
  );
}

export default App;
