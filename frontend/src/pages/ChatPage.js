import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';

const API_URL = 'http://localhost:5000/api';
const STORAGE_KEY = 'soa_chat_history';

const DEFAULT_MESSAGE = {
  role: 'assistant',
  content: 'Hello! I can help you extract the Schedule of Assessments (SOA) table from the ingested protocol. Try asking me:\n\n• "Print the Schedule of Assessments"\n• "Show me the SOA table"\n• "What visits are in the study?"',
  isHtml: false,
};

function loadChatHistory() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      if (Array.isArray(parsed) && parsed.length > 0) return parsed;
    }
  } catch (e) {
    // ignore corrupt data
  }
  return [DEFAULT_MESSAGE];
}

function ChatPage() {
  const [messages, setMessages] = useState(loadChatHistory);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [jsonModal, setJsonModal] = useState(null); // holds JSON data for popup
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
    } catch (e) {
      // storage full or unavailable
    }
  }, [messages]);

  const clearChat = () => {
    setMessages([DEFAULT_MESSAGE]);
    localStorage.removeItem(STORAGE_KEY);
  };

  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed || loading) return;

    const userMsg = { role: 'user', content: trimmed, isHtml: false };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const response = await axios.post(
        `${API_URL}/chat`,
        { message: trimmed },
        { timeout: 180000 }
      );

      const data = response.data;
      if (data.error) {
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: data.error, isHtml: false, soaJson: null },
        ]);
      } else {
        const hasHtml = /<table[\s>]/i.test(data.response);
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: data.response,
            isHtml: hasHtml,
            soaJson: data.soa_json || null,
          },
        ]);
      }
    } catch (err) {
      const errMsg =
        err.response?.data?.error ||
        'Failed to get a response. Make sure the backend is running.';
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: errMsg, isHtml: false },
      ]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const copyContent = (content) => {
    navigator.clipboard.writeText(content);
  };

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`chat-msg ${msg.role}`}>
            <div className="chat-msg-header">
              <span className="chat-role">
                {msg.role === 'user' ? 'You' : 'SOA Assistant'}
              </span>
              {msg.role === 'assistant' && idx > 0 && (
                <div className="chat-msg-actions">
                  {msg.soaJson && (
                    <button
                      className="btn-icon btn-json"
                      onClick={() => setJsonModal(msg.soaJson)}
                      title="View JSON"
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M4 7V4a2 2 0 0 1 2-2h8.5L20 7.5V20a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-3" />
                        <polyline points="14 2 14 8 20 8" />
                        <path d="M5 12h10" />
                        <path d="M5 16h7" />
                      </svg>
                      View JSON
                    </button>
                  )}
                  <button
                    className="btn-icon"
                    onClick={() => copyContent(msg.content)}
                    title="Copy response"
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                    </svg>
                    Copy
                  </button>
                </div>
              )}
            </div>
            {msg.isHtml ? (
              <div
                className="chat-msg-body chat-html-content"
                dangerouslySetInnerHTML={{ __html: msg.content }}
              />
            ) : (
              <div className="chat-msg-body">
                {msg.content.split('\n').map((line, li) => (
                  <React.Fragment key={li}>
                    {line}
                    {li < msg.content.split('\n').length - 1 && <br />}
                  </React.Fragment>
                ))}
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="chat-msg assistant">
            <div className="chat-msg-header">
              <span className="chat-role">SOA Assistant</span>
            </div>
            <div className="chat-msg-body chat-loading">
              <span className="dot-pulse"></span>
              Retrieving from Pinecone & generating with Claude...
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-area">
        <div className="chat-input-actions">
          <button className="btn-clear-chat" onClick={clearChat} disabled={loading}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="3 6 5 6 21 6" />
              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
            </svg>
            Clear Chat
          </button>
        </div>
        <div className="chat-input-wrapper">
          <textarea
            ref={inputRef}
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about the protocol or type 'Print the SOA table'..."
            rows={1}
            disabled={loading}
          />
          <button
            className="chat-send-btn"
            onClick={handleSend}
            disabled={!input.trim() || loading}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </button>
        </div>
      </div>
      {/* JSON Modal */}
      {jsonModal && (
        <div className="json-modal-overlay" onClick={() => setJsonModal(null)}>
          <div className="json-modal" onClick={(e) => e.stopPropagation()}>
            <div className="json-modal-header">
              <h3>SOA Table — JSON</h3>
              <div className="json-modal-actions">
                <button
                  className="btn-icon"
                  onClick={() => {
                    navigator.clipboard.writeText(JSON.stringify(jsonModal, null, 2));
                  }}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                  </svg>
                  Copy JSON
                </button>
                <button className="btn-icon btn-close-modal" onClick={() => setJsonModal(null)}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18" />
                    <line x1="6" y1="6" x2="18" y2="18" />
                  </svg>
                </button>
              </div>
            </div>
            <pre className="json-modal-body">
              {JSON.stringify(jsonModal, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}

export default ChatPage;
