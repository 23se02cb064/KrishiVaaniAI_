import React, { useState } from 'react';

export default function Chat() {
  const [query, setQuery] = useState('');
  const [reply, setReply] = useState('');
  const [loading, setLoading] = useState(false);

  const send = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setReply('');
    try {
      const data = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim(), lang: 'en' }),
      }).then((r) => r.json());
      setReply(data.reply || 'No response received.');
    } catch {
      setReply('Unable to get response. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl space-y-6">
      <div className="card p-6">
        <h3 className="text-lg font-semibold text-slate-900 mb-1">AI Agricultural Assistant</h3>
        <p className="text-sm text-slate-500 mb-4">Ask questions about crops, diseases, weather, or market prices.</p>
        <div className="flex gap-3">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Type your question here..."
            className="flex-1 p-3 border border-slate-300 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent resize-none"
            rows={3}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
          />
          <button onClick={send} disabled={loading || !query.trim()} className="btn btn-primary self-end">
            {loading ? 'Sending...' : 'Send'}
          </button>
        </div>
      </div>

      {reply && (
        <div className="card p-6">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-xl">🤖</span>
            <h4 className="text-base font-semibold text-slate-900">Response</h4>
          </div>
          <p className="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap">{reply}</p>
        </div>
      )}
    </div>
  );
}
