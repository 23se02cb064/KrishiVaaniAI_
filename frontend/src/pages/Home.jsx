import React, { useEffect, useState } from 'react';

export default function Home() {
  const [history, setHistory] = useState([]);
  const [analytics, setAnalytics] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const load = async () => {
      try {
        const [historyRes, analyticsRes] = await Promise.all([
          fetch(`${import.meta.env.VITE_API_URL || ''}/api/predictions?limit=6`),
	    fetch(`${import.meta.env.VITE_API_URL || ''}/api/predictions/analytics`),
        ]);
        const historyData = await historyRes.json();
        const analyticsData = await analyticsRes.json();
        setHistory(historyData.predictions || []);
        setAnalytics(analyticsData.series || []);
      } catch (err) {
        setError(err.message || 'Failed to load history. Please try again.');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {error && (
          <div className="card p-4 bg-red-50 border border-red-200 rounded-xl text-sm text-red-700 mb-4">
            {error}
          </div>
        )}

        <div className="md:col-span-2 lg:col-span-2 card p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold text-slate-900">Recent Predictions</h3>
            <p className="text-sm text-slate-500 mt-1">Your latest disease checks for quick follow-up.</p>
          </div>
        </div>
        <div className="space-y-3">
          {loading ? (
            <div className="text-sm text-slate-500">Loading history...</div>
          ) : history.length === 0 ? (
            <div className="text-sm text-slate-500">No predictions yet. Go to Disease Detection to analyze a leaf.</div>
          ) : (
            history.map((item) => (
              <div key={item.id} className="p-4 bg-slate-50 rounded-xl border border-slate-200">
                <div className="flex items-center justify-between">
                  <span className="font-medium text-slate-900">{item.class_name}</span>
                  <span className="badge badge-success">{Math.round(item.confidence * 100)}%</span>
                </div>
                {item.remedy && <p className="mt-2 text-sm text-slate-600 line-clamp-2">{item.remedy}</p>}
              </div>
            ))
          )}
        </div>
      </div>

      <div className="space-y-6">
        <div className="card p-6">
          <h3 className="text-base font-semibold text-slate-900 mb-3">Activity Trend</h3>
          <div className="space-y-2">
            {analytics.length === 0 ? (
              <p className="text-sm text-slate-500">No analytics yet.</p>
            ) : (
              analytics.map((point) => (
                <div key={point.label} className="flex items-center justify-between text-sm">
                  <span className="text-slate-600">{point.label}</span>
                  <span className="font-medium text-slate-900">{point.count} checks</span>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="card p-6">
          <h3 className="text-base font-semibold text-slate-900 mb-3">Quick Actions</h3>
          <ul className="space-y-2 text-sm text-slate-600">
            <li>• Upload a leaf image for instant diagnosis</li>
            <li>• Chat with AI for crop guidance</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
