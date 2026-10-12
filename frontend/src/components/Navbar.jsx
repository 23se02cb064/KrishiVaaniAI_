import React from 'react';

const PAGE_TITLES = {
  home: 'Dashboard',
  disease: 'Disease Detection',
  chat: 'AI Assistant',
  training: 'Model Training',
};

export default function Navbar({ page }) {
  const title = PAGE_TITLES[page] || 'Dashboard';

  return (
    <header className="bg-white border-b border-slate-200 px-6 py-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-slate-900">{title}</h2>
          <p className="text-sm text-slate-500">KrishiVaaniAI — Intelligent Crop Management</p>
        </div>
        <div className="flex items-center gap-3">
          <span className="badge badge-success">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            System Online
          </span>
        </div>
      </div>
    </header>
  );
}
