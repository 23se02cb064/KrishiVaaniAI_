import React from 'react';

const PAGE_TITLES = {
  home: 'Dashboard',
  disease: 'Disease Detection',
  chat: 'AI Assistant',
  training: 'Model Training',
};

export default function Navbar({ page, toggleMenu, sidebarOpen }) {
  const title = PAGE_TITLES[page] || 'Dashboard';

  return (
    <header className="bg-white border-b border-slate-200 px-4 sm:px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {toggleMenu && (
            <button
              type="button"
              className="lg:hidden p-2 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-500"
              aria-label={sidebarOpen ? 'Close navigation menu' : 'Open navigation menu'}
              aria-expanded={sidebarOpen}
              onClick={toggleMenu}
            >
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                {sidebarOpen ? (
                  <>
                    <line x1="18" y1="6" x2="6" y2="18" />
                    <line x1="6" y1="6" x2="18" y2="18" />
                  </>
                ) : (
                  <>
                    <line x1="3" y1="12" x2="21" y2="12" />
                    <line x1="3" y1="6" x2="21" y2="6" />
                    <line x1="3" y1="18" x2="21" y2="18" />
                  </>
                )}
              </svg>
            </button>
          )}
          <div>
            <h2 className={`text-xl font-semibold text-slate-900 dark:text-white ${!toggleMenu ? 'ml-4' : ''}`}>{title}</h2>
            <p className={`text-sm text-slate-500 dark:text-slate-400 ${!toggleMenu ? 'ml-4' : ''}`}>KrishiVaaniAI — Intelligent Crop Management</p>
          </div>
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
