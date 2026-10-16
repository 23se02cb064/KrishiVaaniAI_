import React from 'react';

const NAV_ITEMS = [
  { id: 'home', label: 'Dashboard', icon: '📊' },
  { id: 'disease', label: 'Disease Detection', icon: '🔬' },
  { id: 'chat', label: 'AI Assistant', icon: '💬' },
  { id: 'training', label: 'Model Training', icon: '🧪' },
];

export default function Sidebar({ onNavigate, theme, setTheme, currentPage, isDesktop = true, onClose }) {
  return (
    <aside
      className={`h-full flex flex-col ${isDesktop ? 'border-r' : ''} ${theme === 'dark' ? 'bg-slate-800' : 'bg-white'}`}
      aria-label="Main navigation"
    >
      <div className="h-full flex flex-col">
        <div className="p-5 border-b ${theme === 'dark' ? 'dark:border-slate-700' : 'border-slate-200'}">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-emerald-600 flex items-center justify-center text-white text-lg font-bold shadow-lg shadow-emerald-200">
              🌱
            </div>
            <div>
              <h1 className={`text-lg font-bold tracking-tight text-slate-900 dark:text-white ${isDesktop ? '' : 'truncate'}`}>KrishiVaaniAI</h1>
              {isDesktop && <p className="text-xs text-slate-500 dark:text-slate-400">Smart Agriculture</p>}
            </div>
          </div>
        </div>

        <nav className="flex-1 p-3 space-y-1" aria-label="Primary navigation">
          {NAV_ITEMS.map((item) => {
            const isActive = currentPage === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => {
                  onNavigate(item.id);
                  if (!isDesktop && onClose) onClose();
                }}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-emerald-50 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400'
                    : theme === 'dark'
                    ? 'text-slate-300 hover:bg-slate-700 hover:text-white'
                    : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
                }`}
              >
                <span className="text-base">{item.icon}</span>
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="p-4 border-t ${theme === 'dark' ? 'dark:border-slate-700' : 'border-slate-200'}">
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={theme === 'dark'}
              onChange={(e) => setTheme(e.target.checked ? 'dark' : 'light')}
              className="w-4 h-4 rounded border-slate-300 text-emerald-600 focus:ring-emerald-500"
            />
            <span className={`text-sm ${theme === 'dark' ? 'text-slate-300' : 'text-slate-600'}`}>Dark mode</span>
          </label>
        </div>
      </div>
    </aside>
  );
}
