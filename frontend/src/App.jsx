import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Disease from './pages/Disease';
import Chat from './pages/Chat';
import Training from './pages/Training';

const PAGES = {
  home: Home,
  disease: Disease,
  chat: Chat,
  training: Training,
};

export default function App() {
  const [page, setPage] = useState('home');
  const [theme, setTheme] = useState('light');
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const PageComponent = PAGES[page] || Home;

  return (
    <div className={`min-h-screen ${theme === 'dark' ? 'bg-slate-900 text-slate-100' : 'bg-slate-50 text-slate-900'}`}>
      {/* Desktop sidebar: always visible, hidden on mobile */}
      <aside
        className={`hidden lg:block w-64 flex-shrink-0 border-r ${
          theme === 'dark' ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200'
        }`}
      >
        <Sidebar
          onNavigate={setPage}
          theme={theme}
          setTheme={setTheme}
          currentPage={page}
          isDesktop={true}
        />
      </aside>

      {/* Mobile sidebar drawer: overlay, hidden on desktop */}
      <aside
        className={`fixed inset-y-0 left-0 z-50 w-64 transform transition-transform duration-300 ease-in-out ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        } lg:hidden ${
          theme === 'dark' ? 'bg-slate-800' : 'bg-white'
        }`}
      >
        <Sidebar
          onNavigate={(p) => { setPage(p); setSidebarOpen(false); }}
          theme={theme}
          setTheme={setTheme}
          currentPage={page}
          isDesktop={false}
          onClose={() => setSidebarOpen(false)}
        />
      </aside>

      {/* Mobile backdrop */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 lg:hidden"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Main content area */}
      <div className="flex-1 min-w-0">
        <Navbar page={page} toggleMenu={() => setSidebarOpen(!sidebarOpen)} sidebarOpen={sidebarOpen} />
        <main className={`flex-1 overflow-y-auto min-h-[calc(100vh-64px)] ${theme === 'dark' ? 'bg-slate-900' : 'bg-slate-50'}`}>
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
            <PageComponent />
          </div>
        </main>
      </div>
    </div>
  );
}
