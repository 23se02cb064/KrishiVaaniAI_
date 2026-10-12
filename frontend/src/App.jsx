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

  const PageComponent = PAGES[page] || Home;

  return (
    <div className={`flex h-screen overflow-hidden ${theme === 'dark' ? 'bg-slate-900 text-slate-100' : 'bg-slate-50 text-slate-900'}`}>
      <Sidebar onNavigate={setPage} theme={theme} setTheme={setTheme} currentPage={page} />
      <div className="flex-1 flex flex-col min-w-0">
        <Navbar page={page} />
        <main className={`flex-1 overflow-y-auto ${theme === 'dark' ? 'bg-slate-900' : 'bg-slate-50'}`}>
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
            <PageComponent />
          </div>
        </main>
      </div>
    </div>
  );
}
