import React, { useState, useEffect } from 'react';
import HeaderNav from './components/HeaderNav';
import DisclaimerBanner from './components/DisclaimerBanner';
import Dashboard from './pages/Dashboard';
import SimulatorPage from './pages/SimulatorPage';
import AuditLogsPage from './pages/AuditLogsPage';
import VerificationPage from './pages/VerificationPage';
import { fetchDashboardStats } from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [selectedTxForVerify, setSelectedTxForVerify] = useState('');
  const [stats, setStats] = useState(null);

  const loadStats = async () => {
    try {
      const res = await fetchDashboardStats();
      if (res.success) {
        setStats(res.stats);
      }
    } catch (err) {
      console.error('Failed to load stats in App:', err);
    }
  };

  useEffect(() => {
    loadStats();
  }, [activeTab]);

  return (
    <div className="min-h-screen flex flex-col bg-[#0b0f19] text-slate-100 font-sans">
      {/* Top Disclaimer Banner for evaluators */}
      <DisclaimerBanner />

      {/* Main Header & Nav */}
      <HeaderNav activeTab={activeTab} setActiveTab={setActiveTab} stats={stats} />

      {/* View Content Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'dashboard' && (
          <Dashboard
            setActiveTab={setActiveTab}
            setSelectedTxForVerify={setSelectedTxForVerify}
          />
        )}

        {activeTab === 'simulator' && (
          <SimulatorPage
            onDecisionComplete={() => {
              loadStats();
            }}
          />
        )}

        {activeTab === 'logs' && (
          <AuditLogsPage
            setActiveTab={setActiveTab}
            setSelectedTxForVerify={setSelectedTxForVerify}
          />
        )}

        {activeTab === 'verification' && (
          <VerificationPage
            selectedTxForVerify={selectedTxForVerify}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/60 py-4 px-4 text-center text-xs text-slate-500 font-mono">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>Component 03 Prototype • Privacy-Preserving AI-to-Smart-Contract Audit Bridge</span>
          <span>SHA-256 Hash Engine • Solidity AuditRegistry • Hardhat / Ethers.js</span>
        </div>
      </footer>
    </div>
  );
}
