import React from 'react';
import { Shield, Cpu, Database, Binary, Activity, Layers, Lock } from 'lucide-react';

export default function HeaderNav({ activeTab, setActiveTab, stats }) {
  const isHardhatLive = stats?.blockchainStatus?.isLiveNetwork;

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: Activity },
    { id: 'simulator', label: 'AI Simulator', icon: Cpu },
    { id: 'logs', label: 'Audit Records', icon: Database },
    { id: 'verification', label: 'Verification Lab', icon: Binary }
  ];

  return (
    <header className="sticky top-0 z-40 bg-[#0b0f19]/90 backdrop-blur-md border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand Logo & Title */}
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setActiveTab('dashboard')}>
            <div className="p-2 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl shadow-lg shadow-blue-500/20">
              <Shield className="w-6 h-6 text-white" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-slate-100 text-lg tracking-tight">Audit Bridge</span>
                <span className="text-[10px] uppercase font-mono px-2 py-0.5 bg-blue-500/10 text-blue-400 border border-blue-500/20 rounded-full font-semibold">
                  Component 03
                </span>
              </div>
              <p className="text-xs text-slate-400 font-mono">Privacy-Preserving AI-to-Smart-Contract</p>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="hidden md:flex space-x-1 bg-slate-900/60 p-1.5 rounded-xl border border-slate-800">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-xs font-medium transition-all duration-150 ${
                    isActive
                      ? 'bg-blue-600 text-white shadow-md shadow-blue-500/25'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </nav>

          {/* Web3 Network Badge & Reset Action */}
          <div className="flex items-center space-x-3">
            <div className={`hidden lg:flex items-center space-x-2 px-3 py-1.5 rounded-lg border text-xs font-mono ${
              isHardhatLive
                ? 'bg-emerald-950/40 text-emerald-300 border-emerald-500/30'
                : 'bg-purple-950/40 text-purple-300 border-purple-500/30'
            }`}>
              <Lock className="w-3.5 h-3.5" />
              <span>{isHardhatLive ? 'Hardhat Node (31337)' : 'Web3 Mock Adapter'}</span>
              <span className={`w-2 h-2 rounded-full ${isHardhatLive ? 'bg-emerald-400 animate-pulse' : 'bg-purple-400'}`}></span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
