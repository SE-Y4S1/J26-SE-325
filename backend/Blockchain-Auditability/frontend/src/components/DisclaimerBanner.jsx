import React from 'react';
import { AlertTriangle, Info, ShieldCheck } from 'lucide-react';

export default function DisclaimerBanner() {
  return (
    <div className="bg-gradient-to-r from-blue-950/80 via-slate-900/90 to-purple-950/80 border-b border-blue-500/20 px-4 py-2.5 text-xs text-slate-300">
      <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center space-x-2">
          <span className="flex h-2 w-2 relative">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
          </span>
          <span className="font-semibold text-amber-300 uppercase tracking-wider text-[11px]">Research Demonstration</span>
          <span className="text-slate-400">|</span>
          <span className="font-medium text-slate-200">
            <span className="bg-blue-500/20 text-blue-300 px-2 py-0.5 rounded border border-blue-500/30 font-mono text-[11px] mr-1.5">Mock AI Decision Input</span>
            Demonstrating AI-to-Smart-Contract Privacy Bridge & Tamper-Evident Off-Chain Verification
          </span>
        </div>
        <div className="flex items-center space-x-3 text-[11px] text-slate-400">
          <span className="flex items-center space-x-1">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            <span>Policy: P001 v1.0</span>
          </span>
          <span>•</span>
          <span className="font-mono text-purple-300">SHA-256 Deterministic</span>
        </div>
      </div>
    </div>
  );
}
