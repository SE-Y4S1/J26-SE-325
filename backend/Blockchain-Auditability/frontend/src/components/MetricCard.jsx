import React from 'react';

export default function MetricCard({ title, value, icon: Icon, color = 'blue', subtext }) {
  const colorStyles = {
    blue: 'border-blue-500/20 bg-blue-950/20 text-blue-400',
    emerald: 'border-emerald-500/20 bg-emerald-950/20 text-emerald-400',
    rose: 'border-rose-500/20 bg-rose-950/20 text-rose-400',
    amber: 'border-amber-500/20 bg-amber-950/20 text-amber-400',
    purple: 'border-purple-500/20 bg-purple-950/20 text-purple-400'
  };

  return (
    <div className="glass-card p-5 rounded-2xl relative overflow-hidden">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">{title}</p>
          <h4 className="text-2xl font-extrabold text-slate-100 mt-1 font-mono tracking-tight">{value}</h4>
          {subtext && <p className="text-[11px] text-slate-400 mt-1 font-mono">{subtext}</p>}
        </div>
        <div className={`p-3 rounded-xl border ${colorStyles[color]}`}>
          <Icon className="w-5 h-5" />
        </div>
      </div>
    </div>
  );
}
