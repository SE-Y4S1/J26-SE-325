import React from 'react';
import { PlayCircle, ShieldCheck, ShieldAlert, UserCheck } from 'lucide-react';

export default function ScenarioPresetBar({ onSelectScenario }) {
  const scenarios = [
    {
      id: 1,
      title: 'Scenario 1: Low Risk (Approve)',
      badge: 'APPROVE',
      badgeColor: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
      icon: ShieldCheck,
      data: {
        transactionId: `TX-APPR-${Math.floor(100 + Math.random() * 900)}`,
        riskScore: 25,
        confidence: 90,
        amount: 2500,
        transactionType: 'TRANSFER',
        modelVersion: 'FraudModel-v2'
      },
      desc: 'Risk=25, Conf=90. Automated approval flow.'
    },
    {
      id: 2,
      title: 'Scenario 2: High Risk & High Conf (Reject)',
      badge: 'REJECT',
      badgeColor: 'bg-rose-500/20 text-rose-300 border-rose-500/30',
      icon: ShieldAlert,
      data: {
        transactionId: `TX-REJ-${Math.floor(100 + Math.random() * 900)}`,
        riskScore: 92,
        confidence: 94,
        amount: 85000,
        transactionType: 'WITHDRAWAL',
        modelVersion: 'FraudModel-v2'
      },
      desc: 'Risk=92, Conf=94. Automated rejection flow.'
    },
    {
      id: 3,
      title: 'Scenario 3: High Risk & Low Conf (Human Review)',
      badge: 'HUMAN REVIEW',
      badgeColor: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
      icon: UserCheck,
      data: {
        transactionId: `TX-REV-${Math.floor(100 + Math.random() * 900)}`,
        riskScore: 92,
        confidence: 52,
        amount: 45000,
        transactionType: 'LOAN_DISBURSEMENT',
        modelVersion: 'FraudModel-v2'
      },
      desc: 'Risk=92, Conf=52. Triggers confidence-aware human review.'
    }
  ];

  return (
    <div className="glass-panel p-4 rounded-xl mb-6">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <PlayCircle className="w-4 h-4 text-blue-400" />
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Research Evaluation Presets
          </h3>
        </div>
        <span className="text-[11px] text-slate-400 font-mono">Click any scenario to pre-fill test parameters</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {scenarios.map((sc) => {
          const Icon = sc.icon;
          return (
            <button
              key={sc.id}
              onClick={() => onSelectScenario(sc.data)}
              className="group text-left p-3 rounded-lg bg-slate-900/70 border border-slate-800 hover:border-blue-500/50 hover:bg-slate-800/60 transition-all duration-150 relative overflow-hidden"
            >
              <div className="flex items-start justify-between mb-1.5">
                <div className="flex items-center space-x-2">
                  <Icon className="w-4 h-4 text-slate-400 group-hover:text-blue-400 transition-colors" />
                  <span className="text-xs font-semibold text-slate-200 group-hover:text-white">
                    Scenario 0{sc.id}
                  </span>
                </div>
                <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${sc.badgeColor}`}>
                  {sc.badge}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 leading-snug group-hover:text-slate-300">
                {sc.desc}
              </p>
            </button>
          );
        })}
      </div>
    </div>
  );
}
