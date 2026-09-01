import React, { useEffect, useState } from 'react';
import { Activity, ShieldCheck, ShieldAlert, UserCheck, Binary, AlertTriangle, ArrowRight, Database, RefreshCw, Trash2 } from 'lucide-react';
import MetricCard from '../components/MetricCard';
import { fetchDashboardStats, fetchAuditRecords, resetDemoData } from '../services/api';

export default function Dashboard({ setActiveTab, setSelectedTxForVerify }) {
  const [stats, setStats] = useState(null);
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [resetting, setResetting] = useState(false);

  const loadData = async () => {
    setLoading(true);
    try {
      const statsRes = await fetchDashboardStats();
      const recordsRes = await fetchAuditRecords();
      setStats(statsRes.stats);
      setRecords(recordsRes.records || []);
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleResetData = async () => {
    if (window.confirm("Are you sure you want to clear all off-chain demo records to reset for a fresh evaluation run?")) {
      setResetting(true);
      try {
        await resetDemoData();
        await loadData();
      } catch (err) {
        console.error('Failed to reset demo data:', err);
      } finally {
        setResetting(false);
      }
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleVerifyClick = (txId) => {
    setSelectedTxForVerify(txId);
    setActiveTab('verification');
  };

  return (
    <div className="space-y-6">
      {/* Hero Welcome Banner */}
      <div className="glass-panel p-6 rounded-2xl relative overflow-hidden bg-gradient-to-r from-blue-950/40 via-slate-900/80 to-purple-950/40 border border-blue-500/20">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center space-x-2">
              <span className="px-2.5 py-1 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 text-xs font-mono font-semibold">
                Component 03 Specification
              </span>
              <span className="text-slate-400">•</span>
              <span className="text-xs text-slate-400 font-mono">University Research Prototype</span>
            </div>
            <h1 className="text-2xl font-black text-slate-100 mt-2 tracking-tight">
              Privacy-Preserving AI-to-Smart-Contract Audit Bridge
            </h1>
            <p className="text-xs text-slate-300 max-w-2xl mt-1 leading-relaxed">
              Demonstrating risk & confidence-aware policy evaluation, off-chain SHA-256 canonical hashing, on-chain smart contract audit commitments, and cryptographic integrity verification.
            </p>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={handleResetData}
              disabled={resetting}
              className="flex items-center space-x-1.5 px-3 py-2.5 bg-rose-950/60 hover:bg-rose-900/80 text-rose-300 border border-rose-500/30 rounded-xl transition-all text-xs font-mono"
              title="Clear Demo Records"
            >
              <Trash2 className="w-4 h-4 text-rose-400" />
              <span>{resetting ? 'Resetting...' : 'Reset Demo Data'}</span>
            </button>

            <button
              onClick={loadData}
              className="p-2.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl transition-all border border-slate-700"
              title="Refresh Data"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            </button>

            <button
              onClick={() => setActiveTab('simulator')}
              className="flex items-center space-x-2 px-4 py-2.5 bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold rounded-xl transition-all shadow-lg shadow-blue-600/25"
            >
              <span>Launch AI Simulator</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <MetricCard
          title="Total Decisions"
          value={stats?.totalDecisions || 0}
          icon={Activity}
          color="blue"
          subtext="Evaluated by Bridge"
        />
        <MetricCard
          title="Approved"
          value={stats?.approved || 0}
          icon={ShieldCheck}
          color="emerald"
          subtext="Automated / Human"
        />
        <MetricCard
          title="Rejected"
          value={stats?.rejected || 0}
          icon={ShieldAlert}
          color="rose"
          subtext="Policy Interventions"
        />
        <MetricCard
          title="Human Review"
          value={stats?.humanReview || 0}
          icon={UserCheck}
          color="amber"
          subtext="Low Conf Triggered"
        />
        <MetricCard
          title="Verified"
          value={stats?.verified || 0}
          icon={Binary}
          color="purple"
          subtext="Hash Matches Chain"
        />
        <MetricCard
          title="Tamper Detected"
          value={stats?.integrityFailures || 0}
          icon={AlertTriangle}
          color="rose"
          subtext="Cryptographic Failures"
        />
      </div>

      {/* Recent Audit Records Section */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2">
            <Database className="w-5 h-5 text-blue-400" />
            <h2 className="text-base font-bold text-slate-100">Recent Audit Records & On-Chain Commitments</h2>
          </div>
          <button
            onClick={() => setActiveTab('logs')}
            className="text-xs text-blue-400 hover:text-blue-300 font-semibold flex items-center space-x-1"
          >
            <span>View All Records ({records.length})</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        {records.length === 0 ? (
          <div className="text-center py-10 text-slate-500 font-mono text-xs">
            No audit records created yet. Click "Launch AI Simulator" to test scenarios.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 uppercase text-[10px]">
                  <th className="pb-3 px-3">Transaction ID</th>
                  <th className="pb-3 px-3">Decision</th>
                  <th className="pb-3 px-3">Risk / Conf</th>
                  <th className="pb-3 px-3">Provenance (Model/Policy)</th>
                  <th className="pb-3 px-3">SHA-256 Commitment Hash</th>
                  <th className="pb-3 px-3 text-right">Verification</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {records.slice(0, 6).map((rec) => (
                  <tr key={rec.transactionId} className="hover:bg-slate-800/30 transition-colors">
                    <td className="py-3.5 px-3 font-semibold text-slate-200">{rec.transactionId}</td>
                    <td className="py-3.5 px-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${
                        rec.finalDecision.includes('APPROVE')
                          ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
                          : rec.finalDecision.includes('REJECT')
                          ? 'bg-rose-500/20 text-rose-300 border-rose-500/30'
                          : 'bg-amber-500/20 text-amber-300 border-amber-500/30'
                      }`}>
                        {rec.finalDecision}
                      </span>
                    </td>
                    <td className="py-3.5 px-3 text-slate-300">
                      R: <span className="text-rose-400 font-bold">{rec.riskScore}</span> | C: <span className="text-amber-400 font-bold">{rec.confidence}%</span>
                    </td>
                    <td className="py-3.5 px-3 text-slate-400 text-[11px]">
                      {rec.modelVersion} / {rec.policyId}-v{rec.policyVersion}
                    </td>
                    <td className="py-3.5 px-3 text-purple-300 text-[11px]">
                      {rec.hash?.substring(0, 16)}...
                    </td>
                    <td className="py-3.5 px-3 text-right">
                      <button
                        onClick={() => handleVerifyClick(rec.transactionId)}
                        className="px-2.5 py-1 bg-slate-800 hover:bg-blue-600 text-slate-200 hover:text-white rounded-lg transition-all text-[11px] font-sans font-medium"
                      >
                        Verify Hash
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
