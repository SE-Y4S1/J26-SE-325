import React, { useState } from 'react';
import { UserCheck, ShieldAlert, CheckCircle2, XCircle, AlertCircle, Cpu, ArrowRight } from 'lucide-react';
import { submitHumanReview } from '../services/api';

export default function HumanReviewModal({ pendingRecord, onClose, onComplete }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  if (!pendingRecord) return null;

  const handleDecision = async (decision) => {
    setLoading(true);
    setError(null);
    try {
      const res = await submitHumanReview(pendingRecord.transactionId, decision);
      if (res.success) {
        onComplete(res.record);
      } else {
        setError(res.message || 'Failed to submit review');
      }
    } catch (err) {
      setError(err.response?.data?.message || err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-fadeIn">
      <div className="glass-panel w-full max-w-xl rounded-2xl overflow-hidden shadow-2xl border border-amber-500/30">
        {/* Header */}
        <div className="bg-gradient-to-r from-amber-950/80 via-slate-900 to-amber-950/80 px-6 py-4 border-b border-amber-500/20 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-amber-500/20 text-amber-400 rounded-lg border border-amber-500/30">
              <UserCheck className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-100">Human Review Authorization Required</h3>
              <p className="text-xs text-amber-300 font-mono">Confidence-Aware Policy Triggered (P001 v1.0)</p>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="p-6 space-y-5">
          {error && (
            <div className="p-3 bg-rose-500/20 border border-rose-500/30 text-rose-300 text-xs rounded-lg flex items-center space-x-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* AI Decision Alert */}
          <div className="p-4 bg-amber-500/10 border border-amber-500/20 rounded-xl space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-amber-300 uppercase tracking-wide flex items-center space-x-1.5">
                <Cpu className="w-4 h-4" />
                <span>AI Recommendation: HUMAN REVIEW</span>
              </span>
              <span className="font-mono text-slate-400">Transaction #{pendingRecord.transactionId}</span>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              {pendingRecord.reason || 'High risk score with low model confidence requires human authorization before recording on-chain.'}
            </p>
          </div>

          {/* Metrics Grid */}
          <div className="grid grid-cols-2 gap-3 font-mono text-xs">
            <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
              <span className="text-slate-400 text-[11px] block">Risk Score</span>
              <span className="text-base font-bold text-rose-400">{pendingRecord.riskScore} / 100</span>
            </div>

            <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
              <span className="text-slate-400 text-[11px] block">Model Confidence</span>
              <span className="text-base font-bold text-amber-400">{pendingRecord.confidence}%</span>
            </div>

            <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
              <span className="text-slate-400 text-[11px] block">Transaction Amount</span>
              <span className="text-sm font-semibold text-slate-200">${pendingRecord.amount?.toLocaleString()}</span>
            </div>

            <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
              <span className="text-slate-400 text-[11px] block">Transaction Type</span>
              <span className="text-sm font-semibold text-blue-400">{pendingRecord.transactionType}</span>
            </div>
          </div>

          {/* Metadata Provenance */}
          <div className="text-[11px] font-mono text-slate-400 bg-slate-900/40 p-3 rounded-lg border border-slate-800 flex justify-between">
            <span>Model: <strong className="text-slate-200">{pendingRecord.modelVersion}</strong></span>
            <span>Policy: <strong className="text-slate-200">{pendingRecord.policyId} (v{pendingRecord.policyVersion})</strong></span>
          </div>

          {/* Review Buttons */}
          <div className="pt-2 flex items-center space-x-3">
            <button
              disabled={loading}
              onClick={() => handleDecision('APPROVE')}
              className="flex-1 flex items-center justify-center space-x-2 py-3 px-4 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white font-semibold rounded-xl transition-all shadow-lg shadow-emerald-600/25 text-xs"
            >
              <CheckCircle2 className="w-4 h-4" />
              <span>Approve Transaction</span>
            </button>

            <button
              disabled={loading}
              onClick={() => handleDecision('REJECT')}
              className="flex-1 flex items-center justify-center space-x-2 py-3 px-4 bg-rose-600 hover:bg-rose-500 disabled:opacity-50 text-white font-semibold rounded-xl transition-all shadow-lg shadow-rose-600/25 text-xs"
            >
              <XCircle className="w-4 h-4" />
              <span>Reject Transaction</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
