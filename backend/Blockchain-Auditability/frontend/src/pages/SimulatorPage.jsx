import React, { useState } from 'react';
import { Cpu, Send, AlertCircle, ShieldCheck, ShieldAlert, UserCheck, Lock, Hash, CheckCircle2 } from 'lucide-react';
import ScenarioPresetBar from '../components/ScenarioPresetBar';
import HumanReviewModal from '../components/HumanReviewModal';
import { evaluateAIDecision } from '../services/api';

export default function SimulatorPage({ onDecisionComplete }) {
  const [formData, setFormData] = useState({
    transactionId: 'TX1001',
    riskScore: 92,
    confidence: 94,
    amount: 5000,
    transactionType: 'TRANSFER',
    modelVersion: 'FraudModel-v2'
  });

  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState([]);
  const [result, setResult] = useState(null);
  const [pendingRecord, setPendingRecord] = useState(null);

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) || 0 : value
    }));
  };

  const handleScenarioSelect = (presetData) => {
    setFormData(presetData);
    setErrors([]);
    setResult(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setErrors([]);
    setResult(null);
    setPendingRecord(null);

    try {
      const res = await evaluateAIDecision(formData);
      if (res.success) {
        if (res.status === 'HUMAN_REVIEW_REQUIRED') {
          setPendingRecord(res.record);
        } else {
          setResult(res);
          if (onDecisionComplete) onDecisionComplete();
        }
      }
    } catch (err) {
      if (err.response?.data?.errors) {
        setErrors(err.response.data.errors);
      } else {
        setErrors([err.response?.data?.message || err.message]);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleHumanReviewComplete = (updatedRecord) => {
    setPendingRecord(null);
    setResult({
      success: true,
      status: 'HUMAN_REVIEW_COMPLETED',
      policyResult: {
        action: updatedRecord.finalDecision,
        reason: 'Human reviewer rendered decision for low-confidence policy trigger.',
        policyId: updatedRecord.policyId,
        policyVersion: updatedRecord.policyVersion
      },
      record: updatedRecord
    });
    if (onDecisionComplete) onDecisionComplete();
  };

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-black text-slate-100 flex items-center space-x-2">
            <Cpu className="w-6 h-6 text-blue-400" />
            <span>AI Decision Simulator</span>
          </h1>
          <p className="text-xs text-slate-400 font-mono mt-1">
            Simulate mock financial risk signals from upstream AI model components & translate into policy commitments.
          </p>
        </div>
      </div>

      {/* Preset Bar */}
      <ScenarioPresetBar onSelectScenario={handleScenarioSelect} />

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Form Column */}
        <div className="lg:col-span-7 space-y-6">
          <form onSubmit={handleSubmit} className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-5">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                1. Input Mock AI Risk Parameters
              </span>
              <span className="text-[11px] text-amber-400 font-mono font-medium bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20">
                Mock AI Decision Input
              </span>
            </div>

            {errors.length > 0 && (
              <div className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl space-y-1">
                <div className="flex items-center space-x-2 text-rose-400 font-semibold text-xs">
                  <AlertCircle className="w-4 h-4" />
                  <span>Input Validation Errors</span>
                </div>
                <ul className="list-disc list-inside text-xs text-rose-300 font-mono space-y-0.5 pl-1">
                  {errors.map((err, i) => (
                    <li key={i}>{err}</li>
                  ))}
                </ul>
              </div>
            )}

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs font-mono">
              <div>
                <label className="block text-slate-400 mb-1">Transaction ID</label>
                <input
                  type="text"
                  name="transactionId"
                  value={formData.transactionId}
                  onChange={handleChange}
                  placeholder="e.g. TX1001"
                  className="w-full bg-slate-900 border border-slate-800 focus:border-blue-500 rounded-lg px-3 py-2.5 text-slate-100 focus:outline-none transition-colors"
                />
              </div>

              <div>
                <label className="block text-slate-400 mb-1">Transaction Type</label>
                <select
                  name="transactionType"
                  value={formData.transactionType}
                  onChange={handleChange}
                  className="w-full bg-slate-900 border border-slate-800 focus:border-blue-500 rounded-lg px-3 py-2.5 text-slate-100 focus:outline-none transition-colors"
                >
                  <option value="TRANSFER">TRANSFER</option>
                  <option value="PAYMENT">PAYMENT</option>
                  <option value="WITHDRAWAL">WITHDRAWAL</option>
                  <option value="LOAN_DISBURSEMENT">LOAN_DISBURSEMENT</option>
                </select>
              </div>

              <div>
                <label className="block text-slate-400 mb-1">Risk Score (0 - 100)</label>
                <input
                  type="number"
                  name="riskScore"
                  min="0"
                  max="100"
                  value={formData.riskScore}
                  onChange={handleChange}
                  className="w-full bg-slate-900 border border-slate-800 focus:border-blue-500 rounded-lg px-3 py-2.5 text-slate-100 focus:outline-none transition-colors font-bold text-rose-400"
                />
              </div>

              <div>
                <label className="block text-slate-400 mb-1">Confidence Score (0 - 100)</label>
                <input
                  type="number"
                  name="confidence"
                  min="0"
                  max="100"
                  value={formData.confidence}
                  onChange={handleChange}
                  className="w-full bg-slate-900 border border-slate-800 focus:border-blue-500 rounded-lg px-3 py-2.5 text-slate-100 focus:outline-none transition-colors font-bold text-amber-400"
                />
              </div>

              <div>
                <label className="block text-slate-400 mb-1">Amount ($ USD)</label>
                <input
                  type="number"
                  name="amount"
                  min="1"
                  value={formData.amount}
                  onChange={handleChange}
                  className="w-full bg-slate-900 border border-slate-800 focus:border-blue-500 rounded-lg px-3 py-2.5 text-slate-100 focus:outline-none transition-colors"
                />
              </div>

              <div>
                <label className="block text-slate-400 mb-1">Model Version Provenance</label>
                <input
                  type="text"
                  name="modelVersion"
                  value={formData.modelVersion}
                  onChange={handleChange}
                  className="w-full bg-slate-900 border border-slate-800 focus:border-blue-500 rounded-lg px-3 py-2.5 text-slate-100 focus:outline-none transition-colors"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full py-3.5 px-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold text-xs rounded-xl shadow-lg shadow-blue-600/25 transition-all flex items-center justify-center space-x-2"
            >
              {loading ? (
                <span>Evaluating Policy & Generating SHA-256 Hash...</span>
              ) : (
                <>
                  <Send className="w-4 h-4" />
                  <span>Evaluate Decision & Commit Audit</span>
                </>
              )}
            </button>
          </form>
        </div>

        {/* Evaluation Output Column */}
        <div className="lg:col-span-5 space-y-6">
          <div className="glass-panel p-6 rounded-2xl border border-slate-800 h-full flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                  2. Policy Engine & Audit Output
                </span>
                <span className="text-xs text-slate-400 font-mono">Policy P001 v1.0</span>
              </div>

              {!result ? (
                <div className="text-center py-16 text-slate-500 space-y-2">
                  <Lock className="w-8 h-8 mx-auto opacity-30" />
                  <p className="text-xs font-mono">Submit parameters to trigger automated policy evaluation</p>
                </div>
              ) : (
                <div className="space-y-4 font-mono text-xs">
                  {/* Decision Badge */}
                  <div className={`p-4 rounded-xl border flex items-start space-x-3 ${
                    result.record.finalDecision.includes('APPROVE')
                      ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
                      : result.record.finalDecision.includes('REJECT')
                      ? 'bg-rose-500/10 border-rose-500/30 text-rose-300'
                      : 'bg-amber-500/10 border-amber-500/30 text-amber-300'
                  }`}>
                    <CheckCircle2 className="w-5 h-5 shrink-0 mt-0.5" />
                    <div>
                      <h4 className="font-bold text-sm">Action: {result.record.finalDecision}</h4>
                      <p className="text-[11px] mt-1 opacity-90 leading-relaxed">
                        {result.policyResult?.reason || result.record.reason}
                      </p>
                    </div>
                  </div>

                  {/* Hash Commitment */}
                  <div className="bg-slate-900/80 p-3.5 rounded-xl border border-slate-800 space-y-1">
                    <span className="text-[11px] text-slate-400 block font-sans font-medium flex items-center space-x-1">
                      <Hash className="w-3.5 h-3.5 text-purple-400" />
                      <span>Deterministic SHA-256 Hash Commitment</span>
                    </span>
                    <p className="text-[11px] text-purple-300 break-all select-all font-bold">
                      {result.record.hash}
                    </p>
                  </div>

                  {/* Blockchain Transaction ID */}
                  <div className="bg-slate-900/80 p-3.5 rounded-xl border border-slate-800 space-y-1">
                    <span className="text-[11px] text-slate-400 block font-sans font-medium">
                      Blockchain Transaction ID
                    </span>
                    <p className="text-[11px] text-blue-300 break-all select-all">
                      {result.record.blockchainTransactionId}
                    </p>
                  </div>

                  {/* Provenance breakdown */}
                  <div className="grid grid-cols-2 gap-2 text-[11px] text-slate-400 bg-slate-900/40 p-3 rounded-lg border border-slate-800">
                    <div>Transaction: <strong className="text-slate-200">{result.record.transactionId}</strong></div>
                    <div>Type: <strong className="text-slate-200">{result.record.transactionType}</strong></div>
                    <div>Model: <strong className="text-slate-200">{result.record.modelVersion}</strong></div>
                    <div>Policy: <strong className="text-slate-200">{result.record.policyId}-v{result.record.policyVersion}</strong></div>
                  </div>
                </div>
              )}
            </div>

            <div className="pt-4 border-t border-slate-800/80 text-[11px] text-slate-500 font-mono">
              Privacy Guarantee: Sensitive financial data remains off-chain in database. Only cryptographic hash commitment is anchored on blockchain.
            </div>
          </div>
        </div>
      </div>

      {/* Human Review Modal if triggered */}
      {pendingRecord && (
        <HumanReviewModal
          pendingRecord={pendingRecord}
          onClose={() => setPendingRecord(null)}
          onComplete={handleHumanReviewComplete}
        />
      )}
    </div>
  );
}
