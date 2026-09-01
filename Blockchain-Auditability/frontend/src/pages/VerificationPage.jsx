import React, { useEffect, useState } from 'react';
import { Binary, ShieldCheck, ShieldAlert, AlertTriangle, RefreshCw, Lock, ArrowRight, Zap, CheckCircle2, XCircle } from 'lucide-react';
import { fetchAuditRecords, verifyAuditRecord, simulateTamperRecord } from '../services/api';

export default function VerificationPage({ selectedTxForVerify }) {
  const [records, setRecords] = useState([]);
  const [selectedTxId, setSelectedTxId] = useState(selectedTxForVerify || '');
  const [verificationResult, setVerificationResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [tamperingLoading, setTamperingLoading] = useState(false);

  const loadRecords = async () => {
    try {
      const res = await fetchAuditRecords();
      setRecords(res.records || []);
      if (res.records && res.records.length > 0 && !selectedTxId) {
        setSelectedTxId(res.records[0].transactionId);
      }
    } catch (err) {
      console.error('Failed to load records for verification:', err);
    }
  };

  useEffect(() => {
    loadRecords();
  }, []);

  useEffect(() => {
    if (selectedTxForVerify) {
      setSelectedTxId(selectedTxForVerify);
      runVerification(selectedTxForVerify);
    }
  }, [selectedTxForVerify]);

  const runVerification = async (txId = selectedTxId) => {
    if (!txId) return;
    setLoading(true);
    setVerificationResult(null);
    try {
      const res = await verifyAuditRecord(txId);
      setVerificationResult(res);
    } catch (err) {
      console.error('Verification error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleTamperSimulation = async () => {
    if (!selectedTxId) return;
    setTamperingLoading(true);
    try {
      await simulateTamperRecord(selectedTxId, 'riskScore', 20);
      // Re-run verification automatically to demonstrate failure instantly!
      await runVerification(selectedTxId);
      await loadRecords();
    } catch (err) {
      console.error('Tampering simulation error:', err);
    } finally {
      setTamperingLoading(false);
    }
  };

  const currentRecord = records.find(r => r.transactionId === selectedTxId);

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-black text-slate-100 flex items-center space-x-2">
            <Binary className="w-6 h-6 text-purple-400" />
            <span>Audit Verification & Tamper Detection Lab</span>
          </h1>
          <p className="text-xs text-slate-400 font-mono mt-1">
            Recompute off-chain SHA-256 canonical hash & validate integrity against immutable blockchain commitment.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Selector & Actions */}
        <div className="lg:col-span-4 space-y-4">
          <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
              Select Audit Record
            </h3>

            <div>
              <label className="block text-xs font-mono text-slate-400 mb-1.5">Transaction ID</label>
              <select
                value={selectedTxId}
                onChange={(e) => {
                  setSelectedTxId(e.target.value);
                  setVerificationResult(null);
                }}
                className="w-full bg-slate-900 border border-slate-800 focus:border-blue-500 rounded-xl px-3 py-2.5 text-xs text-slate-100 font-mono focus:outline-none"
              >
                {records.length === 0 && <option value="">No audit records available</option>}
                {records.map((r) => (
                  <option key={r.transactionId} value={r.transactionId}>
                    {r.transactionId} ({r.finalDecision}) {r.isTampered ? '[TAMPERED]' : ''}
                  </option>
                ))}
              </select>
            </div>

            {currentRecord && (
              <div className="bg-slate-900/60 p-3.5 rounded-xl border border-slate-800 font-mono text-xs space-y-1 text-slate-300">
                <div>Model: <strong className="text-slate-100">{currentRecord.modelVersion}</strong></div>
                <div>Risk / Conf: <strong className="text-rose-400">{currentRecord.riskScore}</strong> / <strong className="text-amber-400">{currentRecord.confidence}%</strong></div>
                <div>Amount: <strong>${currentRecord.amount?.toLocaleString()}</strong></div>
                {currentRecord.isTampered && (
                  <div className="mt-2 text-rose-400 font-bold text-[11px] flex items-center space-x-1">
                    <AlertTriangle className="w-3.5 h-3.5" />
                    <span>Field '{currentRecord.tamperedField}' altered off-chain</span>
                  </div>
                )}
              </div>
            )}

            <button
              onClick={() => runVerification()}
              disabled={loading || !selectedTxId}
              className="w-full py-3 px-4 bg-purple-600 hover:bg-purple-500 disabled:opacity-50 text-white font-bold text-xs rounded-xl shadow-lg shadow-purple-600/25 transition-all flex items-center justify-center space-x-2"
            >
              {loading ? (
                <span>Recomputing Canonical Hash...</span>
              ) : (
                <>
                  <RefreshCw className="w-4 h-4" />
                  <span>Verify Record Integrity</span>
                </>
              )}
            </button>

            {/* Simulated Tampering (Dev Feature) */}
            <div className="pt-3 border-t border-slate-800">
              <span className="text-[11px] font-semibold text-rose-400 uppercase tracking-wider block mb-2">
                Demonstration Testing Feature
              </span>
              <button
                onClick={handleTamperSimulation}
                disabled={tamperingLoading || !selectedTxId}
                className="w-full py-2.5 px-4 bg-rose-950/60 hover:bg-rose-900/80 border border-rose-500/40 text-rose-300 font-semibold text-xs rounded-xl transition-all flex items-center justify-center space-x-2"
              >
                <Zap className="w-4 h-4 text-rose-400" />
                <span>{tamperingLoading ? 'Simulating Tamper...' : 'Simulate Tampering'}</span>
              </button>
              <p className="text-[10px] text-slate-500 mt-1.5 leading-snug">
                Modifies off-chain risk score from {currentRecord?.riskScore || 92} to 20 without altering blockchain hash commitment.
              </p>
            </div>
          </div>
        </div>

        {/* Verification Output */}
        <div className="lg:col-span-8 glass-panel p-6 rounded-2xl border border-slate-800">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300 mb-4 border-b border-slate-800 pb-3">
            Cryptographic Integrity Verification Results
          </h3>

          {!verificationResult ? (
            <div className="text-center py-20 text-slate-500 space-y-2">
              <Binary className="w-10 h-10 mx-auto opacity-30" />
              <p className="text-xs font-mono">Select an audit transaction and click "Verify Record Integrity"</p>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Visual State Banner */}
              <div className={`p-5 rounded-2xl border flex items-center space-x-4 ${
                verificationResult.verified
                  ? 'bg-emerald-950/50 border-emerald-500/50 text-emerald-300 glow-emerald'
                  : 'bg-rose-950/50 border-rose-500/50 text-rose-300 glow-rose'
              }`}>
                {verificationResult.verified ? (
                  <CheckCircle2 className="w-10 h-10 text-emerald-400 shrink-0" />
                ) : (
                  <XCircle className="w-10 h-10 text-rose-400 shrink-0" />
                )}
                <div>
                  <div className="flex items-center space-x-2">
                    <h2 className="text-xl font-black tracking-tight">
                      {verificationResult.verified ? 'VERIFIED' : 'INTEGRITY FAILED'}
                    </h2>
                    <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold uppercase ${
                      verificationResult.verified
                        ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                        : 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                    }`}>
                      {verificationResult.status}
                    </span>
                  </div>
                  <p className="text-xs mt-1 opacity-90 leading-relaxed font-sans">
                    {verificationResult.message}
                  </p>
                </div>
              </div>

              {/* Hash Comparison Matrix */}
              <div className="space-y-3 font-mono text-xs">
                <span className="text-xs font-semibold text-slate-400 font-sans uppercase tracking-wider block">
                  SHA-256 Hash Comparison Matrix
                </span>

                <div className="bg-slate-900/80 p-3.5 rounded-xl border border-slate-800">
                  <span className="text-[11px] text-slate-400 block font-sans mb-1">
                    1. Recomputed Off-Chain SHA-256 Hash (Current DB State):
                  </span>
                  <p className={`p-2 rounded font-bold break-all ${
                    verificationResult.verified ? 'bg-slate-950 text-emerald-300' : 'bg-rose-950/80 text-rose-300 border border-rose-500/30'
                  }`}>
                    {verificationResult.details?.calculatedCurrentHash}
                  </p>
                </div>

                <div className="bg-slate-900/80 p-3.5 rounded-xl border border-slate-800">
                  <span className="text-[11px] text-slate-400 block font-sans mb-1">
                    2. Stored Blockchain Hash Commitment (Immutable Smart Contract):
                  </span>
                  <p className="p-2 rounded bg-slate-950 text-purple-300 font-bold break-all border border-slate-800">
                    {verificationResult.details?.blockchainOnChainHash}
                  </p>
                </div>
              </div>

              {/* Canonical Payload Inspection */}
              {verificationResult.details?.canonicalPayload && (
                <div className="space-y-2">
                  <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">
                    Canonical Payload Used for Hashing
                  </span>
                  <pre className="bg-[#070a12] p-4 rounded-xl border border-slate-800 text-[11px] font-mono text-slate-300 overflow-x-auto max-h-48">
                    {JSON.stringify(verificationResult.details.canonicalPayload, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
