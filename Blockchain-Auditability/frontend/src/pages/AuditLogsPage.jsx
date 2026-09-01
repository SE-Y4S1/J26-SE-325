import React, { useEffect, useState } from 'react';
import { Database, Search, Hash, Lock, CheckCircle2, AlertTriangle, Copy, ExternalLink } from 'lucide-react';
import { fetchAuditRecords } from '../services/api';

export default function AuditLogsPage({ setActiveTab, setSelectedTxForVerify }) {
  const [records, setRecords] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedRecord, setSelectedRecord] = useState(null);
  const [copiedTx, setCopiedTx] = useState(null);

  const loadRecords = async () => {
    try {
      const res = await fetchAuditRecords();
      setRecords(res.records || []);
      if (res.records && res.records.length > 0 && !selectedRecord) {
        setSelectedRecord(res.records[0]);
      }
    } catch (err) {
      console.error('Failed to fetch records:', err);
    }
  };

  useEffect(() => {
    loadRecords();
  }, []);

  const filteredRecords = records.filter(r =>
    r.transactionId.toLowerCase().includes(searchTerm.toLowerCase()) ||
    r.finalDecision.toLowerCase().includes(searchTerm.toLowerCase()) ||
    r.modelVersion.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleCopyHash = (hash, txId) => {
    navigator.clipboard.writeText(hash);
    setCopiedTx(txId);
    setTimeout(() => setCopiedTx(null), 2000);
  };

  const handleVerify = (txId) => {
    setSelectedTxForVerify(txId);
    setActiveTab('verification');
  };

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-black text-slate-100 flex items-center space-x-2">
            <Database className="w-6 h-6 text-blue-400" />
            <span>Off-Chain Audit Repository</span>
          </h1>
          <p className="text-xs text-slate-400 font-mono mt-1">
            Complete off-chain audit logs storing raw financial fields & provenance metadata alongside on-chain hash commitments.
          </p>
        </div>

        {/* Search */}
        <div className="relative w-full md:w-64">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-3" />
          <input
            type="text"
            placeholder="Search TX ID or decision..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-slate-900 border border-slate-800 text-slate-200 text-xs rounded-xl pl-9 pr-4 py-2.5 focus:border-blue-500 focus:outline-none font-mono"
          />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Record List */}
        <div className="lg:col-span-5 glass-panel p-4 rounded-2xl border border-slate-800 space-y-3">
          <div className="flex items-center justify-between px-2 pb-2 border-b border-slate-800">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Audit Records ({filteredRecords.length})
            </span>
            <span className="text-[11px] text-slate-500 font-mono">Sorted Latest First</span>
          </div>

          {filteredRecords.length === 0 ? (
            <div className="text-center py-12 text-slate-500 text-xs font-mono">
              No matching records found.
            </div>
          ) : (
            <div className="space-y-2.5 max-h-[540px] overflow-y-auto pr-1">
              {filteredRecords.map((r) => {
                const isSelected = selectedRecord?.transactionId === r.transactionId;
                return (
                  <div
                    key={r.transactionId}
                    onClick={() => setSelectedRecord(r)}
                    className={`p-3.5 rounded-xl border transition-all cursor-pointer font-mono text-xs ${
                      isSelected
                        ? 'bg-blue-950/40 border-blue-500/50 shadow-md'
                        : 'bg-slate-900/50 border-slate-800/80 hover:bg-slate-800/40'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="font-bold text-slate-100">{r.transactionId}</span>
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${
                        r.finalDecision.includes('APPROVE')
                          ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
                          : r.finalDecision.includes('REJECT')
                          ? 'bg-rose-500/20 text-rose-300 border-rose-500/30'
                          : 'bg-amber-500/20 text-amber-300 border-amber-500/30'
                      }`}>
                        {r.finalDecision}
                      </span>
                    </div>

                    <div className="flex items-center justify-between text-[11px] text-slate-400">
                      <span>Amount: ${r.amount?.toLocaleString()}</span>
                      <span>Risk: <strong className="text-rose-400">{r.riskScore}</strong></span>
                    </div>

                    <div className="mt-2 pt-2 border-t border-slate-800/60 flex items-center justify-between text-[10px] text-slate-500">
                      <span className="truncate max-w-[180px] font-mono text-purple-300">
                        {r.hash?.substring(0, 14)}...
                      </span>
                      <span>{new Date(r.timestamp).toLocaleTimeString()}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Record Detail Inspector */}
        <div className="lg:col-span-7 glass-panel p-6 rounded-2xl border border-slate-800">
          {!selectedRecord ? (
            <div className="text-center py-20 text-slate-500 text-xs font-mono">
              Select an audit record on the left to inspect detailed fields and canonical structure.
            </div>
          ) : (
            <div className="space-y-5">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div>
                  <h3 className="text-base font-bold text-slate-100 font-mono">
                    Transaction #{selectedRecord.transactionId}
                  </h3>
                  <p className="text-xs text-slate-400">Recorded: {new Date(selectedRecord.timestamp).toLocaleString()}</p>
                </div>

                <button
                  onClick={() => handleVerify(selectedRecord.transactionId)}
                  className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-xs font-semibold transition-all flex items-center space-x-1.5 shadow-md shadow-blue-600/25"
                >
                  <span>Verify Integrity</span>
                  <ExternalLink className="w-3.5 h-3.5" />
                </button>
              </div>

              {/* Hashes Banner */}
              <div className="bg-slate-900 p-4 rounded-xl border border-slate-800 space-y-3 font-mono text-xs">
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-slate-400 text-[11px] font-sans font-semibold text-purple-400">
                      Off-Chain SHA-256 Hash Commitment
                    </span>
                    <button
                      onClick={() => handleCopyHash(selectedRecord.hash, selectedRecord.transactionId)}
                      className="text-slate-400 hover:text-white flex items-center space-x-1 text-[11px]"
                    >
                      <Copy className="w-3 h-3" />
                      <span>{copiedTx === selectedRecord.transactionId ? 'Copied!' : 'Copy'}</span>
                    </button>
                  </div>
                  <p className="text-purple-300 bg-slate-950 p-2 rounded border border-slate-800 break-all">
                    {selectedRecord.hash}
                  </p>
                </div>

                <div>
                  <span className="text-slate-400 text-[11px] font-sans font-semibold text-blue-400 block mb-1">
                    Blockchain Transaction Hash (On-Chain)
                  </span>
                  <p className="text-blue-300 bg-slate-950 p-2 rounded border border-slate-800 break-all">
                    {selectedRecord.blockchainTransactionId}
                  </p>
                </div>
              </div>

              {/* JSON Structure */}
              <div>
                <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider block mb-2">
                  Full Canonical Off-Chain Record Structure (JSON)
                </span>
                <pre className="bg-[#070a12] p-4 rounded-xl border border-slate-800 text-[11px] font-mono text-emerald-400 overflow-x-auto max-h-72">
                  {JSON.stringify(selectedRecord, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
