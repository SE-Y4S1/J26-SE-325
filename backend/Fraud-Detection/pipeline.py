"""
pipeline.py
===========
One reusable implementation of the end-to-end detection pipeline:

    Transaction
        |
        v
    Behavioural analysis  -> LSTM-like anomaly score      (fraud_model.py)
    Relational analysis   -> GNN-like relational score     (fraud_model.py)
        |
        v
    Fusion -> final risk score                             (fraud_model.py)
        |
        v
    Deterministic Security Gateway -> ALLOW / STEP-UP / BLOCK   (gateway.py)
        |
        v
    Audit log                                              (audit_log.py)

Both the API (`app.py`) and the offline adversarial harness
(`adversarial_test.py`) call this, so the live demo and the research
experiment are provably running identical logic.
"""

from __future__ import annotations

from typing import Optional

from audit_log import AuditLog
from fraud_model import MODEL_MODE, DualStreamFraudModel, ModelOutput
from gateway import DeterministicGateway, GatewayDecision
from schemas import ScoreResponse, SignalOut, Transaction


class FraudPipeline:
    """Model + gateway + audit trail, wired together."""

    def __init__(
        self,
        model: Optional[DualStreamFraudModel] = None,
        gateway: Optional[DeterministicGateway] = None,
        audit: Optional[AuditLog] = None,
    ) -> None:
        self.model = model or DualStreamFraudModel()
        self.gateway = gateway or DeterministicGateway()
        self.audit = audit
        self.transactions_scored = 0

    def evaluate(
        self,
        transaction: Transaction,
        *,
        persist_state: bool = True,
        attack_type: Optional[str] = None,
    ) -> ScoreResponse:
        """Run one transaction through the whole pipeline."""
        tx = transaction.normalised()

        # 1-3. Advisory ML stage: two independent streams, then fusion.
        model_output: ModelOutput = self.model.analyze(tx)

        # 4. Enforcement stage: the gateway alone chooses the action.
        verdict: GatewayDecision = self.gateway.decide(tx, model_output)

        # 5. Update memory only if this is a real (non-what-if) evaluation.
        if persist_state:
            self.model.commit(tx, verdict.action)

        self.transactions_scored += 1

        # 6. Audit trail.
        if self.audit is not None:
            self.audit.record(
                transaction_id=tx.transaction_id or "unknown",
                user_id=tx.user_id,
                behavioral_score=model_output.behavioral_score,
                graph_score=model_output.graph_score,
                risk_score=model_output.risk_score,
                decision=verdict.action,
                reason=verdict.reason,
                attack_type=attack_type,
            )

        return ScoreResponse(
            transaction_id=tx.transaction_id or "unknown",
            user_id=tx.user_id,
            timestamp=tx.timestamp or 0.0,
            behavioral_score=model_output.behavioral_score,
            graph_score=model_output.graph_score,
            risk_score=model_output.risk_score,
            decision=verdict.action,  # type: ignore[arg-type]
            reason=verdict.reason,
            reasons=verdict.reasons,
            rules_fired=verdict.rules_fired,
            thresholds=verdict.thresholds,
            policy_version=verdict.policy_version,
            behavioral_signals=[
                SignalOut(name=s.name, value=round(s.value, 4),
                          weight=round(s.weight, 3), message=s.message)
                for s in model_output.behavioral_signals
            ],
            graph_signals=[
                SignalOut(name=s.name, value=round(s.value, 4),
                          weight=round(s.weight, 3), message=s.message)
                for s in model_output.graph_signals
            ],
            model_mode=MODEL_MODE,
            attack_type=attack_type,
        )

    def reset(self) -> None:
        """Clear all learned state so a live demo can be repeated cleanly."""
        self.model.reset()
        if self.audit is not None:
            self.audit.clear()
        self.transactions_scored = 0
