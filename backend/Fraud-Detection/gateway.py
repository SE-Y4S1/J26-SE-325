"""
gateway.py
==========
THE DETERMINISTIC SECURITY GATEWAY.

This is the heart of the research contribution. The machine-learning side of
the system is *advisory only*: it produces a number. This module is the only
place in the entire service that is allowed to decide what happens to a
transaction, and it does so with fixed, auditable, human-readable rules:

        ALLOW    - let the transaction through
        STEP-UP  - demand additional authentication (OTP / biometric)
        BLOCK    - refuse the transaction

Why it matters
--------------
A neural network is a probabilistic function that can be attacked, drift, or
fail silently. A bank cannot let such a component hold the authority to move
money. By separating "who estimates risk" from "who enforces policy" we get:

  * Determinism   - the same score + same context always yields the same action.
  * Auditability  - every decision names the rule and the threshold that caused it.
  * Governance    - policy can be changed by a risk officer editing config.py,
                    with no retraining and no redeployment of the model.
  * Fail-safety   - if the model misbehaves, hard rules still bound the outcome.

Decision procedure (strictly ordered, no randomness anywhere)
-------------------------------------------------------------
  1. Map the fused risk score onto a band using the configured thresholds.
  2. Apply escalation rules G1..G5 - each can only raise severity.
  3. Apply de-escalation guard G6 - the single documented rule that may lower
     severity, so that trivial payments from clean accounts are never blocked.
  4. Emit the decision together with every reason and every rule that fired.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from config import SETTINGS, GatewayConfig
from fraud_model import ModelOutput, Signal
from schemas import Transaction


class Decision(str, Enum):
    ALLOW = "ALLOW"
    STEP_UP = "STEP-UP"
    BLOCK = "BLOCK"


# Severity ordering lets us say "escalate to at least STEP-UP" safely.
SEVERITY: Dict[Decision, int] = {
    Decision.ALLOW: 0,
    Decision.STEP_UP: 1,
    Decision.BLOCK: 2,
}


@dataclass
class GatewayDecision:
    """The gateway's complete, auditable verdict."""

    decision: Decision
    risk_score: float
    reason: str                                   # one-line summary
    reasons: List[str] = field(default_factory=list)
    rules_fired: List[str] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)
    policy_version: str = ""
    band: str = ""

    @property
    def action(self) -> str:
        return self.decision.value


@dataclass(frozen=True)
class PolicyRule:
    """A single deterministic rule. `effect` is the minimum severity it forces."""

    rule_id: str
    description: str
    effect: Decision


class DeterministicGateway:
    """Converts an advisory risk score into an enforced action."""

    def __init__(self, config: Optional[GatewayConfig] = None) -> None:
        self.cfg = config or SETTINGS.gateway

    # -- step 1: threshold banding -------------------------------------------
    def classify_band(self, risk_score: float) -> Tuple[Decision, str]:
        """Pure threshold logic - the 'low / medium / high' bands."""
        cfg = self.cfg
        if risk_score < cfg.allow_max:
            return Decision.ALLOW, "low"
        if risk_score < cfg.block_min:
            return Decision.STEP_UP, "medium"
        return Decision.BLOCK, "high"

    # -- step 2 & 3: deterministic policy rules -------------------------------
    def _evaluate_rules(
        self, tx: Transaction, model: ModelOutput
    ) -> Tuple[List[PolicyRule], List[PolicyRule]]:
        """Return (escalations, de-escalations) that apply to this transaction."""
        cfg = self.cfg
        escalations: List[PolicyRule] = []
        deescalations: List[PolicyRule] = []

        # G1 - large value moved through an unfamiliar device or payee.
        if (
            tx.amount >= cfg.hard_block_amount
            and (tx.device_change or tx.beneficiary_change)
            and model.risk_score >= cfg.hard_block_min_risk
        ):
            escalations.append(
                PolicyRule(
                    "G1",
                    f"Amount {tx.amount:,.0f} >= {cfg.hard_block_amount:,.0f} on a new "
                    "device/beneficiary while risk is elevated",
                    Decision.BLOCK,
                )
            )

        # G2 - device change above the step-up floor always needs re-auth.
        if tx.device_change and tx.amount >= cfg.device_change_stepup_amount:
            escalations.append(
                PolicyRule(
                    "G2",
                    f"New device detected with amount >= "
                    f"{cfg.device_change_stepup_amount:,.0f}",
                    Decision.STEP_UP,
                )
            )

        # G3 - strong relational evidence, regardless of behaviour.
        if model.graph_score >= cfg.graph_stepup_score:
            escalations.append(
                PolicyRule(
                    "G3",
                    f"Relational risk {model.graph_score:.2f} >= "
                    f"{cfg.graph_stepup_score:.2f} (suspicious network of "
                    "devices/beneficiaries)",
                    Decision.STEP_UP,
                )
            )

        # G4 - brand-new account moving meaningful money.
        if (
            tx.account_age <= cfg.new_account_max_age_days
            and tx.amount >= cfg.new_account_stepup_amount
        ):
            escalations.append(
                PolicyRule(
                    "G4",
                    f"Account age {tx.account_age}d <= "
                    f"{cfg.new_account_max_age_days}d with a significant amount",
                    Decision.STEP_UP,
                )
            )

        # G5 - both independent streams are confident at the same time.
        if (
            model.behavioral_score >= cfg.dual_stream_block_behavioral
            and model.graph_score >= cfg.dual_stream_block_graph
        ):
            escalations.append(
                PolicyRule(
                    "G5",
                    f"Dual-stream corroboration: behavioural "
                    f"{model.behavioral_score:.2f} and relational "
                    f"{model.graph_score:.2f} both high",
                    Decision.BLOCK,
                )
            )

        # G7 - aggregate value limit. Splitting a transfer into many small
        # pieces must not defeat the limit that applies to one large transfer.
        # This is a pure arithmetic rule on measured facts, not a model score:
        # it is why the gateway catches structuring even while the ML risk
        # score is still sitting in the medium band.
        structuring_total = model.context.get("graph_structuring_total", 0.0)
        structuring_count = model.context.get("graph_structuring_count", 0.0)
        if (
            structuring_total >= cfg.hard_block_amount
            and structuring_count >= cfg.structuring_block_min_transfers
        ):
            escalations.append(
                PolicyRule(
                    "G7",
                    f"Aggregate limit breached: {int(structuring_count)} split transfers "
                    f"to the same payee total {structuring_total:,.0f}, which exceeds the "
                    f"single-transaction limit of {cfg.hard_block_amount:,.0f}",
                    Decision.BLOCK,
                )
            )

        # G6 - de-escalation guard: never block a trivial, clean payment.
        if (
            tx.amount <= cfg.micro_payment_amount
            and tx.account_age >= cfg.micro_payment_min_account_age
            and not tx.device_change
            and model.graph_score < cfg.micro_payment_max_graph
        ):
            deescalations.append(
                PolicyRule(
                    "G6",
                    f"Micro-payment guard: amount <= {cfg.micro_payment_amount:,.0f} "
                    "from an established account with a clean relationship graph - "
                    "capped at STEP-UP",
                    Decision.STEP_UP,
                )
            )

        return escalations, deescalations

    # -- step 4: explanation ---------------------------------------------------
    @staticmethod
    def _model_reasons(model: ModelOutput, limit: int = 5) -> List[str]:
        """Top contributing model signals, in plain English."""
        ranked: List[Signal] = model.all_signals
        return [s.message for s in ranked[:limit]]

    # -- public API -------------------------------------------------------------
    def decide(self, tx: Transaction, model: ModelOutput) -> GatewayDecision:
        cfg = self.cfg

        base_decision, band = self.classify_band(model.risk_score)
        decision = base_decision
        rules_fired: List[str] = []
        reasons: List[str] = []

        escalations, deescalations = self._evaluate_rules(tx, model)

        # Escalations may only increase severity.
        for rule in escalations:
            if SEVERITY[rule.effect] > SEVERITY[decision]:
                decision = rule.effect
            rules_fired.append(f"{rule.rule_id}: {rule.description}")

        # The de-escalation guard may only cap severity downwards, and it is
        # never allowed to undo an explicit hard-block rule - it exists to
        # soften a threshold outcome, not to override a policy limit.
        hard_blocked = any(r.effect is Decision.BLOCK for r in escalations)
        if not hard_blocked:
            for rule in deescalations:
                if SEVERITY[decision] > SEVERITY[rule.effect]:
                    decision = rule.effect
                    rules_fired.append(f"{rule.rule_id}: {rule.description}")

        # ---- build the explanation -------------------------------------------
        reasons.extend(self._model_reasons(model))
        if not reasons:
            # No single indicator crossed its reporting threshold. Say what is
            # actually true for the score rather than declaring it normal.
            if model.risk_score < cfg.allow_max / 2:
                reasons.append("Behaviour and relationships match this user's normal profile")
            else:
                reasons.append(
                    "Several small deviations from the usual pattern, with no single "
                    "strong indicator"
                )

        band_text = {
            "low": f"risk {model.risk_score:.2f} < {cfg.allow_max:.2f} (low band)",
            "medium": (
                f"risk {model.risk_score:.2f} in [{cfg.allow_max:.2f}, "
                f"{cfg.block_min:.2f}) (medium band)"
            ),
            "high": f"risk {model.risk_score:.2f} >= {cfg.block_min:.2f} (high band)",
        }[band]

        # Plain ASCII only: this text is printed to Windows consoles and
        # written to the audit file.
        if decision == base_decision:
            summary = f"{decision.value} - {band_text}"
        else:
            summary = (
                f"{decision.value} - {band_text}, overridden by policy rule "
                f"{rules_fired[-1].split(':')[0] if rules_fired else '?'}"
            )

        if decision == Decision.STEP_UP:
            reasons.append("Action: additional authentication required (OTP / biometric)")
        elif decision == Decision.BLOCK:
            reasons.append("Action: transaction refused and referred for manual review")

        return GatewayDecision(
            decision=decision,
            risk_score=model.risk_score,
            reason=summary,
            reasons=reasons,
            rules_fired=rules_fired,
            thresholds={
                "allow_below": cfg.allow_max,
                "block_at_or_above": cfg.block_min,
            },
            policy_version=cfg.policy_version,
            band=band,
        )
