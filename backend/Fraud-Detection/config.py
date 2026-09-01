"""
config.py
=========
Single source of truth for every tunable constant in the prototype.

Nothing in this project hardcodes a threshold or a weight inline: the model
modules, the deterministic gateway and the API all read their numbers from
here. During the demo you can open this file, change one number, restart, and
show the examiners that the *policy* is configuration, not code.

IMPORTANT (research honesty):
    The weights below are hand-designed heuristics used to *simulate* what a
    trained LSTM autoencoder and a trained GNN would output. They are NOT
    learned parameters. See fraud_model.py for the full disclaimer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


# ---------------------------------------------------------------------------
# Behavioural stream (stand-in for the LSTM autoencoder)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BehavioralConfig:
    """Configuration for the sequence/behavioural anomaly stream."""

    # Relative importance of each feature inside the reconstruction error.
    feature_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "amount": 1.0,
            "typing": 0.9,
            "navigation": 0.8,
            "frequency": 1.0,
        }
    )

    # Prior "profile" assumed for a user we have never seen before.
    # Values are already in normalised (0..1) feature space.
    cold_start_profile: Dict[str, float] = field(
        default_factory=lambda: {
            "amount": 0.567,      # ~ LKR/USD 2,500 typical transfer
            "typing": 0.480,      # ~ 240 keystrokes/min
            "navigation": 0.100,  # "normal" navigation
            "frequency": 0.120,   # ~ 3 transactions per 24h
        }
    )

    # How fast the short-term memory (what the LSTM "remembers") adapts.
    short_term_alpha: float = 0.40
    # How fast the long-term anchor profile adapts. Deliberately very slow and
    # only updated on ALLOWed transactions, so an attacker cannot poison the
    # baseline by pushing through risky transactions.
    anchor_alpha: float = 0.05

    # Gain of the squashing function that turns an error into a 0..1 score.
    recon_gain: float = 3.5   # abrupt novelty  -> reconstruction error
    drift_gain: float = 5.0   # slow drift      -> anchor displacement

    # Normalisation constants used when converting raw fields to 0..1.
    amount_log_max: float = 6.0     # log10(1 + amount) / 6  => 1e6 saturates
    typing_speed_max: float = 500.0  # keystrokes per minute
    frequency_max: float = 25.0      # transactions in the last 24h

    # Plausible human typing band (keystrokes/min). Outside this band the
    # behaviour looks either scripted or hesitant/coerced.
    typing_human_min: float = 60.0
    typing_human_max: float = 600.0

    # A single feature exploding matters even if the other three are normal,
    # so the error is max-pooled with the RMS instead of being averaged away.
    max_pool_factor: float = 0.70

    # Static risk boosts, combined with the sequence error using noisy-OR.
    device_change_boost: float = 0.20
    new_account_boost: float = 0.18
    new_account_days: int = 30
    inhuman_typing_boost: float = 0.25
    automation_boost: float = 0.30

    # "This transfer is N times the user's usual amount" spike detector.
    amount_spike_ratio: float = 8.0        # ratio at which the flag turns on
    amount_spike_saturation: float = 50.0  # ratio at which it is fully on
    amount_spike_boost: float = 0.45

    # Risk value assigned to each navigation pattern label.
    navigation_risk: Dict[str, float] = field(
        default_factory=lambda: {
            "focused": 0.05,
            "normal": 0.10,
            "exploratory": 0.35,
            "hesitant": 0.45,
            "rapid": 0.70,
            "erratic": 0.85,
            "automated": 0.95,
            "scripted": 0.95,
            "unknown": 0.50,
        }
    )

    # Length of the per-user behavioural sequence we keep in memory.
    sequence_length: int = 20


# ---------------------------------------------------------------------------
# Relational stream (stand-in for the Graph Neural Network)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GraphConfig:
    """Configuration for the graph / relational risk stream."""

    # Weight of each relational signal in the aggregated risk.
    signal_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "device_sharing": 1.1,        # one device used by many users
            "beneficiary_hub": 1.2,       # one beneficiary fed by many users
            "device_churn": 0.7,          # one user cycling through devices
            "new_device_link": 0.5,       # first time this user -> this device
            "new_beneficiary_link": 0.5,  # first time this user -> beneficiary
            "location_anomaly": 0.8,      # unseen location / impossible travel
            "shared_ip": 0.6,             # one IP shared across accounts
            "neighbourhood_risk": 1.3,    # risk propagated from flagged nodes
            "structuring_pattern": 1.4,   # many sub-threshold transfers
            "velocity": 0.6,              # burst of activity by one user
        }
    )

    graph_gain: float = 4.0  # squashing gain applied to the normalised sum

    # Fan-out / fan-in saturation: this many extra links = maximum signal.
    fanout_saturation: float = 3.0
    device_churn_baseline: int = 2   # a user may normally own 2 devices
    device_churn_saturation: float = 3.0

    # Message-passing style risk propagation.
    propagation_hops: int = 2
    propagation_decay: float = 0.55
    blocked_node_risk: float = 0.80   # risk written back onto BLOCKed nodes

    # Structuring detection window and shape.
    structuring_window_seconds: int = 6 * 3600
    structuring_min_transfers: int = 3
    structuring_saturation: float = 6.0
    # Amount below which repeated transfers look like deliberate splitting.
    structuring_amount_ceiling: float = 100_000.0
    # A handful of genuinely small payments is normal; the pattern only counts
    # as structuring once the aggregate becomes material.
    structuring_min_total: float = 75_000.0

    # Velocity (burst) detection.
    velocity_window_seconds: int = 3600
    velocity_baseline: int = 3
    velocity_saturation: float = 8.0

    # Two transactions from different locations within this window are
    # physically implausible ("impossible travel").
    impossible_travel_seconds: int = 3600

    # A beneficiary is not fully trusted the moment the first payment clears.
    new_relationship_window_seconds: int = 24 * 3600


# ---------------------------------------------------------------------------
# Late fusion of the two streams
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FusionConfig:
    """How the two independent streams are combined into one risk score."""

    behavioral_weight: float = 0.50
    graph_weight: float = 0.50
    # Extra risk when BOTH streams agree. Corroboration between an unusual
    # behaviour and an unusual relationship is stronger than either alone.
    corroboration_bonus: float = 0.18


# ---------------------------------------------------------------------------
# Deterministic security gateway
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GatewayConfig:
    """
    Thresholds and hard policy limits for the deterministic gateway.

    The ML side never sees these values; it only produces a score. The gateway
    alone converts a score into ALLOW / STEP-UP / BLOCK.
    """

    policy_version: str = "gateway-policy-v1.0"

    # Risk bands.  risk < allow_max                -> ALLOW
    #              allow_max <= risk < block_min   -> STEP-UP
    #              risk >= block_min               -> BLOCK
    allow_max: float = 0.35
    block_min: float = 0.70

    # --- deterministic escalation rules -----------------------------------
    # G1: very large transfer combined with a fresh device or fresh payee.
    hard_block_amount: float = 250_000.0
    hard_block_min_risk: float = 0.40
    # G2: any device change above this amount always needs step-up auth.
    device_change_stepup_amount: float = 25_000.0
    # G3: strong relational evidence always needs step-up auth.
    graph_stepup_score: float = 0.75
    # G4: brand-new account moving real money.
    new_account_max_age_days: int = 7
    new_account_stepup_amount: float = 10_000.0
    # G5: both streams simultaneously confident -> block.
    dual_stream_block_behavioral: float = 0.90
    dual_stream_block_graph: float = 0.60
    # G7: split transfers that together reach the single-transaction hard
    # limit are treated exactly like one transfer of that size.
    structuring_block_min_transfers: int = 4

    # --- deterministic de-escalation guard --------------------------------
    # G6: never block a trivial payment from a long-standing, clean account.
    micro_payment_amount: float = 1_000.0
    micro_payment_min_account_age: int = 180
    micro_payment_max_graph: float = 0.55


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AuditConfig:
    log_file: str = "audit_log.jsonl"
    memory_limit: int = 500  # entries kept in RAM for the /audit endpoint


@dataclass(frozen=True)
class Settings:
    behavioral: BehavioralConfig = field(default_factory=BehavioralConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)


# Imported by every other module.
SETTINGS = Settings()
