"""
fraud_model.py
==============
Dual-stream fraud / behavioural anomaly model.

    Stream A - BehavioralStream : stand-in for an LSTM AUTOENCODER
    Stream B - RelationalStream : stand-in for a GRAPH NEURAL NETWORK
    Fusion   - DualStreamFraudModel : late fusion of the two streams

=============================================================================
 !! RESEARCH HONESTY NOTICE  --  READ BEFORE PRESENTING !!
-----------------------------------------------------------------------------
 NO NEURAL NETWORK IS TRAINED OR EXECUTED IN THIS FILE.

 Both streams are deterministic, hand-designed heuristics that *simulate the
 shape of the output* a trained model would produce:

   * The behavioural stream mimics an LSTM autoencoder by measuring how far
     the current behaviour vector is from an exponentially-weighted memory of
     the user's recent sequence (reconstruction error), plus how far that
     memory has drifted from a long-term anchor profile (drift error).

   * The relational stream mimics a GNN by building a real heterogeneous
     graph (users / devices / beneficiaries / locations / IPs) and running a
     decayed 2-hop risk propagation over it, which is the same "message
     passing" intuition a GNN layer implements.

 Every score is therefore a SIMULATED MODEL OUTPUT, labelled as such in the
 API response via `model_mode = "SIMULATED_PROTOTYPE"`.

 HOW TO PLUG IN THE REAL MODELS LATER
 ------------------------------------
 Both stream classes expose exactly one scoring method:

       score(tx, state) -> StreamResult(score: float, signals: list[Signal])

 To use the real research models, write `TorchBehavioralStream` and
 `TorchRelationalStream` with the same method signature (loading a trained
 PyTorch LSTM autoencoder and a PyTorch-Geometric GNN respectively) and pass
 them to `DualStreamFraudModel(behavioral=..., relational=...)`. Nothing else
 in the project - gateway, API, UI - has to change, because the gateway only
 ever consumes a float.
=============================================================================
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple

from config import SETTINGS, BehavioralConfig, FusionConfig, GraphConfig
from schemas import Transaction

MODEL_MODE = "SIMULATED_PROTOTYPE"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def squash(error: float, gain: float) -> float:
    """
    Map an unbounded, non-negative error onto (0, 1).

    Uses 1 - exp(-gain * error): smooth, monotonic, saturating - the same
    shape you get when normalising a neural network's reconstruction error.
    """
    return clamp(1.0 - math.exp(-gain * max(0.0, error)))


def noisy_or(*probabilities: float) -> float:
    """
    Combine independent evidence: 1 - PROD(1 - p_i).

    Any single strong signal is enough to raise the score, and several weak
    signals reinforce each other - the behaviour we want from an ensemble.
    """
    inverse = 1.0
    for p in probabilities:
        inverse *= (1.0 - clamp(p))
    return clamp(1.0 - inverse)


@dataclass
class Signal:
    """One explainable piece of evidence produced by a stream."""

    name: str
    value: float       # normalised strength, 0..1
    weight: float      # importance inside its stream
    message: str       # plain-English explanation shown to the user

    @property
    def contribution(self) -> float:
        return self.value * self.weight


@dataclass
class StreamResult:
    score: float
    signals: List[Signal] = field(default_factory=list)
    details: Dict[str, float] = field(default_factory=dict)


@dataclass
class ModelOutput:
    """The complete (simulated) model verdict handed to the gateway."""

    behavioral_score: float
    graph_score: float
    risk_score: float
    behavioral_signals: List[Signal] = field(default_factory=list)
    graph_signals: List[Signal] = field(default_factory=list)
    # Raw measured facts (amount ratios, structuring totals, ...) that the
    # deterministic gateway may use for hard limits. These are measurements,
    # not model opinions.
    context: Dict[str, float] = field(default_factory=dict)
    model_mode: str = MODEL_MODE

    @property
    def all_signals(self) -> List[Signal]:
        return sorted(
            self.behavioral_signals + self.graph_signals,
            key=lambda s: s.contribution,
            reverse=True,
        )


# ===========================================================================
# STREAM A - behavioural sequence anomaly  (LSTM autoencoder stand-in)
# ===========================================================================
@dataclass
class BehavioralProfile:
    """Per-user memory: the 'hidden state' our simulated LSTM carries."""

    anchor: Dict[str, float]        # long-term baseline (slow, poison-resistant)
    short_term: Dict[str, float]    # recent behaviour (fast adapting)
    sequence: Deque[Dict[str, float]] = field(default_factory=deque)
    observed: int = 0


class BehavioralStream:
    """
    Simulated LSTM autoencoder over the user's behavioural sequence.

    A trained autoencoder learns to reconstruct *normal* sequences and fails
    (high reconstruction error) on abnormal ones. We reproduce that signal
    with two complementary errors:

      reconstruction_error : distance between the current feature vector and
                             the short-term memory of the sequence. Catches
                             sudden, obvious deviations.

      drift_error          : distance between the short-term memory and the
                             long-term anchor. Catches SLOW-DRIFT attacks,
                             where each individual step is small enough to
                             look normal but the cumulative change is not.

    Static behavioural-biometric / device-fingerprint flags (new device,
    inhuman typing rhythm, scripted navigation, very young account) are then
    merged with noisy-OR.
    """

    def __init__(self, config: BehavioralConfig | None = None) -> None:
        self.cfg = config or SETTINGS.behavioral
        self.profiles: Dict[str, BehavioralProfile] = {}

    # -- feature engineering -------------------------------------------------
    def extract_features(self, tx: Transaction) -> Dict[str, float]:
        """Turn a raw transaction into a normalised 0..1 behaviour vector."""
        cfg = self.cfg
        return {
            "amount": clamp(math.log10(1.0 + tx.amount) / cfg.amount_log_max),
            "typing": clamp(tx.typing_speed / cfg.typing_speed_max),
            "navigation": cfg.navigation_risk.get(tx.navigation_pattern, 0.5),
            "frequency": clamp(tx.transaction_frequency / cfg.frequency_max),
        }

    def _profile(self, user_id: str) -> BehavioralProfile:
        if user_id not in self.profiles:
            base = dict(self.cfg.cold_start_profile)
            self.profiles[user_id] = BehavioralProfile(
                anchor=dict(base),
                short_term=dict(base),
                sequence=deque(maxlen=self.cfg.sequence_length),
            )
        return self.profiles[user_id]

    def _weighted_rms(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        """Weighted root-mean-square distance between two feature vectors."""
        weights = self.cfg.feature_weights
        total_w = sum(weights.values()) or 1.0
        acc = sum(w * (a[k] - b[k]) ** 2 for k, w in weights.items())
        return math.sqrt(acc / total_w)

    def _pooled_error(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        """
        Reconstruction error: RMS distance, max-pooled with the single worst
        feature. Averaging alone would let one extreme feature (say a 100x
        amount) be diluted by three normal ones - exactly the blind spot a
        camouflage attacker aims for.
        """
        weights = self.cfg.feature_weights
        rms = self._weighted_rms(a, b)
        worst = max(abs(a[k] - b[k]) * w for k, w in weights.items())
        return max(rms, self.cfg.max_pool_factor * worst)

    def typical_amount(self, tx: Transaction) -> float:
        """
        Best estimate of this user's usual transfer size.

        Uses the learned anchor once we have seen the user; falls back to the
        `previous_transaction_amount` supplied with the request for a
        cold-start user.
        """
        profile = self._profile(tx.user_id)
        if profile.observed == 0 and tx.previous_transaction_amount > 0:
            return tx.previous_transaction_amount
        return max(1.0, 10 ** (profile.anchor["amount"] * self.cfg.amount_log_max) - 1.0)

    # -- scoring (pure: never mutates state) ---------------------------------
    def score(self, tx: Transaction) -> StreamResult:
        cfg = self.cfg
        profile = self._profile(tx.user_id)
        features = self.extract_features(tx)

        recon_error = self._pooled_error(features, profile.short_term)
        drift_error = self._weighted_rms(profile.short_term, profile.anchor)

        recon_score = squash(recon_error, cfg.recon_gain)
        drift_score = squash(drift_error, cfg.drift_gain)

        signals: List[Signal] = []

        # --- which individual feature drove the reconstruction error? -------
        deviations = {
            k: abs(features[k] - profile.short_term[k]) * w
            for k, w in cfg.feature_weights.items()
        }
        def direction(key: str) -> str:
            return "higher" if features[key] > profile.short_term[key] else "lower"

        if deviations["amount"] > 0.10:
            signals.append(
                Signal(
                    "amount_deviation",
                    clamp(deviations["amount"] * 2.5),
                    1.0,
                    "High transaction amount compared to this user's usual pattern"
                    if direction("amount") == "higher"
                    else "Transaction amount is far below this user's usual pattern",
                )
            )
        if deviations["typing"] > 0.10:
            signals.append(
                Signal(
                    "typing_deviation",
                    clamp(deviations["typing"] * 2.5),
                    0.9,
                    "Unusual typing pattern: typing is much "
                    f"{direction('typing')} than this user's biometric baseline",
                )
            )
        if deviations["navigation"] > 0.15:
            signals.append(
                Signal(
                    "navigation_deviation",
                    clamp(deviations["navigation"] * 1.5),
                    0.8,
                    f"Abnormal in-app navigation behaviour ('{tx.navigation_pattern}')",
                )
            )
        if deviations["frequency"] > 0.10:
            signals.append(
                Signal(
                    "frequency_deviation",
                    clamp(deviations["frequency"] * 2.5),
                    1.0,
                    "Abnormal transaction frequency for this account",
                )
            )
        if drift_score > 0.15:
            signals.append(
                Signal(
                    "behavioural_drift",
                    drift_score,
                    1.0,
                    "Gradual behavioural drift away from the long-term profile",
                )
            )

        # --- static behavioural-biometric / fingerprint boosts ---------------
        boosts: List[float] = []

        if tx.device_change:
            boosts.append(cfg.device_change_boost)
            signals.append(
                Signal("device_change", 1.0, cfg.device_change_boost,
                       "New device detected (device fingerprint changed)")
            )

        if tx.account_age < cfg.new_account_days:
            strength = 1.0 - (tx.account_age / max(1, cfg.new_account_days))
            boosts.append(cfg.new_account_boost * strength)
            signals.append(
                Signal("young_account", clamp(strength), cfg.new_account_boost,
                       f"Account is only {tx.account_age} day(s) old")
            )

        if tx.typing_speed > cfg.typing_human_max or (
            0 < tx.typing_speed < cfg.typing_human_min
        ):
            boosts.append(cfg.inhuman_typing_boost)
            signals.append(
                Signal("inhuman_typing", 1.0, cfg.inhuman_typing_boost,
                       f"Typing speed {tx.typing_speed:.0f} keystrokes/min is "
                       "outside the plausible human range")
            )

        if tx.navigation_pattern in ("automated", "scripted"):
            boosts.append(cfg.automation_boost)
            signals.append(
                Signal("automation", 1.0, cfg.automation_boost,
                       "Navigation looks automated / scripted, not human")
            )

        # Amount spike relative to what this user normally moves.
        usual = self.typical_amount(tx)
        ratio = tx.amount / usual if usual > 0 else 0.0
        if ratio >= cfg.amount_spike_ratio:
            strength = clamp(
                math.log(ratio) / math.log(max(cfg.amount_spike_saturation, 2.0))
            )
            boosts.append(cfg.amount_spike_boost * strength)
            signals.append(
                Signal("amount_spike", strength, cfg.amount_spike_boost,
                       f"High transaction amount: {ratio:.0f}x this user's usual "
                       f"transfer of about {usual:,.0f}")
            )

        score = noisy_or(recon_score, drift_score, *boosts)

        return StreamResult(
            score=score,
            signals=signals,
            details={
                "reconstruction_error": round(recon_error, 4),
                "reconstruction_score": round(recon_score, 4),
                "drift_error": round(drift_error, 4),
                "drift_score": round(drift_score, 4),
                "amount_ratio": round(ratio, 2),
                "observations": float(profile.observed),
            },
        )

    # -- state update --------------------------------------------------------
    def commit(self, tx: Transaction, allowed: bool) -> None:
        """
        Fold the transaction into the user's memory.

        The fast short-term memory always updates (the LSTM sees everything),
        but the slow anchor profile only updates on ALLOWed transactions, so a
        rejected attacker cannot gradually redefine "normal" for the account.
        """
        cfg = self.cfg
        profile = self._profile(tx.user_id)
        features = self.extract_features(tx)

        # While we have barely seen the user, both memories must converge fast
        # onto reality; otherwise the generic cold-start prior would look like
        # "drift" and every new customer would be flagged. Once enough history
        # exists the rates fall back to the configured (slow) values.
        n = profile.observed
        short_alpha = max(cfg.short_term_alpha, 1.0 / (n + 1))
        anchor_alpha = max(cfg.anchor_alpha, 1.0 / (n + 1))

        profile.sequence.append(features)
        profile.observed += 1

        for k, v in features.items():
            profile.short_term[k] = short_alpha * v + (1 - short_alpha) * profile.short_term[k]
            if allowed:
                profile.anchor[k] = anchor_alpha * v + (1 - anchor_alpha) * profile.anchor[k]

    def reset(self) -> None:
        self.profiles.clear()


# ===========================================================================
# STREAM B - relational fraud patterns  (GNN stand-in)
# ===========================================================================
class RelationalStream:
    """
    Simulated Graph Neural Network over a heterogeneous transaction graph.

    We maintain a real graph:

        user  --uses-->      device
        user  --pays-->      beneficiary
        user  --seen_at-->   location
        user  --from-->      ip

    and derive the structural signals a GNN would learn to recognise:
    device sharing (mule farms), beneficiary hubs (collection accounts),
    device churn, impossible travel, shared infrastructure, and - through a
    decayed 2-hop propagation - guilt by association with nodes previously
    involved in a BLOCKed transaction. That propagation is a hand-rolled,
    single-weight version of GNN message passing.
    """

    def __init__(self, config: GraphConfig | None = None) -> None:
        self.cfg = config or SETTINGS.graph
        self.reset()

    def reset(self) -> None:
        # adjacency: node key -> set of neighbour node keys
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)
        # risk written back onto nodes involved in blocked transactions
        self.node_risk: Dict[str, float] = defaultdict(float)
        # transfer history per (user, beneficiary) -> [(timestamp, amount)]
        self.transfers: Dict[Tuple[str, str], List[Tuple[float, float]]] = defaultdict(list)
        # per-user activity timestamps, for velocity
        self.activity: Dict[str, List[float]] = defaultdict(list)
        # last known (location, timestamp) per user, for impossible travel
        self.last_location: Dict[str, Tuple[str, float]] = {}

    # -- graph plumbing ------------------------------------------------------
    @staticmethod
    def _node(kind: str, value: str) -> str:
        return f"{kind}:{value}"

    def _link(self, a: str, b: str) -> None:
        self.adjacency[a].add(b)
        self.adjacency[b].add(a)

    def _neighbours_of_kind(self, node: str, kind: str) -> Set[str]:
        prefix = f"{kind}:"
        return {n for n in self.adjacency.get(node, set()) if n.startswith(prefix)}

    def _tx_nodes(self, tx: Transaction) -> List[str]:
        nodes = [
            self._node("user", tx.user_id),
            self._node("device", tx.device_id),
            self._node("location", tx.location),
        ]
        if tx.beneficiary_id:
            nodes.append(self._node("beneficiary", tx.beneficiary_id))
        if tx.ip_address:
            nodes.append(self._node("ip", tx.ip_address))
        return nodes

    def _propagated_risk(self, seeds: List[str]) -> float:
        """
        Decayed breadth-first risk propagation ("message passing").

        Hop 0 uses the node's own risk, hop 1 and 2 are damped by the decay
        factor, so a device two steps away from a known mule still raises the
        score, but less than a direct link.
        """
        cfg = self.cfg
        best = 0.0
        frontier = set(seeds)
        visited: Set[str] = set()
        damping = 1.0

        for _ in range(cfg.propagation_hops + 1):
            next_frontier: Set[str] = set()
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                best = max(best, self.node_risk.get(node, 0.0) * damping)
                next_frontier |= self.adjacency.get(node, set())
            frontier = next_frontier - visited
            damping *= cfg.propagation_decay
            if not frontier:
                break
        return clamp(best)

    # -- scoring (pure: never mutates state) ---------------------------------
    def score(self, tx: Transaction) -> StreamResult:
        cfg = self.cfg
        w = cfg.signal_weights

        user_n = self._node("user", tx.user_id)
        device_n = self._node("device", tx.device_id)
        location_n = self._node("location", tx.location)
        beneficiary_n = self._node("beneficiary", tx.beneficiary_id) if tx.beneficiary_id else None
        ip_n = self._node("ip", tx.ip_address) if tx.ip_address else None

        signals: List[Signal] = []
        now = tx.timestamp or 0.0

        # 1. device sharing - how many distinct users touched this device?
        device_users = self._neighbours_of_kind(device_n, "user") | {user_n}
        fanout = clamp((len(device_users) - 1) / cfg.fanout_saturation)
        if fanout > 0:
            signals.append(
                Signal("device_sharing", fanout, w["device_sharing"],
                       f"Device is shared by {len(device_users)} different accounts "
                       "(possible mule device)")
            )

        # 2. beneficiary hub - how many distinct users pay this beneficiary?
        fanin = 0.0
        if beneficiary_n:
            senders = self._neighbours_of_kind(beneficiary_n, "user") | {user_n}
            fanin = clamp((len(senders) - 1) / cfg.fanout_saturation)
            if fanin > 0:
                signals.append(
                    Signal("beneficiary_hub", fanin, w["beneficiary_hub"],
                           f"Suspicious beneficiary relationship: {len(senders)} accounts "
                           "send money to this same payee")
                )

        # 3. device churn - one user cycling through many devices
        user_devices = self._neighbours_of_kind(user_n, "device") | {device_n}
        churn = clamp(
            (len(user_devices) - cfg.device_churn_baseline) / cfg.device_churn_saturation
        )
        if churn > 0:
            signals.append(
                Signal("device_churn", churn, w["device_churn"],
                       f"User has now been seen on {len(user_devices)} devices")
            )

        # 4. new user->device edge (device fingerprinting)
        new_device = 1.0 if (device_n not in self.adjacency.get(user_n, set())) else 0.0
        if tx.device_change:
            new_device = 1.0
        if new_device:
            signals.append(
                Signal("new_device_link", 1.0, w["new_device_link"],
                       "New device detected for this account")
            )

        # 5. new / very young user->beneficiary edge. A relationship stays
        #    "young" for a while instead of being trusted after one payment,
        #    which is what makes a burst to a fresh payee stand out.
        new_beneficiary = 0.0
        beneficiary_message = "Money is going to a newly added beneficiary"
        if beneficiary_n:
            if beneficiary_n not in self.adjacency.get(user_n, set()):
                new_beneficiary = 1.0
            else:
                history = self.transfers.get((tx.user_id, tx.beneficiary_id or ""), [])
                if history and (now - min(ts for ts, _ in history)
                                <= cfg.new_relationship_window_seconds):
                    new_beneficiary = 0.6
                    beneficiary_message = (
                        "Beneficiary relationship was created less than "
                        f"{cfg.new_relationship_window_seconds // 3600}h ago"
                    )
        if tx.beneficiary_change:
            new_beneficiary = 1.0
            beneficiary_message = "Money is going to a newly added beneficiary"
        if new_beneficiary:
            signals.append(
                Signal("new_beneficiary_link", new_beneficiary,
                       w["new_beneficiary_link"], beneficiary_message)
            )

        # 6. location anomaly - unseen location, or impossible travel
        location_signal = 0.0
        location_message = ""
        if location_n not in self.adjacency.get(user_n, set()) and self.adjacency.get(user_n):
            location_signal = 0.6
            location_message = f"Unusual location for this account ({tx.location})"
        previous = self.last_location.get(tx.user_id)
        if previous and previous[0] != tx.location:
            if now - previous[1] < cfg.impossible_travel_seconds:
                location_signal = 1.0
                location_message = (
                    f"Impossible travel: {previous[0]} -> {tx.location} within the hour"
                )
        if location_signal > 0:
            signals.append(
                Signal("location_anomaly", location_signal, w["location_anomaly"],
                       location_message or f"Unusual location ({tx.location})")
            )

        # 7. shared IP infrastructure
        shared_ip = 0.0
        if ip_n:
            ip_users = self._neighbours_of_kind(ip_n, "user") | {user_n}
            shared_ip = clamp((len(ip_users) - 1) / cfg.fanout_saturation)
            if shared_ip > 0:
                signals.append(
                    Signal("shared_ip", shared_ip, w["shared_ip"],
                           f"IP address shared with {len(ip_users) - 1} other account(s)")
                )

        # 8. neighbourhood risk propagated from previously blocked nodes
        neighbourhood = self._propagated_risk(self._tx_nodes(tx))
        if neighbourhood > 0.01:
            signals.append(
                Signal("neighbourhood_risk", neighbourhood, w["neighbourhood_risk"],
                       "Connected (within 2 hops) to entities involved in blocked fraud")
            )

        # 9. structuring - repeated sub-threshold transfers to the same payee
        structuring = 0.0
        structuring_count = 0
        structuring_total = 0.0
        if beneficiary_n and tx.amount <= cfg.structuring_amount_ceiling:
            history = self.transfers.get((tx.user_id, tx.beneficiary_id or ""), [])
            small = [
                (ts, amt) for ts, amt in history
                if now - ts <= cfg.structuring_window_seconds
                and amt <= cfg.structuring_amount_ceiling
            ]
            structuring_count = len(small) + 1
            structuring_total = sum(a for _, a in small) + tx.amount
            if (
                structuring_count >= cfg.structuring_min_transfers
                and structuring_total >= cfg.structuring_min_total
            ):
                structuring = clamp(structuring_count / cfg.structuring_saturation)
                signals.append(
                    Signal("structuring_pattern", structuring, w["structuring_pattern"],
                           f"Structuring pattern: {structuring_count} sub-threshold "
                           f"transfers to the same payee totalling "
                           f"{structuring_total:,.0f} within "
                           f"{cfg.structuring_window_seconds // 3600}h")
                )

        # 10. velocity - burst of activity from one user
        recent_activity = [
            ts for ts in self.activity.get(tx.user_id, [])
            if now - ts <= cfg.velocity_window_seconds
        ]
        velocity = clamp(
            (len(recent_activity) + 1 - cfg.velocity_baseline) / cfg.velocity_saturation
        )
        if velocity > 0:
            signals.append(
                Signal("velocity", velocity, w["velocity"],
                       f"Abnormal transaction frequency: {len(recent_activity) + 1} "
                       "transactions in the last hour")
            )

        # --- aggregate ------------------------------------------------------
        total_weight = sum(w.values()) or 1.0
        weighted_sum = sum(s.value * s.weight for s in signals)
        score = squash(weighted_sum / total_weight, cfg.graph_gain)

        return StreamResult(
            score=score,
            signals=signals,
            details={
                "weighted_sum": round(weighted_sum, 4),
                "device_fanout": float(len(device_users)),
                "beneficiary_fanin": float(fanin),
                "neighbourhood_risk": round(neighbourhood, 4),
                "structuring_count": float(structuring_count),
                "structuring_total": round(structuring_total, 2),
                "velocity_count": float(len(recent_activity) + 1),
                "graph_nodes": float(len(self.adjacency)),
            },
        )

    # -- state update --------------------------------------------------------
    def commit(self, tx: Transaction, blocked: bool) -> None:
        """Insert the transaction's edges, and propagate risk if it was blocked."""
        now = tx.timestamp or 0.0
        user_n = self._node("user", tx.user_id)

        for node in self._tx_nodes(tx):
            if node != user_n:
                self._link(user_n, node)

        if tx.beneficiary_id:
            self.transfers[(tx.user_id, tx.beneficiary_id)].append((now, tx.amount))
        self.activity[tx.user_id].append(now)
        self.last_location[tx.user_id] = (tx.location, now)

        if blocked:
            # Guilt propagation: the entities used in a blocked transaction
            # become risky nodes for every future graph query.
            for node in self._tx_nodes(tx):
                if node.startswith("user:"):
                    continue  # don't permanently condemn the account itself
                self.node_risk[node] = max(
                    self.node_risk[node], self.cfg.blocked_node_risk
                )

    # -- synthetic demo data -------------------------------------------------
    def seed_demo_graph(self) -> None:
        """
        Load a tiny SYNTHETIC relationship graph so the GNN-like stream has
        something to reason about on the very first request of a demo.

        This is fabricated data for the presentation only - no real users,
        no external dataset, no API key.
        """
        base_ts = 0.0

        # A shared "mule" device used by several throw-away accounts.
        for user in ("u_muleA", "u_muleB", "u_muleC"):
            self._link(self._node("user", user), self._node("device", "dev_shared_99"))
            self._link(self._node("user", user), self._node("ip", "203.0.113.77"))

        # A collection account fed by several unrelated users.
        for user in ("u_muleA", "u_muleB", "u_muleC", "u_victim7"):
            self._link(self._node("user", user), self._node("beneficiary", "ben_mule_77"))

        # These entities were involved in previously blocked fraud.
        self.node_risk[self._node("device", "dev_shared_99")] = 0.85
        self.node_risk[self._node("beneficiary", "ben_mule_77")] = 0.90

        # Normal, well-established customers used by the demo presets.
        established = {
            "u_1001": ("dev_trusted_01", "Colombo, LK", "ben_family_02", "192.168.1.20"),
            "u_2002": ("dev_phone_22", "Colombo, LK", "ben_utility_22", "192.168.2.22"),
        }
        for user, (device, location, beneficiary, ip) in established.items():
            node = self._node("user", user)
            self._link(node, self._node("device", device))
            self._link(node, self._node("location", location))
            self._link(node, self._node("beneficiary", beneficiary))
            self._link(node, self._node("ip", ip))
            self.last_location[user] = (location, base_ts)
            # An old payment history, so these payees are trusted relationships
            # rather than links that were created moments ago.
            self.transfers[(user, beneficiary)].append((base_ts, 1_200.0))


# ===========================================================================
# FUSION
# ===========================================================================
class DualStreamFraudModel:
    """
    Late-fusion wrapper around the two streams.

    Swap either stream for a real trained model by passing an object that
    implements `score(tx) -> StreamResult` and `commit(tx, flag)`.
    """

    def __init__(
        self,
        behavioral: Optional[BehavioralStream] = None,
        relational: Optional[RelationalStream] = None,
        fusion: Optional[FusionConfig] = None,
        seed_graph: bool = True,
    ) -> None:
        self.behavioral = behavioral or BehavioralStream()
        self.relational = relational or RelationalStream()
        self.fusion = fusion or SETTINGS.fusion
        if seed_graph:
            self.relational.seed_demo_graph()

    def fuse(self, behavioral_score: float, graph_score: float) -> float:
        """
        Weighted sum plus a corroboration bonus.

        The bonus is what makes the dual-stream design resilient: an attacker
        who perfectly imitates normal behaviour still carries their
        relationships with them, and vice-versa.
        """
        f = self.fusion
        risk = (
            f.behavioral_weight * behavioral_score
            + f.graph_weight * graph_score
            + f.corroboration_bonus * behavioral_score * graph_score
        )
        return clamp(risk)

    def analyze(self, tx: Transaction) -> ModelOutput:
        """Score a transaction. Pure - call `commit()` to update memory."""
        behavioral = self.behavioral.score(tx)
        relational = self.relational.score(tx)
        risk = self.fuse(behavioral.score, relational.score)

        context: Dict[str, float] = {}
        context.update({f"behavioral_{k}": v for k, v in behavioral.details.items()})
        context.update({f"graph_{k}": v for k, v in relational.details.items()})

        return ModelOutput(
            behavioral_score=round(behavioral.score, 4),
            graph_score=round(relational.score, 4),
            risk_score=round(risk, 4),
            behavioral_signals=behavioral.signals,
            graph_signals=relational.signals,
            context=context,
        )

    def commit(self, tx: Transaction, decision: str) -> None:
        """Update both streams once the gateway has ruled on the transaction."""
        self.behavioral.commit(tx, allowed=(decision == "ALLOW"))
        self.relational.commit(tx, blocked=(decision == "BLOCK"))

    def reset(self, seed_graph: bool = True) -> None:
        """Wipe all learned state - handy between live demo runs."""
        self.behavioral.reset()
        self.relational.reset()
        if seed_graph:
            self.relational.seed_demo_graph()
