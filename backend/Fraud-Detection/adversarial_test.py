"""
adversarial_test.py
===================
SAFE, FULLY SIMULATED adversarial robustness harness.

WHAT THIS IS
------------
A research experiment that generates synthetic transaction sequences shaped
like three evasion strategies described in our project, then measures how the
dual-stream score and the deterministic gateway respond at every stage.

    1. CAMOUFLAGE  - the attacker keeps the fraudulent relationships but
                     imitates the victim's normal behaviour, trying to drive
                     the behavioural stream to zero.
    2. SLOW DRIFT  - the attacker changes behaviour in steps too small to
                     trigger an alert, hoping the profile follows them.
    3. STRUCTURING - the attacker splits one large transfer into many small
                     ones, each individually unremarkable.

WHAT THIS IS NOT
----------------
This file contains NO attack tooling. It does not touch any network, any real
bank, any real account or any external dataset. It only builds dictionaries of
made-up numbers and feeds them to our own scoring function, in-process. It is
the equivalent of a unit test for model robustness.

Run standalone:      python adversarial_test.py
Or through the API:  POST /simulate-attack {"attack_type": "camouflage"}
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from fraud_model import MODEL_MODE, DualStreamFraudModel
from gateway import DeterministicGateway
from pipeline import FraudPipeline
from schemas import ScoreResponse, Transaction

# Fixed clock so every run of the experiment is reproducible.
BASE_TS = 1_700_000_000.0
HOUR = 3600.0

DISCLAIMER = (
    "Simulated scenario on synthetic data. No real accounts, no network "
    "activity, and no trained neural network are involved."
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _fresh_pipeline() -> FraudPipeline:
    """
    An isolated engine per experiment.

    The attack simulation must never contaminate the state of the live demo,
    so it gets its own model instance (with the same seeded demo graph) and no
    audit sink.
    """
    return FraudPipeline(
        model=DualStreamFraudModel(),
        gateway=DeterministicGateway(),
        audit=None,
    )


def _row(step_no: int, label: str, amount: float, response: ScoreResponse) -> Dict[str, Any]:
    """Flatten a pipeline response into one row of the experiment table."""
    rule_ids = [rule.split(":")[0] for rule in response.rules_fired]
    prefix = f"[{'+'.join(rule_ids)}] " if rule_ids else ""
    reason = response.reasons[0] if response.reasons else response.reason
    return {
        "step": step_no,
        "label": label,
        "amount": round(amount, 2),
        "behavioral_score": response.behavioral_score,
        "graph_score": response.graph_score,
        "risk_score": response.risk_score,
        "decision": response.decision,
        "reason": prefix + reason,
    }


def _behaviour_only_decision(behavioral_score: float) -> str:
    """
    What a SINGLE-STREAM (behaviour-only) system would have decided.

    Used to quantify the benefit of the dual-stream design: if this says ALLOW
    while the full engine says STEP-UP or BLOCK, the relational stream is what
    saved the transaction.
    """
    decision, _ = DeterministicGateway().classify_band(behavioral_score)
    return decision.value


def _warm_up(
    pipeline: FraudPipeline,
    *,
    user_id: str,
    device_id: str,
    location: str,
    beneficiary_id: str,
    ip_address: str,
    account_age: int,
    amounts: List[float],
    typing_speed: float = 240.0,
    start_ts: float = BASE_TS,
    spacing: float = 30 * HOUR,
) -> None:
    """
    Teach the engine what this user's normal life looks like.

    Transactions are spaced well apart so that the velocity and structuring
    detectors (which are short-window) stay silent during the warm-up.
    """
    previous = amounts[0]
    for index, amount in enumerate(amounts):
        pipeline.evaluate(
            Transaction(
                transaction_id=f"{user_id}_warmup_{index}",
                user_id=user_id,
                amount=amount,
                location=location,
                device_id=device_id,
                device_change=False,
                typing_speed=typing_speed,
                navigation_pattern="normal",
                transaction_frequency=3,
                beneficiary_change=False,
                beneficiary_id=beneficiary_id,
                ip_address=ip_address,
                previous_transaction_amount=previous,
                account_age=account_age,
                timestamp=start_ts + index * spacing,
            ),
            persist_state=True,
        )
        previous = amount


# ---------------------------------------------------------------------------
# 1. CAMOUFLAGE ATTACK
# ---------------------------------------------------------------------------
def camouflage_attack(steps: int = 8) -> Dict[str, Any]:
    """
    The attacker progressively hides their behaviour while still using the
    same mule device and the same collection account.

    Expected research finding: the behavioural score collapses towards zero,
    but the relational score is untouched, because relationships cannot be
    faked as easily as typing speed. The fused score therefore stays in an
    actionable band and the gateway still challenges or blocks.
    """
    pipeline = _fresh_pipeline()
    user_id = "u_attacker_07"

    def attempt(*, amount, typing, nav, freq, age, device_change, beneficiary_change, ts):
        return pipeline.evaluate(
            Transaction(
                user_id=user_id,
                amount=amount,
                location="Unknown, XX",
                device_id="dev_shared_99",        # known mule device
                device_change=device_change,
                typing_speed=typing,
                navigation_pattern=nav,
                transaction_frequency=freq,
                beneficiary_change=beneficiary_change,
                beneficiary_id="ben_mule_77",     # known collection account
                ip_address="203.0.113.77",
                previous_transaction_amount=2000.0,
                account_age=age,
                timestamp=ts,
            ),
            # Each camouflage level is an independent what-if against the same
            # starting state, so nothing is committed to memory.
            persist_state=False,
        )

    baseline_response = attempt(
        amount=480_000.0, typing=780.0, nav="automated", freq=18, age=4,
        device_change=True, beneficiary_change=True, ts=BASE_TS,
    )
    baseline = _row(0, "No camouflage (blatant fraud attempt)", 480_000.0, baseline_response)

    # Navigation labels the attacker cycles through as they get more careful.
    nav_ladder = ["automated", "erratic", "rapid", "exploratory", "normal"]

    rows: List[Dict[str, Any]] = []
    for i in range(1, steps + 1):
        t = i / steps  # 0 -> 1 : camouflage effort

        # Geometric interpolation of the amount: 480,000 down to 4,500.
        amount = (480_000.0 ** (1 - t)) * (4_500.0 ** t)
        typing = 780.0 * (1 - t) + 250.0 * t
        nav = nav_ladder[min(int(t * len(nav_ladder)), len(nav_ladder) - 1)]
        freq = int(round(18 * (1 - t) + 3 * t))
        age = int(round(4 * (1 - t) + 400 * t))   # switch to an aged mule account
        hide_newness = t >= 0.5                   # payee/device "added long ago"

        response = attempt(
            amount=amount, typing=typing, nav=nav, freq=freq, age=age,
            device_change=not hide_newness, beneficiary_change=not hide_newness,
            ts=BASE_TS + i * HOUR,
        )
        rows.append(_row(i, f"Camouflage level {i}/{steps}", amount, response))

    final = rows[-1]
    behaviour_only = _behaviour_only_decision(final["behavioral_score"])
    missed = behaviour_only == "ALLOW"
    detected = final["decision"] != "ALLOW"

    summary = (
        f"Behavioural score fell from {baseline['behavioral_score']:.2f} to "
        f"{final['behavioral_score']:.2f} as the attacker imitated normal usage, "
        f"but the relational score barely moved ({baseline['graph_score']:.2f} -> "
        f"{final['graph_score']:.2f}) because the mule device and the collection "
        f"account are unchanged. Final fused risk {final['risk_score']:.2f} -> "
        f"gateway decision {final['decision']}. A behaviour-only model would have "
        f"decided {behaviour_only}."
    )

    return {
        "attack_type": "camouflage",
        "user_id": user_id,
        "title": "Camouflage attack",
        "description": (
            "The attacker keeps the fraudulent relationships (shared mule device, "
            "known collection account, shared IP) but progressively imitates the "
            "victim's normal behaviour to drive the behavioural anomaly score down."
        ),
        "baseline": baseline,
        "steps": rows,
        "summary": summary,
        "detected": detected,
        "single_stream_would_have_missed": missed,
        "disclaimer": DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# 2. SLOW-DRIFT ATTACK
# ---------------------------------------------------------------------------
def slow_drift_attack(steps: int = 10) -> Dict[str, Any]:
    """
    A compromised account is nudged a little further from normal each day, so
    that no single transaction looks anomalous.

    Expected research finding: the reconstruction error stays small (each step
    is close to the previous one) but the DRIFT term - the distance between
    the fast short-term memory and the slow, poison-resistant anchor profile -
    keeps growing, so the risk score climbs steadily until the gateway steps in.
    """
    pipeline = _fresh_pipeline()
    user_id = "u_drift_11"
    device_id = "dev_home_11"
    location = "Kandy, LK"
    beneficiary_id = "ben_landlord_11"
    ip_address = "192.168.4.11"

    _warm_up(
        pipeline,
        user_id=user_id,
        device_id=device_id,
        location=location,
        beneficiary_id=beneficiary_id,
        ip_address=ip_address,
        account_age=900,
        # A long, boring history so the profile is mature before the attack.
        amounts=[
            2400.0, 2600.0, 2500.0, 2450.0, 2550.0, 2500.0,
            2480.0, 2520.0, 2460.0, 2540.0, 2500.0, 2510.0,
        ],
    )

    drift_start = BASE_TS + 10 * 30 * HOUR

    def transfer(amount, typing, nav, freq, ts, previous, persist=True):
        return pipeline.evaluate(
            Transaction(
                user_id=user_id,
                amount=amount,
                location=location,
                device_id=device_id,
                device_change=False,
                typing_speed=typing,
                navigation_pattern=nav,
                transaction_frequency=freq,
                beneficiary_change=False,
                beneficiary_id=beneficiary_id,
                ip_address=ip_address,
                previous_transaction_amount=previous,
                account_age=900,
                timestamp=ts,
            ),
            persist_state=persist,
        )

    baseline_response = transfer(2500.0, 240.0, "normal", 3, drift_start, 2500.0, persist=False)
    baseline = _row(0, "Established normal behaviour (after warm-up)", 2500.0, baseline_response)

    rows: List[Dict[str, Any]] = []
    previous = 2500.0
    for i in range(1, steps + 1):
        t = i / steps
        # Every individual step is a small, continuous change from the one
        # before it. The navigation label never changes, so there is no single
        # discrete event for a conventional rule engine to trigger on.
        amount = 2500.0 * (1.0 + 23.0 * t)          # 2,500 -> 60,000
        typing = 240.0 + 230.0 * t                  # 240   -> 470 keystrokes/min
        freq = 3 + int(round(10 * t))               # 3     -> 13 per day

        response = transfer(
            amount, typing, "normal", freq, drift_start + i * 30 * HOUR, previous
        )
        previous = amount
        rows.append(_row(i, f"Drift day {i}/{steps}", amount, response))

    escalated_at = next(
        (r["step"] for r in rows if r["decision"] != "ALLOW"), None
    )
    final = rows[-1]
    summary = (
        f"Risk rose gradually from {baseline['risk_score']:.2f} to "
        f"{final['risk_score']:.2f} across {steps} small steps, with no single "
        f"step looking dramatic. "
        + (
            f"The gateway escalated at step {escalated_at} "
            f"({rows[escalated_at - 1]['decision']}) and ended at "
            f"{final['decision']}."
            if escalated_at
            else "The gateway did not escalate within this window."
        )
        + " The drift term compares the fast short-term memory with the slow "
        "anchor profile, and the anchor only learns from ALLOWed transactions, "
        "so the attacker cannot redefine 'normal' for the account."
    )

    return {
        "attack_type": "slow_drift",
        "user_id": user_id,
        "title": "Slow-drift attack",
        "description": (
            "Behaviour is shifted a small amount every day - larger transfers, "
            "faster typing, more frequent activity - so that no single transaction "
            "looks anomalous to a short-memory detector."
        ),
        "baseline": baseline,
        "steps": rows,
        "summary": summary,
        "detected": escalated_at is not None,
        "single_stream_would_have_missed": False,
        "disclaimer": DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# 3. STRUCTURING ATTACK
# ---------------------------------------------------------------------------
def structuring_attack(steps: int = 8) -> Dict[str, Any]:
    """
    One large transfer is split into many smaller ones to stay under the
    reporting/limit threshold.

    Expected research finding: each individual transfer is unremarkable to the
    behavioural stream, but the relational stream accumulates the repeated
    user -> beneficiary edge, and the DETERMINISTIC GATEWAY applies the same
    aggregate value limit that a single large transfer would have faced
    (rule G7). This is the clearest demonstration of why enforcement must not
    be left to the model's score alone.
    """
    pipeline = _fresh_pipeline()
    user_id = "u_biz_21"
    device_id = "dev_office_21"
    location = "Colombo, LK"
    ip_address = "192.168.9.21"
    target = "ben_offshore_21"
    split_amount = 45_000.0
    lump_sum = split_amount * steps

    # A small business that normally moves about 30,000 per transfer.
    _warm_up(
        pipeline,
        user_id=user_id,
        device_id=device_id,
        location=location,
        beneficiary_id="ben_supplier_A",
        ip_address=ip_address,
        account_age=1200,
        amounts=[
            30_000.0, 28_000.0, 32_000.0, 30_500.0, 29_500.0,
            31_000.0, 29_000.0, 30_200.0, 30_800.0, 29_800.0,
            30_400.0, 29_600.0,
        ],
    )

    attack_start = BASE_TS + 10 * 30 * HOUR

    # --- baseline: the attacker tries the whole amount in one go -----------
    baseline_response = pipeline.evaluate(
        Transaction(
            user_id=user_id,
            amount=lump_sum,
            location=location,
            device_id=device_id,
            device_change=False,
            typing_speed=250.0,
            navigation_pattern="normal",
            transaction_frequency=4,
            beneficiary_change=True,
            beneficiary_id=target,
            ip_address=ip_address,
            previous_transaction_amount=29_500.0,
            account_age=1200,
            timestamp=attack_start,
        ),
        persist_state=False,  # what-if only: the lump sum is never committed
    )
    baseline = _row(
        0, f"Single lump-sum transfer of {lump_sum:,.0f}", lump_sum, baseline_response
    )

    # --- the split attempts -------------------------------------------------
    rows: List[Dict[str, Any]] = []
    previous = 29_500.0
    for i in range(1, steps + 1):
        response = pipeline.evaluate(
            Transaction(
                user_id=user_id,
                amount=split_amount,
                location=location,
                device_id=device_id,
                device_change=False,
                typing_speed=250.0,
                navigation_pattern="normal",
                transaction_frequency=4 + i,
                beneficiary_change=(i == 1),
                beneficiary_id=target,
                ip_address=ip_address,
                previous_transaction_amount=previous,
                account_age=1200,
                timestamp=attack_start + i * 12 * 60,  # one every 12 minutes
            ),
            persist_state=True,
        )
        previous = split_amount
        rows.append(
            _row(i, f"Split transfer {i}/{steps} of {split_amount:,.0f}", split_amount, response)
        )

    caught_at = next((r["step"] for r in rows if r["decision"] != "ALLOW"), None)
    blocked_at = next((r["step"] for r in rows if r["decision"] == "BLOCK"), None)
    final = rows[-1]

    summary = (
        f"A single transfer of {lump_sum:,.0f} is {baseline['decision']}. Split into "
        f"{steps} transfers of {split_amount:,.0f}, the first ones look normal "
        f"(behavioural score around {rows[0]['behavioral_score']:.2f}). "
        + (
            f"The relational stream starts flagging the repeated payee at step "
            f"{caught_at}, "
            if caught_at
            else "The relational stream did not flag the pattern, "
        )
        + (
            f"and the deterministic gateway blocks from step {blocked_at} onwards, "
            f"once the running total reaches the same hard limit that applies to a "
            f"single transfer."
            if blocked_at
            else f"and the run ended at {final['decision']}."
        )
    )

    return {
        "attack_type": "structuring",
        "user_id": user_id,
        "title": "Structuring (smurfing) attack",
        "description": (
            "One large transfer is broken into many smaller transfers to the same "
            "beneficiary, each small enough to look ordinary on its own."
        ),
        "baseline": baseline,
        "steps": rows,
        "summary": summary,
        "detected": caught_at is not None,
        "single_stream_would_have_missed": bool(
            rows and _behaviour_only_decision(final["behavioral_score"]) == "ALLOW"
        ),
        "disclaimer": DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
ATTACKS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "camouflage": camouflage_attack,
    "slow_drift": slow_drift_attack,
    "structuring": structuring_attack,
}


def run_attack(attack_type: str, steps: Optional[int] = None) -> Dict[str, Any]:
    """Run one simulation by name. Raises KeyError for an unknown name."""
    key = (attack_type or "").strip().lower()
    if key not in ATTACKS:
        raise KeyError(
            f"Unknown attack type '{attack_type}'. Choose one of: {sorted(ATTACKS)}"
        )
    return ATTACKS[key](steps) if steps else ATTACKS[key]()


# ---------------------------------------------------------------------------
# console report:  python adversarial_test.py
# ---------------------------------------------------------------------------
def _print_report(result: Dict[str, Any]) -> None:
    line = "-" * 104
    print()
    print("=" * 104)
    print(f" {result['title'].upper()}  [{result['attack_type']}]")
    print("=" * 104)
    print(f" {result['description']}")
    print(line)
    header = (
        f"{'#':>3} | {'Stage':<44} | {'Behav':>6} | {'Graph':>6} | "
        f"{'Risk':>6} | {'Decision':<8}"
    )
    print(header)
    print(line)

    for row in [result["baseline"], *result["steps"]]:
        print(
            f"{row['step']:>3} | {row['label'][:44]:<44} | "
            f"{row['behavioral_score']:>6.3f} | {row['graph_score']:>6.3f} | "
            f"{row['risk_score']:>6.3f} | {row['decision']:<8}"
        )
        print(f"    | reason: {row['reason'][:94]}")

    print(line)
    print(f" RESULT   : {'DETECTED' if result['detected'] else 'NOT DETECTED'}")
    if result.get("single_stream_would_have_missed"):
        print(" NOTE     : a behaviour-only (single-stream) model would have ALLOWED this.")
    print(f" FINDING  : {result['summary']}")
    print(line)


def main() -> None:
    print()
    print("#" * 104)
    print("#  ADVERSARIAL ROBUSTNESS SIMULATION")
    print(f"#  Model mode: {MODEL_MODE}")
    print(f"#  {DISCLAIMER}")
    print("#" * 104)

    for name in ("camouflage", "slow_drift", "structuring"):
        _print_report(run_attack(name))

    print()
    print("All three simulations completed.")
    print()


if __name__ == "__main__":
    main()
