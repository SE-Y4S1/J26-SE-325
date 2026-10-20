"""Score one transaction and apply the deterministic gateway."""

from __future__ import annotations

import json
import threading
from collections import OrderedDict, defaultdict, deque
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

from fraud_engine.gateway.drift import DriftMonitor
from fraud_engine.gateway.feedback import FeedbackStore
from fraud_engine.gateway.policy import enforce
from fraud_engine.gateway.ratelimit import RateLimiter
from fraud_engine.models.bundle import fusion_from_bundle, load_bundle, load_gnn, load_lstm
from fraud_engine.models.gnn import predict_proba
from fraud_engine.models.lstm_ae import reconstruction_errors
from fraud_engine.paths import LOG_DIR

BANNED_FIELDS = {
    "email",
    "full_name",
    "name",
    "first_name",
    "last_name",
    "phone",
    "ssn",
    "pan",
    "card_number",
    "password",
    "ip",
    "ip_address",
    "address",
}


class FraudService:
    def __init__(self, settings: dict[str, Any], bundle_path: Path | None = None) -> None:
        self.settings = settings
        self.lock = threading.Lock()
        gateway = settings["gateway"]
        self.limiter = RateLimiter(
            int(gateway["rate_limit"]),
            float(gateway["rate_window_seconds"]),
            float(gateway["rate_limit_block_multiplier"]),
        )
        self.drift = DriftMonitor(None, float(gateway["drift_delta"]), int(gateway["drift_window"]))
        self.feedback = FeedbackStore(minimum_samples=int(gateway["feedback_min_samples"]))
        self.history: dict[str, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=32))
        self.log_dir = Path(settings["log_dir"]) if settings.get("log_dir") else LOG_DIR
        self.recent: deque[dict[str, Any]] = deque(maxlen=50)
        self.decision_counts = {"ALLOW": 0, "STEP_UP": 0, "BLOCK": 0}
        # Full responses by transaction id, so the explanation layer (Component 4) can read
        # the decision this gateway enforced. Holds only what score() returned: no account
        # key, device key or amount, matching what the decision log keeps.
        self.responses: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.max_responses = 1000
        self.bundle = None
        self.preprocessor = None
        self.lstm = None
        self.gnn = None
        self.fusion = None
        self.logistic = None
        self.thresholds = {
            "step_up": 0.55,
            "block": 0.80,
            "step_up_fpr_budget": gateway["step_up_fpr"],
            "block_fpr_budget": gateway["block_fpr"],
            "source": "heuristic_until_a_model_is_trained",
        }
        self.model_version = "heuristic-untrained"
        self.benchmark_model = False
        if bundle_path and bundle_path.is_file():
            self._load(bundle_path)
        self._bind_drift_reference()

    def _load(self, path: Path) -> None:
        from fraud_engine.data.preprocess import TabularPreprocessor

        bundle = load_bundle(path)
        self.bundle = bundle
        self.preprocessor = TabularPreprocessor.from_dict(bundle["preprocessor"])
        self.lstm = load_lstm(bundle)
        self.gnn = load_gnn(bundle)
        self.fusion = fusion_from_bundle(bundle)
        self.logistic = bundle.get("logistic_model")
        self.thresholds = dict(bundle["thresholds"])
        self.model_version = str(bundle.get("model_version", "trained-bundle"))
        self.benchmark_model = bool(bundle.get("benchmark_valid"))

    def _bind_drift_reference(self) -> None:
        """Compare new scores with the legitimate step-up cut.

        That cut is already chosen from legitimate validation scores. Leaving the
        reference empty made concept_drift_step_up unreachable, including after a
        trained bundle was loaded.
        """
        step_up = self.thresholds.get("step_up")
        if step_up is None:
            return
        self.drift.set_reference(np.array([float(step_up)], dtype=float))

    def score(self, payload: dict[str, Any], now: float | None = None) -> dict[str, Any]:
        _reject_sensitive(payload)
        simulated = payload.get("simulated_signals")
        if simulated is not None and simulated.get("origin") != "simulated":
            raise ValueError("Simulated behavioural signals must set origin to 'simulated'.")
        amount = float(payload["amount"])
        account = payload.get("account_key") or payload["transaction_id"]
        clock = float(now if now is not None else datetime.now(timezone.utc).timestamp())
        with self.lock:
            rate_status = self.limiter.observe(str(account), clock)
            fraud_score, streams, factors, notes = self._model_score(payload, str(account))
            drift_flag = self.drift.update(fraud_score)
            decision, reasons = enforce(
                fraud_score,
                self.thresholds,
                amount,
                float(self.settings["gateway"]["hard_block_amount"]),
                payload.get("ip_reputation"),
                float(self.settings["gateway"]["reputation_step_up_below"]),
                float(self.settings["gateway"]["reputation_block_below"]),
                rate_status,
                drift_flag,
                bool(self.settings["gateway"]["drift_forces_step_up"]),
            )
            self.decision_counts[decision] += 1
            record = {
                "transaction_id": payload["transaction_id"],
                "scored_at": datetime.fromtimestamp(clock, timezone.utc).isoformat(),
                "decision": decision,
                "fraud_score": fraud_score,
                "model_version": self.model_version,
                "reasons": reasons,
            }
            self.recent.append(record)
            self._log(record)
        if simulated is not None:
            notes.append("Simulated signals were received and were not treated as customer biometrics.")
        response = {
            "transaction_id": payload["transaction_id"],
            "fraud_score": fraud_score,
            "decision": decision,
            "model_version": self.model_version,
            "benchmark_model": self.benchmark_model,
            "explanation": {
                "rules": reasons,
                "streams": streams,
                "top_factors": factors,
                "notes": notes,
            },
        }
        with self.lock:
            self.responses[str(payload["transaction_id"])] = response
            self.responses.move_to_end(str(payload["transaction_id"]))
            while len(self.responses) > self.max_responses:
                self.responses.popitem(last=False)
        return response

    def lookup(self, transaction_id: str) -> dict[str, Any] | None:
        with self.lock:
            return self.responses.get(transaction_id)

    def _model_score(self, payload: dict[str, Any], account: str) -> tuple[float, dict[str, float], list[dict[str, Any]], list[str]]:
        if self.preprocessor is None or self.lstm is None or self.gnn is None or self.fusion is None:
            return self._heuristic(payload)
        record = dict(payload.get("features") or {})
        record.update(payload.get("categories") or {})
        record.setdefault("TransactionAmt", payload["amount"])
        if payload.get("product_code"):
            record.setdefault("ProductCD", payload["product_code"])
        vector = self.preprocessor.transform_record(record)
        missing = [name for name in self.preprocessor.feature_names if _raw_name(name) not in record and name not in record]
        notes = []
        if missing:
            notes.append(
                f"{len(missing)} model features were missing and were replaced with training medians or unseen-category codes."
            )
        history = self.history[account]
        seq_len = int(self.bundle["seq_len"])
        window = list(history)[-(seq_len - 1) :] + [vector]
        sequences = np.zeros((1, seq_len, vector.shape[0]), dtype=np.float32)
        masks = np.zeros((1, seq_len), dtype=np.float32)
        offset = seq_len - len(window)
        for step, item in enumerate(window):
            sequences[0, offset + step] = item
            masks[0, offset + step] = 1.0
        errors = reconstruction_errors(self.lstm, sequences, masks, batch_size=1)
        low = float(self.bundle["lstm_norm"]["low"])
        high = float(self.bundle["lstm_norm"]["high"])
        lstm_score = float(np.clip((errors[0] - low) / max(high - low, 1e-6), 0.0, 1.0))
        nodes = list(history)[-int(self.settings["models"]["max_neighbors"]) :] + [vector]
        node_matrix = np.stack(nodes).astype(np.float32)
        if len(nodes) == 1:
            edge_index = np.zeros((2, 0), dtype=np.int64)
        else:
            destination = len(nodes) - 1
            sources = np.arange(destination, dtype=np.int64)
            targets = np.full(destination, destination, dtype=np.int64)
            edge_index = np.vstack([sources, targets])
        gnn_scores = predict_proba(self.gnn, node_matrix, edge_index)
        gnn_score = float(gnn_scores[-1])
        fused = float(self.fusion.score(np.array([lstm_score]), np.array([gnn_score]))[0])
        history.append(vector)
        factors = _linear_factors(self.logistic, self.preprocessor.feature_names, vector)
        return fused, {"lstm": lstm_score, "gnn": gnn_score, "fused": fused}, factors, notes

    def _heuristic(self, payload: dict[str, Any]) -> tuple[float, dict[str, float], list[dict[str, Any]], list[str]]:
        hard = float(self.settings["gateway"]["hard_block_amount"])
        score = min(1.0, float(payload["amount"]) / hard)
        reputation = payload.get("ip_reputation")
        if reputation is not None:
            score = max(score, 1.0 - float(reputation))
        notes = [
            "No trained artifact is loaded. This score is a fixed heuristic, not a benchmark model.",
            "Train with scripts/train_benchmark.py after the public dataset is available.",
        ]
        return float(score), {"heuristic": float(score)}, [{"name": "amount", "contribution": float(score)}], notes

    def apply_feedback(self) -> dict[str, Any]:
        updated = self.feedback.calibrate(
            float(self.settings["gateway"]["step_up_fpr"]),
            float(self.settings["gateway"]["block_fpr"]),
        )
        with self.lock:
            previous = dict(self.thresholds)
            self.thresholds = updated
            self._bind_drift_reference()
        return {"previous": previous, "current": updated}

    def dashboard_html(self) -> str:
        body = [
            "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Fraud analyst feedback</title>",
            "<style>body{font-family:Segoe UI,sans-serif;margin:2rem;background:#fff;color:#111;}table{border-collapse:collapse;}td,th{border:1px solid #ccc;padding:0.4rem;color:#111;background:#fff;}</style>",
            "</head><body><h1>Component 02 analyst feedback</h1>",
            "<p>Decisions below come from the deterministic gateway. This page does not show account numbers, device strings, or network addresses.</p>",
            "<table><tr><th>Transaction</th><th>Score</th><th>Decision</th><th>Model</th></tr>",
        ]
        for row in list(self.recent):
            body.append(
                "<tr><td>{tx}</td><td>{score:.4f}</td><td>{decision}</td><td>{model}</td></tr>".format(
                    tx=escape(str(row["transaction_id"])),
                    score=float(row["fraud_score"]),
                    decision=escape(str(row["decision"])),
                    model=escape(str(row["model_version"])),
                )
            )
        body.append("</table></body></html>")
        return "".join(body)

    def _log(self, record: dict[str, Any]) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / "decisions.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def _reject_sensitive(payload: dict[str, Any]) -> None:
    keys = set(payload)
    keys.update((payload.get("features") or {}).keys())
    keys.update((payload.get("categories") or {}).keys())
    found = sorted(key for key in keys if str(key).lower() in BANNED_FIELDS)
    if found:
        raise ValueError(f"Remove sensitive fields: {found}. Send an account_key or device_key instead.")


def _raw_name(feature_name: str) -> str:
    if feature_name.startswith("freq__"):
        return feature_name[len("freq__") :]
    if feature_name.startswith("binary__"):
        return feature_name[len("binary__") :]
    return feature_name


def _linear_factors(model: Any, names: list[str], vector: np.ndarray) -> list[dict[str, Any]]:
    if model is None or not hasattr(model, "coef_"):
        return []
    coefficients = np.asarray(model.coef_).ravel()
    if len(coefficients) != len(vector):
        return []
    contribution = coefficients * vector
    order = np.argsort(np.abs(contribution))[::-1][:5]
    return [
        {"name": names[int(index)], "contribution": float(contribution[int(index)])}
        for index in order
    ]
