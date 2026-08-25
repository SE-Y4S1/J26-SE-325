"""Sentiment scoring and daily aggregation.

TWO INTERCHANGEABLE BACKENDS
----------------------------
finbert -- ProsusAI/finbert, the proposal-named approach. A financial-domain BERT is a
           solved problem, and training one from scratch would be scope creep with a worse
           result. Needs ~420MB of weights from HuggingFace.
ollama  -- a locally installed instruction model (gemma4-e4b) prompted for the same
           three-way judgement. NO DOWNLOAD: it uses the model already on the machine.

Selection is `auto` by default: FinBERT when its weights are already cached, otherwise
Ollama, otherwise a clear error. Nothing here will ever start a multi-hundred-megabyte
download -- on a slow link that turns a pipeline run into an overnight job, and a silent
stall is worse than an explicit fallback.

Both backends emit the SAME (polarity, confidence) pair, so downstream aggregation, the
feature store and every test are backend-agnostic. Record which backend produced a given
run's features: they are not identical scorers, and mixing them within one experiment would
be a confound.

polarity   = P(positive) - P(negative), in [-1, 1]  -- neutral sits near zero while
             confidence is preserved: 0.9/0.05/0.05 reads far more positive than
             0.4/0.35/0.25.
confidence = 1 - P(neutral), i.e. how much of the mass is directional at all.

Scoring is the slowest step in the pipeline either way, so scores are cached per headline
hash and survive across runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from data.schema import NewsItem

logger = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"

# Local instruction model, already installed via Ollama. Used when FinBERT weights are absent.
OLLAMA_MODEL = "gemma4-e4b"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# FinBERT is a BERT-base: 512 wordpiece limit. Bodies longer than this are dropped rather
# than truncated -- a truncated article's sentiment is not reliably the whole article's.
MAX_BODY_CHARS = 1200

SCORE_CACHE_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "cache" / "finbert_scores.json"

# Trailing window for sentiment_momentum.
MOMENTUM_WINDOW_DAYS = 5

_model_cache: dict[str, object] = {}


def finbert_weights_cached() -> bool:
    """Whether FinBERT's weights are genuinely present locally.

    Checks for a real weight file, not just the directory: an interrupted download leaves
    config.json plus zero-byte `.incomplete` placeholders, which would otherwise look like a
    cache hit and send us into a stalling download.
    """
    cache = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{MODEL_NAME.replace('/', '--')}"
    if not cache.exists():
        return False
    return any(
        f.is_file() and not f.name.endswith(".incomplete") and f.stat().st_size > 1_000_000
        for f in cache.rglob("*")
    )


def ollama_available(model: str = OLLAMA_MODEL) -> bool:
    """Whether the Ollama daemon is up and the model is pulled."""
    import requests

    try:
        response = requests.get(OLLAMA_BASE_URL.replace("/v1", "/api/tags"), timeout=3)
        response.raise_for_status()
        names = {m.get("name", "").split(":")[0] for m in response.json().get("models", [])}
        return model.split(":")[0] in names
    except Exception:  # noqa: BLE001
        return False


def resolve_backend(preference: str = "auto") -> str:
    """Pick a scoring backend. Never triggers a download."""
    if preference == "finbert":
        if not finbert_weights_cached():
            raise RuntimeError(
                f"backend='finbert' requested but {MODEL_NAME} weights are not cached. "
                f"Download them first (`huggingface-cli download {MODEL_NAME}`) or use "
                "backend='ollama'."
            )
        return "finbert"

    if preference == "ollama":
        if not ollama_available():
            raise RuntimeError(
                f"backend='ollama' requested but {OLLAMA_MODEL} is not available. "
                "Start Ollama and check `ollama list`."
            )
        return "ollama"

    if preference != "auto":
        raise ValueError(f"unknown backend {preference!r}; expected auto|finbert|ollama")

    if finbert_weights_cached():
        return "finbert"
    if ollama_available():
        logger.info("FinBERT weights absent; scoring with local Ollama %s instead", OLLAMA_MODEL)
        return "ollama"
    raise RuntimeError(
        f"no sentiment backend available: {MODEL_NAME} weights are not cached and Ollama "
        f"({OLLAMA_MODEL}) is not reachable. Start Ollama, or pre-download FinBERT."
    )


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _load_score_cache() -> dict[str, list[float]]:
    if not SCORE_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(SCORE_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("unreadable FinBERT score cache (%s); starting fresh", exc)
        return {}


def _save_score_cache(cache: dict[str, list[float]]) -> None:
    SCORE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCORE_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")


def _get_pipeline():
    """Lazily construct the FinBERT pipeline, once per process.

    Imported inside the function so that merely importing this module does not pull in
    transformers/torch -- which matters for the FastAPI service and for fast test collection.
    """
    if "pipe" not in _model_cache:
        from transformers import pipeline

        logger.info("loading %s (first call downloads ~420MB)", MODEL_NAME)
        _model_cache["pipe"] = pipeline(
            "text-classification",
            model=MODEL_NAME,
            top_k=None,          # return all three class scores, not just the argmax
            truncation=True,
            max_length=512,
        )
    return _model_cache["pipe"]


def _to_polarity(scores: list[dict[str, float]]) -> tuple[float, float]:
    """Collapse FinBERT's 3-class output to (polarity, confidence).

    polarity   = P(positive) - P(negative), in [-1, 1]
    confidence = 1 - P(neutral), i.e. how much of the mass is directional at all
    """
    by_label = {entry["label"].lower(): float(entry["score"]) for entry in scores}
    positive = by_label.get("positive", 0.0)
    negative = by_label.get("negative", 0.0)
    neutral = by_label.get("neutral", 0.0)
    return positive - negative, 1.0 - neutral


OLLAMA_SENTIMENT_PROMPT = (
    "You are a financial sentiment classifier. Judge the sentiment of the headline for the "
    "company or asset it concerns, from an investor's point of view.\n"
    "Reply with ONLY a JSON object: "
    '{"positive": <float>, "negative": <float>, "neutral": <float>} '
    "-- three probabilities that sum to 1.0. No other text."
)


def _score_one_with_ollama(text: str, *, model: str, base_url: str) -> tuple[float, float]:
    """Score a single headline with the local instruction model.

    Asks for the same three-way probability split FinBERT produces, so the two backends are
    interchangeable downstream. `format='json'` constrains decoding, which matters: a small
    model asked for JSON in prose will otherwise wrap it in commentary.

    A malformed or failed response yields (0.0, 0.0) -- neutral with ZERO confidence. That is
    deliberately distinguishable from a confident neutral, and because aggregate_daily weights
    by confidence, an unparseable score contributes nothing rather than dragging the daily
    mean toward zero.
    """
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="ollama")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": OLLAMA_SENTIMENT_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.0,          # deterministic: re-running must not move the features
            response_format={"type": "json_object"},
        )
        payload = json.loads(response.choices[0].message.content or "{}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ollama sentiment call failed (%s); scoring as no-confidence neutral", exc)
        return 0.0, 0.0

    try:
        positive = float(payload.get("positive", 0.0))
        negative = float(payload.get("negative", 0.0))
        neutral = float(payload.get("neutral", 0.0))
    except (TypeError, ValueError):
        return 0.0, 0.0

    total = positive + negative + neutral
    if total <= 0:
        return 0.0, 0.0
    # Renormalize: instruction models routinely return three numbers that sum to 0.95 or 1.1.
    positive, negative, neutral = positive / total, negative / total, neutral / total

    return float(np.clip(positive - negative, -1.0, 1.0)), float(np.clip(1.0 - neutral, 0.0, 1.0))


def _prepare_text(item: NewsItem) -> str:
    """Headline, plus the body when it is short enough to be safely encoded."""
    if item.body and len(item.body) <= MAX_BODY_CHARS:
        return f"{item.headline}. {item.body}"
    return item.headline


def score_headlines(
    items: list[NewsItem],
    *,
    batch_size: int = 32,
    backend: str = "auto",
) -> pd.DataFrame:
    """Polarity per NewsItem. Returns (timestamp, symbol, source, polarity, confidence).

    `backend` is auto|finbert|ollama. Auto prefers FinBERT when its weights are already
    cached and falls back to the local Ollama model otherwise -- it never downloads.
    """
    if not items:
        return pd.DataFrame(columns=["timestamp", "symbol", "source", "polarity", "confidence"])

    cache = _load_score_cache()
    texts = [_prepare_text(item) for item in items]
    hashes = [_text_hash(text) for text in texts]

    pending_idx = [i for i, h in enumerate(hashes) if h not in cache]
    if pending_idx:
        chosen = resolve_backend(backend)
        if chosen == "ollama":
            _score_pending_with_ollama(texts, hashes, pending_idx, cache)
            _save_score_cache(cache)
            return _assemble(items, hashes, cache)
        pipe = _get_pipeline()
        pending_texts = [texts[i] for i in pending_idx]
        logger.info("scoring %d new headlines with FinBERT (%d cached)",
                    len(pending_texts), len(items) - len(pending_texts))

        for start in range(0, len(pending_texts), batch_size):
            batch = pending_texts[start : start + batch_size]
            outputs = pipe(batch)
            for offset, scores in enumerate(outputs):
                idx = pending_idx[start + offset]
                cache[hashes[idx]] = list(_to_polarity(scores))
        _save_score_cache(cache)

    rows = []
    for item, digest in zip(items, hashes, strict=True):
        polarity, confidence = cache[digest]
        rows.append(
            {
                "timestamp": item.timestamp,
                "symbol": item.symbol,
                "source": item.source,
                "polarity": polarity,
                "confidence": confidence,
            }
        )
    return pd.DataFrame(rows)


def _score_pending_with_ollama(
    texts: list[str], hashes: list[str], pending_idx: list[int], cache: dict[str, list[float]]
) -> None:
    """Score uncached headlines one at a time.

    Ollama has no batch endpoint, so this is serial and slow -- which is exactly why the
    on-disk cache matters: the cost is paid once per unique headline, ever.
    """
    logger.info("scoring %d headlines with Ollama %s (%d cached)",
                len(pending_idx), OLLAMA_MODEL, len(texts) - len(pending_idx))

    for count, idx in enumerate(pending_idx, start=1):
        cache[hashes[idx]] = list(
            _score_one_with_ollama(texts[idx], model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        )
        if count % 25 == 0:
            _save_score_cache(cache)      # checkpoint: a long run must survive interruption
            logger.info("  %d/%d scored", count, len(pending_idx))


def _assemble(
    items: list[NewsItem], hashes: list[str], cache: dict[str, list[float]]
) -> pd.DataFrame:
    """Build the scored frame from cached (polarity, confidence) pairs."""
    return pd.DataFrame(
        [
            {
                "timestamp": item.timestamp,
                "symbol": item.symbol,
                "source": item.source,
                "polarity": cache[digest][0],
                "confidence": cache[digest][1],
            }
            for item, digest in zip(items, hashes, strict=True)
        ]
    )


def aggregate_daily(scored: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-headline scores to one row per (symbol, date).

    Produces:
      mean_sentiment     -- confidence-weighted mean polarity that day. Weighting by
                            confidence stops a pile of hedged, near-neutral headlines from
                            diluting one decisive one.
      sentiment_volume   -- headline count, a proxy for attention/news intensity
      sentiment_momentum -- change in mean_sentiment vs its trailing 5-day mean, so the
                            model sees sentiment *shifts*, not just levels
    """
    if scored.empty:
        return pd.DataFrame(columns=["symbol", "date", "mean_sentiment", "sentiment_volume", "sentiment_momentum"])

    frame = scored.copy()
    frame["date"] = pd.to_datetime(frame["timestamp"]).dt.normalize()

    def _weighted_mean(group: pd.DataFrame) -> float:
        weights = group["confidence"].to_numpy(dtype=float)
        values = group["polarity"].to_numpy(dtype=float)
        total = weights.sum()
        # All-neutral day: fall back to the unweighted mean rather than dividing by zero.
        return float(np.average(values, weights=weights)) if total > 0 else float(values.mean())

    grouped = frame.groupby(["symbol", "date"], sort=True)
    daily = grouped.apply(_weighted_mean, include_groups=False).rename("mean_sentiment").reset_index()
    daily["sentiment_volume"] = grouped.size().to_numpy()

    daily = daily.sort_values(["symbol", "date"]).reset_index(drop=True)
    daily["sentiment_momentum"] = daily.groupby("symbol", sort=False)["mean_sentiment"].transform(
        lambda s: s - s.rolling(MOMENTUM_WINDOW_DAYS, min_periods=1).mean().shift(1)
    ).fillna(0.0)

    return daily


def build_sentiment_features(
    symbol: str,
    start: date,
    end: date,
    *,
    query: str | None = None,
    include_social: bool = False,
) -> pd.DataFrame:
    """End-to-end: fetch -> score -> aggregate for one symbol."""
    from data.ingestion import fetch_news, fetch_social

    items = fetch_news(symbol, start, end, query=query)
    if include_social:
        items = items + fetch_social(symbol, start, end)

    if not items:
        logger.info("no headlines for %s (%s..%s)", symbol, start, end)
        return pd.DataFrame(columns=["symbol", "date", "mean_sentiment", "sentiment_volume", "sentiment_momentum"])

    return aggregate_daily(score_headlines(items))
