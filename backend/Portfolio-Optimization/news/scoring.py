"""Polarity per relevant item, routed by language.

  English   features.sentiment.score_headlines -- FinBERT when its weights are cached, else
            the local Ollama model. Shares that module's per-text score cache.
  Sinhala / Tamil
            Laya-multilingual (a typed "positive / neutral / negative for investors" question)
            when the `laya` package and its checkpoint are present. FinBERT is English-only,
            and a 3B general model scoring Sinhala would be guessing, so without Laya these
            items are left UNSCORED: they still count toward attention, never toward the
            sentiment mean.

Language is detected from the script (Sinhala U+0D80-0DFF, Tamil U+0B80-0BFF), which needs
no model and cannot be wrong about which script a string is in.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib.util import find_spec

from news.relevance import RelevantItem

logger = logging.getLogger(__name__)

# Scoring on CPU is the slowest step; cap how many uncached items one refresh scores.
MAX_SCORED_PER_REFRESH = 80

_SINHALA = re.compile(r"[඀-෿]")
_TAMIL = re.compile(r"[஀-௿]")


@dataclass(frozen=True)
class ScoredItem:
    relevant: RelevantItem
    language: str
    polarity: float | None          # None = not scored
    confidence: float | None
    backend: str | None


def detect_language(text: str) -> str:
    sinhala, tamil = len(_SINHALA.findall(text)), len(_TAMIL.findall(text))
    if sinhala or tamil:
        return "si" if sinhala >= tamil else "ta"
    return "en"


@lru_cache(maxsize=1)
def laya_router():  # noqa: ANN201 - laya.Router, imported lazily
    """A Laya Router when the package is importable and a checkpoint is cached, else None.

    Never triggers a download: on this project's development link the 842 MB checkpoint
    would take hours, so a missing checkpoint means "Laya unavailable", not "fetch it".
    """
    if find_spec("laya") is None:
        return None
    from pathlib import Path

    hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--convaiinnovations--laya"
    weights = [f for f in hub.rglob("*.safetensors")] if hub.exists() else []
    if not any(f.stat().st_size > 1_000_000 for f in weights):
        logger.info("laya installed but no checkpoint cached; semantic stages disabled")
        return None
    try:
        import laya

        return laya.Router()
    except Exception as exc:  # noqa: BLE001
        logger.warning("laya Router failed to start: %s", exc)
        return None


_LOCAL_SENTIMENT_QUESTION = {
    "sentiment": {
        "type": "choice",
        "instructions": "For an investor in the company or asset this text is about, is the news:",
        "criteria": {"positive": "good for the share price", "neutral": "no clear effect",
                     "negative": "bad for the share price"},
    }
}


def _choice_to_polarity(answer: dict) -> tuple[float, float] | None:
    """Laya `choice` answer -> (polarity, confidence), same contract as FinBERT's.

    Field names verified against laya 0.4.1 (agent.py): a choice answer carries `choice`,
    `probabilities` (label -> p) and `confidence`. The older fallbacks stay for safety; with no
    distribution at all the argmax alone is used at reduced confidence.
    """
    probabilities = answer.get("probabilities") or answer.get("distribution") or answer.get("scores")
    if isinstance(probabilities, dict) and probabilities:
        positive = float(probabilities.get("positive", 0.0))
        negative = float(probabilities.get("negative", 0.0))
        neutral = float(probabilities.get("neutral", 0.0))
        return positive - negative, 1.0 - neutral
    choice = answer.get("choice")
    if choice in ("positive", "negative", "neutral"):
        return {"positive": 0.5, "negative": -0.5, "neutral": 0.0}[choice], 0.5
    return None


def _score_local_language(items: list[RelevantItem]) -> list[tuple[float, float] | None]:
    router = laya_router()
    if router is None:
        return [None] * len(items)
    out: list[tuple[float, float] | None] = []
    for relevant in items:
        state = f"{relevant.item.title}\n{(relevant.item.text or '')[:1500]}"
        try:
            answer = router.predict(state, _LOCAL_SENTIMENT_QUESTION, model="multilingual")["answers"]["sentiment"]
            out.append(_choice_to_polarity(answer))
        except Exception as exc:  # noqa: BLE001
            logger.info("laya sentiment failed: %s", exc)
            out.append(None)
    return out


def score_items(items: list[RelevantItem]) -> tuple[list[ScoredItem], str | None]:
    """Score every item it can. Returns the items and the English backend used (or None)."""
    from data.schema import NewsItem
    from features import sentiment

    languages = [detect_language(f"{r.item.title} {r.item.text or ''}") for r in items]
    english = [i for i, lang in enumerate(languages) if lang == "en"]
    local = [i for i, lang in enumerate(languages) if lang != "en"]

    scores: dict[int, tuple[float, float, str]] = {}
    backend: str | None = None

    # Most recent first, so the cap drops the stalest items.
    english.sort(key=lambda i: items[i].item.published_at, reverse=True)
    to_score = english[:MAX_SCORED_PER_REFRESH]
    if to_score:
        try:
            backend = sentiment.resolve_backend("auto")
            news_items = [
                NewsItem(timestamp=items[i].item.published_at, symbol="-", source=items[i].item.channel,
                         headline=items[i].item.title, body=items[i].item.text)
                for i in to_score
            ]
            frame = sentiment.score_headlines(news_items, backend=backend)
            for i, (_, row) in zip(to_score, frame.iterrows(), strict=True):
                scores[i] = (float(row["polarity"]), float(row["confidence"]), backend)
        except RuntimeError as exc:
            logger.warning("no English sentiment backend: %s", exc)
            backend = None

    for i, result in zip(local, _score_local_language([items[i] for i in local]), strict=True):
        if result is not None:
            scores[i] = (result[0], result[1], "laya-multilingual")

    scored = []
    for i, relevant in enumerate(items):
        polarity, confidence, used = scores.get(i, (None, None, None))
        scored.append(ScoredItem(relevant, languages[i], polarity, confidence, used))
    return scored, backend
