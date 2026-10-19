"""Per-symbol sentiment features from scored items.

PER TIER (company / sector / macro) -- kept separate because they are different signals:
  mean_sentiment    weighted mean polarity of scored items
  momentum          recent mean minus the earlier mean within the window (a shift, not a level)
  items / scored    how many relevant items, and how many could be scored
  attention_spike   how unusual the recent item count is versus the window's own baseline,
                    (recent - expected) / sqrt(expected + 1). Often a stronger signal than
                    polarity: a sudden burst of coverage precedes moves in either direction.

WEIGHT of one item = source weight (official > news > social)
                   x recency decay exp(-age / tau)
                   x scorer confidence
                   x engagement factor (social only, log-damped so one viral post cannot
                     outvote the rest)

CLEANING before any of that:
  * cross-channel duplicates (same story via Google News, GDELT and RSS) collapse to one,
    preferring official > news > social;
  * social spam is dropped: copy-pasted text from 3+ authors, posts stuffed with cashtags or
    links, and near-empty posts.

STORAGE: only these aggregates plus source links leave this module. Social posts contribute
numbers and a URL, never their text.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any

from news.models import SourceKind, Tier
from news.scoring import ScoredItem

DEFAULT_WEIGHTS = {"official": 1.0, "news": 0.8, "social": 0.4}
TIER_BLEND = {Tier.COMPANY: 0.6, Tier.SECTOR: 0.25, Tier.MACRO: 0.15}
MOMENTUM_RECENT_DAYS = 3
TOP_SOURCES = 8

_KIND_RANK = {SourceKind.OFFICIAL: 0, SourceKind.NEWS: 1, SourceKind.SOCIAL: 2}
_NON_WORD = re.compile(r"[^a-z0-9஀-௿඀-෿ ]+")


def _normalise(text: str) -> str:
    return " ".join(_NON_WORD.sub(" ", text.lower()).split())


def dedupe(items: list[ScoredItem]) -> list[ScoredItem]:
    """One entry per story: same URL, or same normalised title."""
    ordered = sorted(items, key=lambda s: (_KIND_RANK[s.relevant.item.kind], s.relevant.item.published_at))
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    out = []
    for scored in ordered:
        item = scored.relevant.item
        title_key = _normalise(item.title)[:120]
        url_key = (item.url or "").split("?")[0].rstrip("/")
        if (url_key and url_key in seen_urls) or (title_key and title_key in seen_titles):
            continue
        if url_key:
            seen_urls.add(url_key)
        if title_key:
            seen_titles.add(title_key)
        out.append(scored)
    return out


def drop_spam(items: list[ScoredItem]) -> tuple[list[ScoredItem], int]:
    """Remove low-quality social posts. Returns (kept, dropped_count)."""
    social_texts = Counter()
    authors_by_text: dict[str, set[str]] = defaultdict(set)
    for scored in items:
        item = scored.relevant.item
        if item.kind is SourceKind.SOCIAL:
            key = _normalise(item.text or item.title)
            social_texts[key] += 1
            authors_by_text[key].add(item.author or f"anon-{id(item)}")

    kept = []
    for scored in items:
        item = scored.relevant.item
        if item.kind is SourceKind.SOCIAL:
            body = item.text or item.title
            key = _normalise(body)
            if (len(key) < 15
                    or len(authors_by_text[key]) >= 3
                    or body.count("$") > 5
                    or len(re.findall(r"https?://", body)) > 3):
                continue
        kept.append(scored)
    return kept, len(items) - len(kept)


def _engagement_factor(item) -> float:  # noqa: ANN001 - RawItem
    if item.kind is not SourceKind.SOCIAL or not item.engagement:
        return 1.0
    e = item.engagement
    raw = e.get("likes", 0.0) + 2.0 * e.get("shares", 0.0) + e.get("comments", 0.0)
    return min(3.0, 1.0 + math.log1p(raw) / 5.0)


def item_weight(scored: ScoredItem, now: datetime, *, tau_days: float,
                source_weights: dict[str, float] | None = None) -> float:
    weights = source_weights or DEFAULT_WEIGHTS
    item = scored.relevant.item
    age_days = max(0.0, (now - item.published_at).total_seconds() / 86400)
    confidence = scored.confidence if scored.confidence is not None else 0.0
    return (weights.get(item.kind.value, 0.5) * math.exp(-age_days / tau_days)
            * confidence * _engagement_factor(item))


def _weighted_mean(pairs: list[tuple[float, float]]) -> float | None:
    total = sum(w for _, w in pairs)
    if total <= 0:
        return None
    return sum(v * w for v, w in pairs) / total


def aggregate(
    items: list[ScoredItem],
    *,
    now: datetime,
    window_days: int = 14,
    spike_window_days: int = 2,
    tau_days: float = 3.0,
    source_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """The sentiment block stored in the active set's metrics."""
    cutoff = now - timedelta(days=window_days)
    in_window = [s for s in items if s.relevant.item.published_at >= cutoff]
    cleaned, spam = drop_spam(dedupe(in_window))

    tiers: dict[str, Any] = {}
    for tier in Tier:
        tier_items = [s for s in cleaned if s.relevant.tier is tier]
        scored = [s for s in tier_items if s.polarity is not None]
        weighted = [(s.polarity, item_weight(s, now, tau_days=tau_days, source_weights=source_weights))
                    for s in scored]

        recent_cut = now - timedelta(days=MOMENTUM_RECENT_DAYS)
        recent = [(s.polarity, w) for s, (_, w) in zip(scored, weighted, strict=True)
                  if s.relevant.item.published_at >= recent_cut]
        earlier = [(s.polarity, w) for s, (_, w) in zip(scored, weighted, strict=True)
                   if s.relevant.item.published_at < recent_cut]
        recent_mean, earlier_mean = _weighted_mean(recent), _weighted_mean(earlier)

        spike_cut = now - timedelta(days=spike_window_days)
        recent_count = sum(1 for s in tier_items if s.relevant.item.published_at >= spike_cut)
        baseline_days = max(1, window_days - spike_window_days)
        baseline_rate = (len(tier_items) - recent_count) / baseline_days
        expected = baseline_rate * spike_window_days

        tiers[tier.value] = {
            "mean_sentiment": _round(_weighted_mean(weighted)),
            "momentum": _round(recent_mean - earlier_mean) if recent_mean is not None and earlier_mean is not None else None,
            "items": len(tier_items),
            "scored": len(scored),
            "attention_spike": round((recent_count - expected) / math.sqrt(expected + 1.0), 3),
            "recent_items": recent_count,
        }

    blend = [(tiers[t.value]["mean_sentiment"], w) for t, w in TIER_BLEND.items()
             if tiers[t.value]["mean_sentiment"] is not None]
    overall = _weighted_mean([(v, w) for v, w in blend]) if blend else None

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "window_days": window_days,
        "overall_sentiment": _round(overall),
        "tiers": tiers,
        "items_relevant": len(cleaned),
        "spam_dropped": spam,
        "languages": dict(Counter(s.language for s in cleaned)),
        "top_sources": top_sources(cleaned, now, tau_days=tau_days, source_weights=source_weights),
    }


def top_sources(items: list[ScoredItem], now: datetime, *, tau_days: float,
                source_weights: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """Most influential items, for the UI and for explanations. Social: link only."""
    def influence(s: ScoredItem) -> float:
        weight = item_weight(s, now, tau_days=tau_days, source_weights=source_weights)
        # Unscored official items (CSE filings) still matter as events.
        if s.polarity is None:
            return 0.05 if s.relevant.item.kind is SourceKind.OFFICIAL else 0.0
        return weight * (0.2 + abs(s.polarity))

    ranked = sorted(items, key=influence, reverse=True)[:TOP_SOURCES]
    out = []
    for s in ranked:
        item = s.relevant.item
        is_social = item.kind is SourceKind.SOCIAL
        out.append({
            "channel": item.channel,
            "kind": item.kind.value,
            "tier": s.relevant.tier.value,
            "title": None if is_social else item.title,
            "outlet": item.outlet,
            "url": item.url,
            "published_at": item.published_at.isoformat(timespec="seconds"),
            "polarity": _round(s.polarity),
            "language": s.language,
        })
    return out


def daily_rows(items: list[ScoredItem], symbol: str) -> list[dict[str, Any]]:
    """Per (date, tier) aggregates, appended to the symbol's sentiment history. This is what
    Phase 3's forecaster trains on -- a time series of numbers, no text."""
    buckets: dict[tuple[str, str], list[ScoredItem]] = defaultdict(list)
    for s in items:
        buckets[(s.relevant.item.published_at.date().isoformat(), s.relevant.tier.value)].append(s)
    rows = []
    for (day, tier), group in sorted(buckets.items()):
        scored = [(s.polarity, s.confidence or 0.0) for s in group if s.polarity is not None]
        rows.append({
            "symbol": symbol, "date": day, "tier": tier, "items": len(group), "scored": len(scored),
            "mean_sentiment": _round(_weighted_mean(scored)) if scored else None,
        })
    return rows


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)
