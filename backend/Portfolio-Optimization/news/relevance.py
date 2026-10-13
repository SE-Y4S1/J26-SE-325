"""Is an item actually about this stock -- and at which tier?

A keyword hit is not proof. "COMB" is a hair comb, "DIAL" is a phone dial, and a Google
News search for "Commercial Bank of Ceylon" happily returns articles that mention it once in
a list of 20 banks. Two stages:

  1. TERM MATCH (always). Word-boundary matching of the profile's tiers against title + text.
     Short all-caps tickers (<= 5 chars) must appear in capitals ("JKH", never "jkh"), since
     lower-case matches of short tokens are almost always something else. Any exclusion term
     vetoes the item. The item takes the most direct tier it matches: company > sector >
     macro. An item matching nothing is dropped, whatever query found it.

  2. SEMANTIC CHECK (when a Laya checkpoint is available locally). A zero-shot yes/no
     question -- "is this about <company> or something that affects its share price?" --
     filters what term matching lets through. When Laya is not installed this stage is
     skipped and recorded as such (`relevance_method: keyword`), never faked.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache

from news.models import KeywordProfile, RawItem, Tier

logger = logging.getLogger(__name__)

SEMANTIC_THRESHOLD = 0.5


@dataclass(frozen=True)
class RelevantItem:
    item: RawItem
    tier: Tier
    matched: tuple[str, ...]
    relevance: float | None = None      # Laya probability, when stage 2 ran


@lru_cache(maxsize=4096)
def _pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term.strip())
    is_short_ticker = term.isupper() and len(term.replace("$", "")) <= 5 and " " not in term
    flags = 0 if is_short_ticker else re.IGNORECASE
    # \b does not work next to '$' or non-Latin scripts; use explicit non-word lookarounds.
    if re.search(r"[஀-௿඀-෿]", term):
        return re.compile(escaped)
    return re.compile(rf"(?<![\w$]){escaped}(?!\w)", flags)


def _matches(text: str, terms: list[str]) -> tuple[str, ...]:
    return tuple(term for term in terms if term and _pattern(term).search(text))


def _context(exchange: str) -> tuple[list[str], set[str]]:
    from news.keywords import load_config

    spec = (load_config().get("context_terms") or {}).get(exchange) or {}
    return list(spec.get("terms") or []), {o.lower() for o in spec.get("outlets") or []}


def _in_context(item: RawItem, text: str, exchange: str) -> bool:
    """Whether a sector/macro item is about the right economy. Exchanges without configured
    context terms (crypto, US) always pass."""
    terms, outlets = _context(exchange)
    if not terms and not outlets:
        return True
    if (item.outlet or "").lower() in outlets:
        return True
    return bool(_matches(text, terms))


def classify(item: RawItem, profile: KeywordProfile) -> RelevantItem | None:
    """Stage 1. None when the item is irrelevant or vetoed."""
    text = f"{item.title}\n{item.text or ''}"
    if _matches(text, profile.exclude):
        return None
    hit = _matches(text, profile.company_terms())
    if hit:
        return RelevantItem(item=item, tier=Tier.COMPANY, matched=hit)
    for tier, terms in ((Tier.SECTOR, profile.sector_terms), (Tier.MACRO, profile.macro_terms)):
        hit = _matches(text, terms)
        if hit and _in_context(item, text, profile.exchange):
            return RelevantItem(item=item, tier=tier, matched=hit)
    return None


def filter_relevant(items: list[RawItem], profile: KeywordProfile) -> tuple[list[RelevantItem], str]:
    """Both stages. Returns the kept items and the method used ('keyword' or 'keyword+laya')."""
    kept = [r for r in (classify(item, profile) for item in items) if r is not None]
    scored = semantic_relevance(kept, profile)
    if scored is None:
        return kept, "keyword"
    return [r for r in scored if (r.relevance or 0.0) >= SEMANTIC_THRESHOLD], "keyword+laya"


def semantic_relevance(items: list[RelevantItem], profile: KeywordProfile) -> list[RelevantItem] | None:
    """Laya zero-shot relevance. None when Laya is unavailable."""
    from news.scoring import laya_router

    router = laya_router()
    if router is None or not items:
        return None

    question = {
        "relevant": {
            "type": "noul",
            "instructions": (
                f"Is this text about {profile.name} ({profile.symbol}), or about something "
                "that would plausibly move its share or token price (its sector, its main "
                "markets, or the economy it operates in)?"
            ),
        }
    }
    out = []
    for relevant in items:
        state = f"{relevant.item.title}\n{(relevant.item.text or '')[:1500]}"
        try:
            result = router.predict(state, question)
            probability = float(result["answers"]["relevant"]["noul"])
        except Exception as exc:  # noqa: BLE001 - fall back to stage 1 for this item
            logger.info("laya relevance failed (%s); keeping keyword verdict", exc)
            probability = 1.0
        out.append(RelevantItem(relevant.item, relevant.tier, relevant.matched, probability))
    return out
