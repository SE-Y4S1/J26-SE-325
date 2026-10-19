"""The channel interface and the query-building rules every channel shares."""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from news.models import KeywordProfile, RawItem, SourceKind, Tier

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) j26-se-325-component1/0.1"


def utcnow() -> datetime:
    """Naive UTC 'now'. Every timestamp in the news pipeline is naive UTC, so they compare."""
    from datetime import timezone

    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass(frozen=True)
class ChannelStatus:
    name: str
    kind: str
    available: bool
    reason: str | None = None
    requires_login: bool = False


@dataclass(frozen=True)
class Query:
    """One search a channel should run, and the tier the results are presumed to belong to.
    Relevance filtering re-checks the tier against the item text later."""

    text: str
    tier: Tier
    terms: tuple[str, ...]


class Channel(ABC):
    name: str = "base"
    kind: SourceKind = SourceKind.NEWS
    requires_login: bool = False
    # Exchanges this channel is useful for; empty means all.
    regions: tuple[str, ...] = ()
    # Whether the channel searches by keyword (True) or returns a whole feed to be filtered.
    searches: bool = True

    @abstractmethod
    def status(self) -> ChannelStatus: ...

    @abstractmethod
    def fetch(self, profile: KeywordProfile, since: datetime, *, limit: int = 40) -> list[RawItem]: ...

    def applies_to(self, profile: KeywordProfile) -> bool:
        return not self.regions or profile.exchange in self.regions

    def _status(self, available: bool, reason: str | None = None) -> ChannelStatus:
        return ChannelStatus(self.name, self.kind.value, available, reason, self.requires_login)


# --------------------------------------------------------------------------------------
# Query building
# --------------------------------------------------------------------------------------

def search_terms(profile: KeywordProfile, *, max_terms: int = 3) -> list[str]:
    """Company terms suitable as standalone queries: names, not people/brands, not cashtags
    and not ambiguous bare tickers (those are relevance evidence only)."""
    terms = [t for t in profile.company if not t.startswith("$")]
    # Longest names first: "John Keells Holdings" is more precise than "JKH".
    terms.sort(key=lambda t: (-len(t.split()), -len(t)))
    return terms[:max_terms]


def or_query(terms: list[str]) -> str:
    """'"John Keells" OR "JKH"' -- quoted phrases joined with OR."""
    return " OR ".join(f'"{t}"' for t in terms if t)


def queries_for(profile: KeywordProfile, *, include_context: bool = True) -> list[Query]:
    """Company query always; sector and macro queries when the channel wants context."""
    queries = []
    company = search_terms(profile)
    if company:
        queries.append(Query(or_query(company), Tier.COMPANY, tuple(company)))
    if profile.local_names:
        queries.append(Query(or_query(profile.local_names[:2]), Tier.COMPANY, tuple(profile.local_names[:2])))
    if include_context:
        if profile.sector_terms:
            terms = profile.sector_terms[:3]
            queries.append(Query(or_query(terms), Tier.SECTOR, tuple(terms)))
        if profile.macro_terms:
            terms = profile.macro_terms[:3]
            queries.append(Query(or_query(terms), Tier.MACRO, tuple(terms)))
    return queries


# --------------------------------------------------------------------------------------
# Small shared helpers
# --------------------------------------------------------------------------------------

class TTLCache:
    """In-process cache. Sector and macro queries repeat across every symbol in a sector;
    without this, preparing ten bank stocks would run the same banking query ten times."""

    def __init__(self, ttl_seconds: float) -> None:
        self.ttl = ttl_seconds
        self._data: dict[Any, tuple[float, Any]] = {}

    def get(self, key: Any) -> Any:
        hit = self._data.get(key)
        if hit and time.time() - hit[0] < self.ttl:
            return hit[1]
        return None

    def put(self, key: Any, value: Any) -> None:
        self._data[key] = (time.time(), value)

    def clear(self) -> None:
        self._data.clear()


_TAGS = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"\s+")


def strip_html(text: str | None) -> str | None:
    if not text:
        return None
    import html

    cleaned = _SPACES.sub(" ", html.unescape(_TAGS.sub(" ", text))).strip()
    return cleaned or None
