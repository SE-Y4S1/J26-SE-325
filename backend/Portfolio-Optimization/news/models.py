"""Shared models for the news pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class Tier(StrEnum):
    """How directly an item bears on the stock. Aggregated separately, because "the central
    bank cut rates" and "John Keells missed earnings" are different signals and averaging
    them together would blur both."""

    COMPANY = "company"
    SECTOR = "sector"
    MACRO = "macro"


class SourceKind(StrEnum):
    """Source class, which sets an item's weight in aggregation."""

    OFFICIAL = "official"   # exchange filings and announcements
    NEWS = "news"           # editorial outlets
    SOCIAL = "social"       # posts by individuals


class RawItem(BaseModel):
    """One piece of content returned by a channel, before relevance and scoring."""

    channel: str
    kind: SourceKind
    title: str
    text: str | None = None
    url: str | None = None
    published_at: datetime
    outlet: str | None = None
    author: str | None = None
    # likes / shares / comments / views where the channel reports them.
    engagement: dict[str, float] = Field(default_factory=dict)


class KeywordProfile(BaseModel):
    """The search terms for one symbol, in tiers.

    `company` terms decide whether an item is about the company at all. `people_brands` are
    company-tier too, but are never used on their own as search queries -- a CEO's name or
    a brand alone pulls in too much unrelated content -- only as relevance evidence.
    `exclude` terms veto a match (the comb-not-COMB problem).
    """

    symbol: str
    exchange: str
    name: str
    sector: str | None = None
    company: list[str] = Field(default_factory=list)
    people_brands: list[str] = Field(default_factory=list)
    sector_terms: list[str] = Field(default_factory=list)
    macro_terms: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    # Sinhala / Tamil names, used for local-language search and matching.
    local_names: list[str] = Field(default_factory=list)
    linkedin_url: str | None = None
    # auto: generated, not yet reviewed. reviewed: a person has checked or edited it.
    source: str = "auto"
    generated_at: datetime | None = None
    # Which generator steps contributed: catalogue, template, llm.
    provenance: list[str] = Field(default_factory=list)

    def company_terms(self) -> list[str]:
        return [*self.company, *self.people_brands, *self.local_names]
