"""Shared pydantic models for every row that enters the pipeline.

One MarketBar type covers equities, ETFs and forex so downstream code never branches on
asset class for I/O. `asset_class` is carried as a field because it IS a modelling feature
(Phase 4 embeds it on the residual head) -- not because the schema differs.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class AssetClass(StrEnum):
    EQUITY = "equity"
    ETF = "etf"
    FOREX = "forex"


class MarketBar(BaseModel):
    """One OHLCV bar. Volume is 0.0 for forex -- yfinance does not report FX volume, so
    liquidity for FX comes from configs/universe.yaml::forex[].notional_adv_usd instead."""

    timestamp: datetime
    symbol: str
    asset_class: AssetClass
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)


class NewsItem(BaseModel):
    """A headline (news or social). `source` tags provenance: 'gdelt', 'newsapi', 'reddit'."""

    timestamp: datetime
    symbol: str
    source: str
    headline: str
    body: str | None = None
