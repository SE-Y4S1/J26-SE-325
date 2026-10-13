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
    # Colombo Stock Exchange listings. A separate class rather than EQUITY because CSE names
    # trade far more thinly than US large caps, which is exactly the regime the liquidity-
    # aware optimizer exists for -- the residual head should be able to tell them apart.
    CSE_EQUITY = "cse_equity"
    # Binance spot pairs. 24/7 trading and a very different volatility regime.
    CRYPTO = "crypto"


class MarketBar(BaseModel):
    """One OHLCV bar. Volume is 0.0 for forex -- yfinance does not report FX volume, so
    liquidity for FX comes from configs/universe.yaml::forex[].notional_adv_usd instead.

    `open` is optional because the CSE chart endpoint does not publish an opening price.
    Leaving it None is deliberate: back-filling it with the previous close would invent a
    gap-free open that never traded, and nothing downstream (MACD/RSI/MFI/ATR) needs it.
    """

    timestamp: datetime
    symbol: str
    asset_class: AssetClass
    open: float | None = Field(default=None, gt=0)
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
