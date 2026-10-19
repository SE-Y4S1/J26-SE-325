"""Market-data and active-set endpoints.

  GET  /market/search    one search box across CSE, Binance and Yahoo
  POST /market/quote     live price, liquidity and volatility for one instrument, in its
                         native currency and in the base currency
  POST /analysis/track   register a portfolio's symbols; preparation runs in the background
  GET  /analysis/status  where each symbol's preparation stands
  POST /analysis/refresh re-run preparation (news goes stale within hours)
  GET  /analysis/channels               which news/social channels can run, and why not
  GET  /analysis/keywords/{symbol}      a held symbol's keyword profile
  PUT  /analysis/keywords/{symbol}      correct it (marks it reviewed; never regenerated over)
  GET  /analysis/sentiment/{symbol}     per-tier sentiment, attention and top sources
  GET  /analysis/fundamentals/{symbol}  statements, ratios, F/Z scores, red flags (or tokenomics)
  GET  /analysis/technical/{symbol}     trend, momentum, volume, range, relative strength
  GET  /analysis/forecast/{symbol}      forecast bands per horizon (hybrid, or labelled empirical)

Search and quote never register anything: they are called while a user is still typing,
and tracking every keystroke's match would make the active set meaningless.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from service.auth import CallerIdentity, require_user
from service.market_contracts import (
    AnalysisForecastResponse,
    ForecastBand,
    ChannelsResponse,
    ChannelStatusOut,
    FundamentalsOut,
    FundamentalsResponse,
    InstrumentOut,
    KeywordProfileOut,
    KeywordProfileUpdate,
    QuoteRequest,
    QuoteResponse,
    RefreshRequest,
    SearchResponse,
    SentimentOut,
    SentimentResponse,
    StatusResponse,
    SymbolStatus,
    TechnicalOut,
    TechnicalResponse,
    TrackRequest,
    TrackResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _instrument_out(instrument) -> InstrumentOut:  # noqa: ANN001 - data.instruments.Instrument
    return InstrumentOut(
        symbol=instrument.symbol, name=instrument.name, exchange=instrument.exchange.value,
        asset_class=instrument.asset_class.value, currency=instrument.currency,
        last_price=instrument.last_price,
    )


@router.get("/market/search", response_model=SearchResponse)
def market_search(
    q: str = Query(min_length=1, max_length=64),
    limit: int = Query(default=20, ge=1, le=50),
    caller: CallerIdentity | None = Depends(require_user),
) -> SearchResponse:
    from data.instruments import search

    return SearchResponse(query=q, results=[_instrument_out(i) for i in search(q, limit=limit)])


@router.post("/market/quote", response_model=QuoteResponse)
def market_quote(
    request: QuoteRequest,
    caller: CallerIdentity | None = Depends(require_user),
) -> QuoteResponse:
    from data.fx import BASE_CURRENCY, get_rate
    from data.history import get_history
    from data.http import UpstreamError
    from data.instruments import resolve
    from features.liquidity import compute_liquidity_metrics

    try:
        instrument = resolve(request.symbol, request.exchange)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except UpstreamError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None

    try:
        history = get_history(instrument)
        fx = get_rate(instrument.currency, BASE_CURRENCY)
    except UpstreamError as exc:
        # 503, not 500: the service is fine, an exchange is not. The frontend shows this as
        # "exchange unreachable, enter the price manually".
        raise HTTPException(status_code=503, detail=str(exc)) from None

    metrics = compute_liquidity_metrics(history.bars)
    value_native = request.quantity * metrics.last_close if request.quantity is not None else None

    return QuoteResponse(
        instrument=_instrument_out(instrument),
        price=metrics.last_close,
        as_of=metrics.as_of,
        currency=instrument.currency,
        base_currency=BASE_CURRENCY,
        fx_rate_to_base=fx.rate,
        fx_as_of=fx.as_of,
        avg_daily_volume=metrics.adv_value,
        avg_daily_volume_base=metrics.adv_value * fx.rate,
        daily_volatility=metrics.daily_volatility,
        volatility_pct=metrics.volatility_pct,
        zero_volume_share=metrics.zero_volume_share,
        sessions_per_week=metrics.sessions_per_week,
        history_days=metrics.history_days,
        estimated=metrics.estimated,
        stale=history.stale,
        value_native=value_native,
        value_base=value_native * fx.rate if value_native is not None else None,
    )


@router.post("/analysis/track", response_model=TrackResponse)
def analysis_track(
    request: TrackRequest,
    background: BackgroundTasks,
    caller: CallerIdentity | None = Depends(require_user),
) -> TrackResponse:
    from data import active_set
    from data.http import UpstreamError
    from data.instruments import resolve

    instruments = []
    rejected: dict[str, str] = {}
    for symbol in dict.fromkeys(s.strip().upper() for s in request.symbols if s.strip()):
        existing = active_set.get_instrument(symbol)
        if existing is not None:
            instruments.append(existing)
            continue
        try:
            instruments.append(resolve(symbol))
        except (LookupError, UpstreamError) as exc:
            rejected[symbol] = str(exc)

    to_prepare = active_set.track(instruments)
    if to_prepare:
        # Runs after the response is sent; the UI polls /analysis/status.
        background.add_task(active_set.prepare_many, to_prepare)

    rows = active_set.status([i.symbol for i in instruments]) if instruments else []
    return TrackResponse(tracked=[SymbolStatus(**row) for row in rows], rejected=rejected)


@router.get("/analysis/status", response_model=StatusResponse)
def analysis_status(
    symbols: str | None = Query(default=None, description="Comma-separated; omit for all"),
    caller: CallerIdentity | None = Depends(require_user),
) -> StatusResponse:
    from data import active_set

    wanted = [s.strip().upper() for s in symbols.split(",") if s.strip()] if symbols else None
    return StatusResponse(symbols=[SymbolStatus(**row) for row in active_set.status(wanted)])


@router.post("/analysis/refresh", response_model=StatusResponse)
def analysis_refresh(
    request: RefreshRequest,
    background: BackgroundTasks,
    caller: CallerIdentity | None = Depends(require_user),
) -> StatusResponse:
    from data import active_set

    rows = active_set.status([s.strip().upper() for s in request.symbols] if request.symbols else None)
    background.add_task(active_set.prepare_many, [row["symbol"] for row in rows])
    return StatusResponse(symbols=[SymbolStatus(**row) for row in rows])


@router.get("/analysis/channels", response_model=ChannelsResponse)
def analysis_channels(caller: CallerIdentity | None = Depends(require_user)) -> ChannelsResponse:
    from news.channels import channel_statuses

    return ChannelsResponse(channels=[
        ChannelStatusOut(name=s.name, kind=s.kind, available=s.available, reason=s.reason,
                         requires_login=s.requires_login)
        for s in channel_statuses()
    ])


@router.get("/analysis/keywords/{symbol}", response_model=KeywordProfileOut)
def get_keywords(symbol: str, caller: CallerIdentity | None = Depends(require_user)) -> KeywordProfileOut:
    from news.keywords import load_profile

    profile = load_profile(symbol.strip().upper())
    if profile is None:
        raise HTTPException(status_code=404, detail=f"no keyword profile for {symbol}; track it first")
    return KeywordProfileOut(**profile.model_dump())


@router.put("/analysis/keywords/{symbol}", response_model=KeywordProfileOut)
def put_keywords(
    symbol: str,
    update: KeywordProfileUpdate,
    caller: CallerIdentity | None = Depends(require_user),
) -> KeywordProfileOut:
    from news.keywords import update_profile

    changes = update.model_dump(exclude_unset=True)
    if not changes:
        raise HTTPException(status_code=422, detail="no fields to update")
    try:
        profile = update_profile(symbol.strip().upper(), changes)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    return KeywordProfileOut(**profile.model_dump())


@router.get("/analysis/sentiment/{symbol}", response_model=SentimentResponse)
def get_sentiment(symbol: str, caller: CallerIdentity | None = Depends(require_user)) -> SentimentResponse:
    from data import active_set

    symbol = symbol.strip().upper()
    rows = active_set.status([symbol])
    if not rows:
        raise HTTPException(status_code=404, detail=f"{symbol} is not tracked")
    row = rows[0]
    block = (row.get("metrics") or {}).get("sentiment")
    if not block:
        detail = "preparing" if row["status"] == active_set.PREPARING else (row.get("error") or "no sentiment yet")
        return SentimentResponse(symbol=symbol, status=row["status"], detail=detail)
    return SentimentResponse(symbol=symbol, status=row["status"], sentiment=SentimentOut(**block))


def _step_block(symbol: str, step: str) -> tuple[dict | None, dict]:
    """(metrics block for a pipeline step, registry row). 404 when untracked."""
    from data import active_set

    rows = active_set.status([symbol])
    if not rows:
        raise HTTPException(status_code=404, detail=f"{symbol} is not tracked")
    row = rows[0]
    return (row.get("metrics") or {}).get(step), row


def _missing_detail(row: dict, step: str) -> str:
    from data import active_set

    if row["status"] == active_set.PREPARING:
        return "preparing"
    errors = [e for e in (row.get("error") or "").split("; ") if e.startswith(f"{step}:")]
    return errors[0] if errors else f"no {step} yet"


@router.get("/analysis/fundamentals/{symbol}", response_model=FundamentalsResponse)
def get_fundamentals(symbol: str, caller: CallerIdentity | None = Depends(require_user)) -> FundamentalsResponse:
    symbol = symbol.strip().upper()
    block, row = _step_block(symbol, "fundamentals")
    if not block:
        return FundamentalsResponse(symbol=symbol, status=row["status"], detail=_missing_detail(row, "fundamentals"))
    return FundamentalsResponse(symbol=symbol, status=row["status"], fundamentals=FundamentalsOut(**block))


@router.get("/analysis/technical/{symbol}", response_model=TechnicalResponse)
def get_technical(symbol: str, caller: CallerIdentity | None = Depends(require_user)) -> TechnicalResponse:
    symbol = symbol.strip().upper()
    block, row = _step_block(symbol, "technical")
    if not block:
        return TechnicalResponse(symbol=symbol, status=row["status"], detail=_missing_detail(row, "technical"))
    return TechnicalResponse(symbol=symbol, status=row["status"], technical=TechnicalOut(**block))


@router.get("/analysis/forecast/{symbol}", response_model=AnalysisForecastResponse)
def get_analysis_forecast(symbol: str, caller: CallerIdentity | None = Depends(require_user)
                          ) -> AnalysisForecastResponse:
    symbol = symbol.strip().upper()
    block, row = _step_block(symbol, "forecast")
    if not block:
        return AnalysisForecastResponse(symbol=symbol, status=row["status"], detail=_missing_detail(row, "forecast"))
    bands = [ForecastBand(**band) for _, band in sorted(block.items(), key=lambda kv: kv[1].get("horizon", 0))]
    return AnalysisForecastResponse(symbol=symbol, status=row["status"], bands=bands)
