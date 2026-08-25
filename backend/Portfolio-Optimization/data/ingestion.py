"""Raw data acquisition: OHLCV, news, social.

Every fetch goes through data.cache.cached_fetch so repeated runs do not re-hit
rate-limited APIs.

Source decisions (see README "Deviations from the build brief"):
  * Forex uses yfinance FX tickers ('EURUSD=X'), not Alpha Vantage. Free, unlimited, deep
    history, and it reuses the equity code path -- one less provider to justify.
  * News is a GDELT + NewsAPI hybrid: GDELT supplies multi-year historical backfill (the
    NewsAPI free tier caps at ~30 days, which cannot support a walk-forward backtest),
    NewsAPI supplies the recent/live window for the running service.
  * Reddit is implemented but disabled in configs/universe.yaml.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests

from data.cache import cached_fetch
from data.schema import AssetClass, MarketBar, NewsItem

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["timestamp", "symbol", "asset_class", "open", "high", "low", "close", "volume"]
NEWS_COLUMNS = ["timestamp", "symbol", "source", "headline", "body"]

# NewsAPI's free "everything" endpoint refuses `from` dates older than ~30 days. Anything
# beyond this horizon must come from GDELT.
NEWSAPI_HORIZON_DAYS = 28

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
NEWSAPI_EVERYTHING = "https://newsapi.org/v2/everything"

REQUEST_TIMEOUT = 30

# yfinance rate-limits aggressively and signals it the same way as a delisting: an empty
# frame. Retrying with backoff is the only way to tell them apart, and without it a universe
# resolution silently loses whichever symbols happened to be throttled -- observed dropping
# SPY, XOM, BA, TLT and XLE from a single run.
OHLCV_MAX_ATTEMPTS = 3
OHLCV_BACKOFF_SECONDS = 5.0


# --------------------------------------------------------------------------------------
# OHLCV
# --------------------------------------------------------------------------------------

def _fetch_ohlcv_frame(
    symbol: str,
    asset_class: AssetClass,
    start: date,
    end: date,
    *,
    max_attempts: int = OHLCV_MAX_ATTEMPTS,
) -> pd.DataFrame:
    """Raw yfinance pull, normalized to OHLCV_COLUMNS, with backoff on empty responses."""
    import time

    import yfinance as yf

    raw = None
    for attempt in range(1, max_attempts + 1):
        # yfinance treats `end` as exclusive; the public API here is inclusive, which is what
        # a caller reasoning about trading days expects.
        ticker = yf.Ticker(symbol)
        raw = ticker.history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=True,
            raise_errors=False,
        )
        if raw is not None and not raw.empty:
            break

        if attempt < max_attempts:
            delay = OHLCV_BACKOFF_SECONDS * attempt      # linear backoff; yfinance recovers fast
            logger.info(
                "empty OHLCV for %s (attempt %d/%d); retrying in %.0fs",
                symbol, attempt, max_attempts, delay,
            )
            time.sleep(delay)

    if raw is None or raw.empty:
        logger.warning(
            "no OHLCV for %s (%s..%s) after %d attempts -- rate-limited or genuinely delisted",
            symbol, start, end, max_attempts,
        )
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    frame = raw.reset_index()
    # yfinance names the index 'Date' for daily bars and 'Datetime' for intraday.
    time_col = "Date" if "Date" in frame.columns else "Datetime"
    frame = frame.rename(
        columns={
            time_col: "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    frame["symbol"] = symbol
    frame["asset_class"] = asset_class.value

    # Strip tz so all three asset classes share one comparable calendar. FX trades ~24h and
    # equities do not; normalizing to naive dates keeps the join key consistent.
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_localize(None).dt.normalize()

    frame = frame[OHLCV_COLUMNS]
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    # yfinance reports FX volume as 0 -- expected, not a data error. See universe.yaml.
    frame["volume"] = frame["volume"].fillna(0.0)
    return frame.sort_values("timestamp").reset_index(drop=True)


def fetch_ohlcv(
    symbol: str,
    asset_class: AssetClass,
    start: date,
    end: date,
    *,
    force_refresh: bool = False,
) -> list[MarketBar]:
    """Daily bars for equities, ETFs and forex via yfinance.

    Forex bars come back with volume == 0.0; that is expected, not a data error.
    `end` is treated as inclusive, unlike yfinance's native half-open interval.
    """
    frame = cached_fetch(
        "ohlcv",
        symbol,
        start,
        end,
        lambda: _fetch_ohlcv_frame(symbol, asset_class, start, end),
        force_refresh=force_refresh,
        # A listed instrument always has bars, so an empty frame means the fetch failed --
        # yfinance cannot distinguish a rate-limit from a delisting. Caching it would drop the
        # symbol from every future run. See data/cache.py::cached_fetch.
        cache_empty=False,
    )
    return frame_to_bars(frame)


def fetch_forex(pair: str, start: date, end: date, *, force_refresh: bool = False) -> list[MarketBar]:
    """Thin alias over fetch_ohlcv for FX pairs, kept because the brief names it explicitly."""
    return fetch_ohlcv(pair, AssetClass.FOREX, start, end, force_refresh=force_refresh)


def frame_to_bars(frame: pd.DataFrame) -> list[MarketBar]:
    """Validate a normalized OHLCV frame into MarketBar models."""
    if frame.empty:
        return []
    return [MarketBar(**row) for row in frame.to_dict(orient="records")]


def bars_to_frame(bars: list[MarketBar]) -> pd.DataFrame:
    """Inverse of frame_to_bars, for the feature layer."""
    if not bars:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    return pd.DataFrame([bar.model_dump() for bar in bars])[OHLCV_COLUMNS]


# --------------------------------------------------------------------------------------
# News
# --------------------------------------------------------------------------------------

def _fetch_gdelt(query: str, symbol: str, start: date, end: date) -> pd.DataFrame:
    """GDELT DOC 2.0 article search. Free, no key, multi-year history.

    GDELT indexes prose, not tickers, so `query` is the company name from universe.yaml --
    searching for 'AAPL' would miss almost every article about Apple.
    """
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": 250,
        "startdatetime": f"{start.strftime('%Y%m%d')}000000",
        "enddatetime": f"{end.strftime('%Y%m%d')}235959",
        "sort": "DateDesc",
    }
    try:
        response = requests.get(GDELT_DOC_API, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 - GDELT returns HTML on error and rate-limits hard
        logger.warning("GDELT fetch failed for %s (%s..%s): %s", symbol, start, end, exc)
        return pd.DataFrame(columns=NEWS_COLUMNS)

    articles = payload.get("articles", []) or []
    rows = []
    for article in articles:
        stamp = article.get("seendate")
        if not stamp:
            continue
        try:
            timestamp = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        rows.append(
            {
                "timestamp": timestamp.replace(tzinfo=None),
                "symbol": symbol,
                "source": "gdelt",
                "headline": article.get("title", "").strip(),
                "body": None,
            }
        )
    frame = pd.DataFrame(rows, columns=NEWS_COLUMNS)
    return frame[frame["headline"].astype(bool)] if not frame.empty else frame


def _fetch_newsapi(query: str, symbol: str, start: date, end: date) -> pd.DataFrame:
    """NewsAPI /everything. Needs NEWSAPI_KEY; free tier is limited to ~28 days of history."""
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        logger.debug("NEWSAPI_KEY unset; skipping NewsAPI for %s", symbol)
        return pd.DataFrame(columns=NEWS_COLUMNS)

    params = {
        "q": query,
        "from": start.isoformat(),
        "to": end.isoformat(),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
    }
    try:
        response = requests.get(
            NEWSAPI_EVERYTHING,
            params=params,
            headers={"X-Api-Key": api_key},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("NewsAPI fetch failed for %s (%s..%s): %s", symbol, start, end, exc)
        return pd.DataFrame(columns=NEWS_COLUMNS)

    rows = []
    for article in payload.get("articles", []) or []:
        published = article.get("publishedAt")
        if not published:
            continue
        try:
            timestamp = datetime.fromisoformat(published.replace("Z", "+00:00"))
        except ValueError:
            continue
        rows.append(
            {
                "timestamp": timestamp.replace(tzinfo=None),
                "symbol": symbol,
                "source": "newsapi",
                "headline": (article.get("title") or "").strip(),
                "body": article.get("description"),
            }
        )
    frame = pd.DataFrame(rows, columns=NEWS_COLUMNS)
    return frame[frame["headline"].astype(bool)] if not frame.empty else frame


def _dedupe_headlines(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop near-duplicate headlines.

    Wire stories get syndicated verbatim across dozens of outlets. Without this, a single
    press release would count as 40 headlines and inflate `sentiment_volume` -- which the
    model would read as a genuine attention spike.
    """
    if frame.empty:
        return frame
    normalized = frame["headline"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
    return frame.loc[~normalized.duplicated()].reset_index(drop=True)


def _fetch_news_frame(symbol: str, query: str, start: date, end: date) -> pd.DataFrame:
    """Route a date range across providers and merge.

    GDELT covers history; NewsAPI covers the recent window where its free tier works and its
    headlines are cleaner. Overlap is fine -- deduplication handles it.
    """
    cutoff = date.today() - timedelta(days=NEWSAPI_HORIZON_DAYS)
    frames = [_fetch_gdelt(query, symbol, start, end)]

    if end >= cutoff:
        frames.append(_fetch_newsapi(query, symbol, max(start, cutoff), end))

    merged = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(
        not f.empty for f in frames
    ) else pd.DataFrame(columns=NEWS_COLUMNS)

    if merged.empty:
        return merged
    merged = _dedupe_headlines(merged)
    return merged.sort_values("timestamp").reset_index(drop=True)


def fetch_news(
    symbol: str,
    start: date,
    end: date,
    *,
    query: str | None = None,
    force_refresh: bool = False,
) -> list[NewsItem]:
    """Headlines for a symbol, merged across enabled providers and de-duplicated.

    `query` is the company name (GDELT indexes prose, not tickers); it defaults to the
    symbol, which is right for ETFs but should be supplied from universe.yaml for equities.
    """
    search_term = query or symbol
    frame = cached_fetch(
        "news",
        symbol,
        start,
        end,
        lambda: _fetch_news_frame(symbol, search_term, start, end),
        force_refresh=force_refresh,
    )
    return frame_to_news(frame)


def fetch_social(
    symbol: str,
    start: date,
    end: date,
    *,
    subreddits: tuple[str, ...] = ("wallstreetbets", "stocks", "investing"),
    force_refresh: bool = False,
) -> list[NewsItem]:
    """Reddit submissions via praw, tagged source='reddit'. Disabled by default.

    Implemented against the same NewsItem schema so enabling it later is a config change,
    not a code change. Returns empty (with a warning) when credentials are absent, so a
    disabled feed degrades quietly instead of breaking the pipeline.
    """
    if not all(os.getenv(var) for var in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET")):
        logger.info("Reddit credentials unset; social feed disabled for %s", symbol)
        return []

    frame = cached_fetch(
        "social",
        symbol,
        start,
        end,
        lambda: _fetch_reddit_frame(symbol, start, end, subreddits),
        force_refresh=force_refresh,
    )
    return frame_to_news(frame)


def _fetch_reddit_frame(
    symbol: str, start: date, end: date, subreddits: tuple[str, ...]
) -> pd.DataFrame:
    """praw search across subreddits, filtered to the date range.

    Reddit's search API has no date filter, so we over-fetch and filter client-side.
    """
    try:
        import praw
    except ImportError:
        logger.warning("praw not installed; social feed unavailable")
        return pd.DataFrame(columns=NEWS_COLUMNS)

    try:
        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT", "j26-se-325-component1/0.1"),
            check_for_async=False,
        )
        rows: list[dict[str, Any]] = []
        start_ts = datetime.combine(start, datetime.min.time()).timestamp()
        end_ts = datetime.combine(end, datetime.max.time()).timestamp()

        for name in subreddits:
            for submission in reddit.subreddit(name).search(symbol, sort="new", limit=250):
                created = getattr(submission, "created_utc", None)
                if created is None or not (start_ts <= created <= end_ts):
                    continue
                rows.append(
                    {
                        "timestamp": datetime.fromtimestamp(created),
                        "symbol": symbol,
                        "source": "reddit",
                        "headline": submission.title,
                        "body": (submission.selftext or None),
                    }
                )
        return pd.DataFrame(rows, columns=NEWS_COLUMNS)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Reddit fetch failed for %s: %s", symbol, exc)
        return pd.DataFrame(columns=NEWS_COLUMNS)


def frame_to_news(frame: pd.DataFrame) -> list[NewsItem]:
    """Validate a normalized news frame into NewsItem models."""
    if frame.empty:
        return []
    records = frame.where(pd.notna(frame), None).to_dict(orient="records")
    return [NewsItem(**row) for row in records]
