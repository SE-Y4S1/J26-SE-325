"""Channels that need no login.

  google_news        keyword search over Google News RSS. The broadest per-stock news source
                     found: 100 results for "John Keells" across EconomyNext, Ada Derana,
                     Daily FT, TradingView... Sri Lankan edition for CSE symbols.
  gdelt              GDELT DOC 2.0 keyword search (free, global, multilingual).
  rss                configured outlet feeds, filtered by keyword afterwards.
  cse_announcements  the company's own CSE disclosures (official source).
  newsapi            when NEWSAPI_KEY is set.
  youtube            keyword video search via yt-dlp, when installed.
  jina_web           web search via Jina, when JINA_API_KEY is set. Also runs a
                     site:linkedin.com search, which is how public LinkedIn posts are reached
                     without a login (anonymous LinkedIn page reads are rate-blocked).

Every network call goes through `http_get`, which the tests replace with recorded payloads.
"""

from __future__ import annotations

import email.utils
import logging
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from importlib.util import find_spec
from typing import Any
from urllib.parse import quote_plus

import requests

from news.channels.base import (
    USER_AGENT,
    Channel,
    ChannelStatus,
    TTLCache,
    or_query,
    queries_for,
    search_terms,
    strip_html,
    utcnow,
)
from news.models import KeywordProfile, RawItem, SourceKind

logger = logging.getLogger(__name__)

TIMEOUT = 30
_cache = TTLCache(ttl_seconds=1800)


def http_get(url: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None,
             method: str = "GET", data: dict[str, Any] | None = None) -> requests.Response:
    """The single network seam for public channels."""
    merged = {"User-Agent": USER_AGENT, **(headers or {})}
    response = requests.request(method, url, params=params, headers=merged, data=data,
                                timeout=TIMEOUT, allow_redirects=True)
    response.raise_for_status()
    return response


def _naive_utc(stamp: datetime) -> datetime:
    return stamp.astimezone(timezone.utc).replace(tzinfo=None) if stamp.tzinfo else stamp


def _parse_rfc822(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return _naive_utc(email.utils.parsedate_to_datetime(value))
    except (TypeError, ValueError):
        return None


def parse_rss(xml_text: str | bytes, *, channel: str, kind: SourceKind = SourceKind.NEWS,
              outlet: str | None = None, split_outlet: bool = False) -> list[RawItem]:
    """RSS 2.0 items -> RawItem. `split_outlet` handles Google News' 'Headline - Outlet'."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("%s: unparseable feed (%s)", channel, exc)
        return []

    items = []
    for node in root.iter("item"):
        title = strip_html(node.findtext("title")) or ""
        published = _parse_rfc822(node.findtext("pubDate"))
        if not title or published is None:
            continue
        item_outlet = outlet or node.findtext("source")
        if split_outlet and " - " in title:
            head, tail = title.rsplit(" - ", 1)
            title, item_outlet = head.strip(), item_outlet or tail.strip()
        items.append(
            RawItem(
                channel=channel, kind=kind, title=title,
                text=strip_html(node.findtext("description")),
                url=node.findtext("link"), published_at=published, outlet=item_outlet,
            )
        )
    return items


def _recent(items: list[RawItem], since: datetime) -> list[RawItem]:
    return [i for i in items if i.published_at >= since]


# --------------------------------------------------------------------------------------

class GoogleNewsChannel(Channel):
    name = "google_news"
    kind = SourceKind.NEWS

    EDITIONS = {"cse": ("en-LK", "LK", "LK:en")}
    DEFAULT_EDITION = ("en-US", "US", "US:en")

    def status(self) -> ChannelStatus:
        return self._status(True)

    def fetch(self, profile: KeywordProfile, since: datetime, *, limit: int = 40) -> list[RawItem]:
        hl, gl, ceid = self.EDITIONS.get(profile.exchange, self.DEFAULT_EDITION)
        days = min(30, max(1, (utcnow() - since).days))
        out: list[RawItem] = []
        for query in queries_for(profile):
            key = (self.name, query.text, gl)
            items = _cache.get(key)
            if items is None:
                url = (f"https://news.google.com/rss/search?q={quote_plus(query.text + f' when:{days}d')}"
                       f"&hl={hl}&gl={gl}&ceid={ceid}")
                items = parse_rss(http_get(url).content, channel=self.name, split_outlet=True)
                _cache.put(key, items)
            out.extend(_recent(items, since)[:limit])
        return out


class GdeltChannel(Channel):
    name = "gdelt"
    kind = SourceKind.NEWS
    API = "https://api.gdeltproject.org/api/v2/doc/doc"
    # GDELT allows one request per 5 seconds per IP and answers faster callers with 429.
    # Preparing several symbols back to back hit that in the first live run.
    MIN_INTERVAL_SECONDS = 6.0
    _last_call = 0.0

    @classmethod
    def _throttle(cls) -> None:
        import time

        wait = cls.MIN_INTERVAL_SECONDS - (time.monotonic() - cls._last_call)
        if wait > 0:
            time.sleep(wait)
        cls._last_call = time.monotonic()

    def status(self) -> ChannelStatus:
        return self._status(True)

    def fetch(self, profile: KeywordProfile, since: datetime, *, limit: int = 40) -> list[RawItem]:
        terms = search_terms(profile)
        if not terms:
            return []
        days = max(1, (utcnow() - since).days)
        # GDELT requires OR'd terms in parentheses.
        query = f"({or_query(terms)})" if len(terms) > 1 else or_query(terms)
        key = (self.name, query, days)
        items = _cache.get(key)
        if items is None:
            self._throttle()
            payload = http_get(self.API, params={
                "query": query, "mode": "ArtList", "format": "json",
                "maxrecords": min(limit, 250), "timespan": f"{days}d", "sort": "DateDesc",
            }).json()
            items = parse_gdelt(payload)
            _cache.put(key, items)
        return _recent(items, since)[:limit]


def parse_gdelt(payload: dict[str, Any]) -> list[RawItem]:
    items = []
    for article in payload.get("articles", []) or []:
        title = (article.get("title") or "").strip()
        try:
            stamp = datetime.strptime(article.get("seendate", ""), "%Y%m%dT%H%M%SZ")
        except ValueError:
            continue
        if title:
            items.append(RawItem(channel="gdelt", kind=SourceKind.NEWS, title=title,
                                 url=article.get("url"), published_at=stamp,
                                 outlet=article.get("domain")))
    return items


class RssChannel(Channel):
    """Outlet feeds. Not a search: every item is returned and relevance filtering decides."""

    name = "rss"
    kind = SourceKind.NEWS
    searches = False

    def __init__(self, feeds: list[dict[str, Any]]) -> None:
        self.feeds = feeds

    def status(self) -> ChannelStatus:
        return self._status(bool(self.feeds), None if self.feeds else "no feeds configured")

    def fetch(self, profile: KeywordProfile, since: datetime, *, limit: int = 40) -> list[RawItem]:
        out: list[RawItem] = []
        for feed in self.feeds:
            if feed.get("region") and profile.exchange not in feed["region"]:
                continue
            key = (self.name, feed["url"])
            items = _cache.get(key)
            if items is None:
                try:
                    items = parse_rss(http_get(feed["url"]).content, channel=f"rss:{feed['name']}",
                                      outlet=feed["name"])
                except requests.RequestException as exc:
                    logger.info("feed %s unavailable: %s", feed["name"], exc)
                    items = []
                _cache.put(key, items)
            out.extend(_recent(items, since))
        return out


class CseAnnouncementsChannel(Channel):
    """The company's own CSE disclosures. Categories only (the text is usually a PDF), so
    these count toward attention and appear as sources, but are rarely scoreable."""

    name = "cse_announcements"
    kind = SourceKind.OFFICIAL
    regions = ("cse",)
    API = "https://www.cse.lk/api/getAnnouncementByCompany"

    def status(self) -> ChannelStatus:
        return self._status(True)

    def fetch(self, profile: KeywordProfile, since: datetime, *, limit: int = 40) -> list[RawItem]:
        payload = http_get(self.API, method="POST", data={"symbol": profile.symbol}).json()
        return _recent(parse_cse_announcements(payload, profile.symbol), since)[:limit]


def parse_cse_announcements(payload: dict[str, Any], symbol: str) -> list[RawItem]:
    items = []
    for row in payload.get("reqCompanyAnnouncement", []) or []:
        created = row.get("createdDate")
        if not created:
            continue
        category = (row.get("announcementCategory") or "Announcement").title()
        remarks = (row.get("remarks") or "").strip()
        title = f"{row.get('company') or symbol}: {category}" + (f" - {remarks}" if remarks else "")
        items.append(RawItem(
            channel="cse_announcements", kind=SourceKind.OFFICIAL, title=title,
            text=remarks or None,
            url=f"https://www.cse.lk/pages/company-profile/company-profile.component.html?symbol={symbol}",
            published_at=datetime.fromtimestamp(created / 1000, tz=timezone.utc).replace(tzinfo=None),
            outlet="CSE",
        ))
    return items


class NewsApiChannel(Channel):
    name = "newsapi"
    kind = SourceKind.NEWS

    def status(self) -> ChannelStatus:
        return self._status(bool(os.getenv("NEWSAPI_KEY")), None if os.getenv("NEWSAPI_KEY") else "NEWSAPI_KEY not set")

    def fetch(self, profile: KeywordProfile, since: datetime, *, limit: int = 40) -> list[RawItem]:
        terms = search_terms(profile)
        if not terms:
            return []
        payload = http_get(
            "https://newsapi.org/v2/everything",
            params={"q": or_query(terms), "from": since.date().isoformat(), "language": "en",
                    "sortBy": "publishedAt", "pageSize": min(limit, 100)},
            headers={"X-Api-Key": os.environ["NEWSAPI_KEY"]},
        ).json()
        items = []
        for article in payload.get("articles", []) or []:
            try:
                stamp = _naive_utc(datetime.fromisoformat(article["publishedAt"].replace("Z", "+00:00")))
            except (KeyError, ValueError, AttributeError):
                continue
            if article.get("title"):
                items.append(RawItem(channel=self.name, kind=SourceKind.NEWS, title=article["title"],
                                     text=article.get("description"), url=article.get("url"),
                                     published_at=stamp, outlet=(article.get("source") or {}).get("name")))
        return items


class YouTubeChannel(Channel):
    """Video search via yt-dlp (what Agent-Reach uses for YouTube). Titles and descriptions
    only; transcripts would multiply the cost for little extra signal in a headline model."""

    name = "youtube"
    kind = SourceKind.SOCIAL

    def status(self) -> ChannelStatus:
        ok = find_spec("yt_dlp") is not None
        return self._status(ok, None if ok else "yt-dlp not installed (pip install yt-dlp)")

    def fetch(self, profile: KeywordProfile, since: datetime, *, limit: int = 10) -> list[RawItem]:
        import yt_dlp

        terms = search_terms(profile, max_terms=1)
        if not terms:
            return []
        # ignoreerrors: one removed/private video in the results must not abort the search
        # (it did on the first live run); such entries come back as None and are skipped.
        options = {"quiet": True, "skip_download": True, "noplaylist": True, "extract_flat": False,
                   "ignoreerrors": True, "no_warnings": True}
        with yt_dlp.YoutubeDL(options) as ydl:
            result = ydl.extract_info(f"ytsearch{min(limit, 10)}:{terms[0]}", download=False)
        items = []
        for entry in (result or {}).get("entries", []) or []:
            if not entry:
                continue
            stamp = entry.get("upload_date")
            if not stamp or not entry.get("title"):
                continue
            items.append(RawItem(
                channel=self.name, kind=SourceKind.SOCIAL, title=entry["title"],
                text=(entry.get("description") or "")[:1000] or None,
                url=entry.get("webpage_url"), published_at=datetime.strptime(stamp, "%Y%m%d"),
                outlet=entry.get("channel"),
                engagement={k: float(entry[v]) for k, v in (("likes", "like_count"), ("views", "view_count"),
                                                             ("comments", "comment_count")) if entry.get(v)},
            ))
        return _recent(items, since)


class JinaWebChannel(Channel):
    """Web search through Jina (needs JINA_API_KEY), plus a site:linkedin.com pass."""

    name = "jina_web"
    kind = SourceKind.NEWS
    API = "https://s.jina.ai/"

    def status(self) -> ChannelStatus:
        ok = bool(os.getenv("JINA_API_KEY"))
        return self._status(ok, None if ok else "JINA_API_KEY not set")

    def fetch(self, profile: KeywordProfile, since: datetime, *, limit: int = 20) -> list[RawItem]:
        terms = search_terms(profile, max_terms=2)
        if not terms:
            return []
        headers = {"Authorization": f"Bearer {os.environ['JINA_API_KEY']}", "Accept": "application/json",
                   "X-Respond-With": "no-content"}
        out: list[RawItem] = []
        for query, kind in ((or_query(terms), SourceKind.NEWS),
                            (f"{or_query(terms)} site:linkedin.com", SourceKind.SOCIAL)):
            payload = http_get(self.API, params={"q": query}, headers=headers).json()
            for row in (payload.get("data") or [])[:limit]:
                stamp = row.get("date") or row.get("publishedTime")
                try:
                    published = _naive_utc(datetime.fromisoformat(str(stamp).replace("Z", "+00:00")))
                except ValueError:
                    # Undated web pages cannot be placed on the sentiment timeline.
                    continue
                if row.get("title"):
                    out.append(RawItem(channel="linkedin_public" if kind is SourceKind.SOCIAL else self.name,
                                       kind=kind, title=row["title"], text=row.get("description"),
                                       url=row.get("url"), published_at=published))
        return _recent(out, since)


def default_since(days: int) -> datetime:
    return utcnow() - timedelta(days=days)
