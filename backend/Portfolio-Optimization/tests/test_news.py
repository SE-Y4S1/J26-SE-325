"""Phase 2 tests: keyword profiles, channels, relevance, scoring, aggregation, the
sentiment step and its endpoints.

Feeds and API payloads are RECORDED (tests/fixtures/news/, captured 2026-10-10 from Google
News, EconomyNext, GDELT and cse.lk). Sentiment models are stubbed: these tests check the
pipeline's logic, not FinBERT's opinions.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from data import active_set
from data.instruments import Exchange, Instrument
from data.schema import AssetClass
from news import aggregate as agg
from news import keywords, pipeline, relevance, scoring
from news.channels import agent_reach, public
from news.channels.base import queries_for, search_terms, utcnow
from news.models import KeywordProfile, RawItem, SourceKind, Tier

FIXTURES = Path(__file__).parent / "fixtures" / "news"

JKH = Instrument(symbol="JKH.N0000", name="JOHN KEELLS HOLDINGS PLC", exchange=Exchange.CSE,
                 asset_class=AssetClass.CSE_EQUITY, currency="LKR", provider_id="297")
COMB = Instrument(symbol="COMB.N0000", name="COMMERCIAL BANK OF CEYLON PLC", exchange=Exchange.CSE,
                  asset_class=AssetClass.CSE_EQUITY, currency="LKR", provider_id="208")
BTC = Instrument(symbol="BTCUSDT", name="BTC / USDT", exchange=Exchange.BINANCE,
                 asset_class=AssetClass.CRYPTO, currency="USD", provider_id="BTC")
NVDA = Instrument(symbol="NVDA", name="NVIDIA Corporation", exchange=Exchange.YAHOO,
                  asset_class=AssetClass.EQUITY, currency="USD")


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


@pytest.fixture(autouse=True)
def no_models(monkeypatch, tmp_path):
    """No Ollama, no Laya, isolated profile/sentiment stores."""
    monkeypatch.setattr(scoring, "laya_router", lambda: None)
    monkeypatch.setattr(keywords, "PROFILE_DIR", tmp_path / "keywords")
    monkeypatch.setattr(keywords, "OVERRIDES_PATH", tmp_path / "overrides.yaml")
    monkeypatch.setattr(pipeline, "SENTIMENT_DIR", tmp_path / "sentiment")
    public._cache.clear()


def item(title: str, *, days_ago: float = 0.5, kind: SourceKind = SourceKind.NEWS, channel: str = "test",
         text: str | None = None, url: str | None = None, author: str | None = None,
         engagement: dict | None = None) -> RawItem:
    return RawItem(channel=channel, kind=kind, title=title, text=text, url=url, author=author,
                   published_at=utcnow() - timedelta(days=days_ago), engagement=engagement or {})


# --------------------------------------------------------------------------------------
# Keyword profiles
# --------------------------------------------------------------------------------------

def test_cse_name_variants() -> None:
    names = keywords.name_variants(JKH)
    assert names[:3] == ["John Keells Holdings PLC", "John Keells Holdings", "John Keells"]
    assert "JKH" in names


def test_ambiguous_cse_ticker_is_not_a_term() -> None:
    names = keywords.name_variants(COMB)
    assert "Commercial Bank of Ceylon" in names
    assert "COMB" not in names


def test_crypto_uses_coin_names() -> None:
    assert keywords.name_variants(BTC)[:2] == ["Bitcoin", "BTC"]


def test_us_ticker_gets_cashtag() -> None:
    names = keywords.name_variants(NVDA)
    assert {"NVIDIA", "NVDA", "$NVDA"} <= set(names)


def test_profile_without_llm_uses_templates() -> None:
    profile = keywords.generate_profile(COMB, use_llm=False)
    assert profile.provenance == ["catalogue", "template"]
    assert "Central Bank of Sri Lanka" in profile.macro_terms
    assert "hair comb" in profile.exclude          # from ambiguous_tickers
    assert profile.source == "auto"


def test_llm_output_is_filtered(monkeypatch) -> None:
    def chatty_model(system, user, **kwargs):
        return {
            "aliases": ["JKH", "Keells", "bank", "A very long sentence that is clearly not a name at all"],
            "brands_subsidiaries": ["Keells Super", "Cinnamon Hotels", 42],
            "key_people": ["Krishan Balendra", "x"],
            "sector": "made_up_sector",
            "local_names": ["ජෝන් කීල්ස්", "John Keells in English"],
            "linkedin_slug": "john-keells-holdings-plc",
        }

    monkeypatch.setattr("news.llm.chat_json", chatty_model)
    profile = keywords.generate_profile(JKH)

    assert "Keells" in profile.company
    assert "bank" not in profile.company                       # generic word dropped
    assert not any("sentence" in t for t in profile.company)  # sentence-like dropped
    assert {"Keells Super", "Cinnamon Hotels", "Krishan Balendra"} <= set(profile.people_brands)
    assert profile.sector == "other"                           # invalid key -> default
    assert profile.local_names == ["ජෝන් කීල්ස්"]              # only real Sinhala script kept
    assert profile.linkedin_url == "https://www.linkedin.com/company/john-keells-holdings-plc"
    assert "llm" in profile.provenance


def test_llm_failure_still_yields_a_profile(monkeypatch) -> None:
    def down(*args, **kwargs):
        raise RuntimeError("Ollama is not running")

    monkeypatch.setattr("news.llm.chat_json", down)
    profile = keywords.generate_profile(JKH)
    assert "llm" not in profile.provenance
    assert "John Keells" in profile.company


def test_reviewed_profile_is_never_regenerated(monkeypatch) -> None:
    monkeypatch.setattr("news.llm.chat_json", lambda *a, **k: {})
    keywords.ensure_profile(JKH)
    keywords.update_profile("JKH.N0000", {"company": ["John Keells", "JKH", "Keells Group"]})

    again = keywords.ensure_profile(JKH)
    assert again.source == "reviewed"
    assert again.company == ["John Keells", "JKH", "Keells Group"]


def test_update_rejects_non_editable_fields(monkeypatch) -> None:
    monkeypatch.setattr("news.llm.chat_json", lambda *a, **k: {})
    keywords.ensure_profile(JKH)
    with pytest.raises(ValueError):
        keywords.update_profile("JKH.N0000", {"symbol": "HACK"})


def test_repo_overrides_win(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("news.llm.chat_json", lambda *a, **k: {})
    (tmp_path / "overrides.yaml").write_text(
        "COMB.N0000:\n  company: [ComBank, Commercial Bank of Ceylon]\n  sector: banks\n", encoding="utf-8")
    profile = keywords.ensure_profile(COMB)
    assert profile.company == ["ComBank", "Commercial Bank of Ceylon"]
    assert profile.sector == "banks"
    assert profile.source == "reviewed"


# --------------------------------------------------------------------------------------
# Queries and channels
# --------------------------------------------------------------------------------------

@pytest.fixture
def jkh_profile() -> KeywordProfile:
    # Without the LLM the sector is unknown ("other"); pin it so every tier has terms.
    profile = keywords.generate_profile(JKH, use_llm=False)
    terms = keywords.load_config()["sectors"]["conglomerate"]
    return profile.model_copy(update={"sector": "conglomerate", "sector_terms": terms})


def test_search_terms_prefer_full_names(jkh_profile) -> None:
    terms = search_terms(jkh_profile)
    assert terms[0] == "John Keells Holdings PLC"
    assert len(terms) == 3


def test_queries_cover_all_tiers(jkh_profile) -> None:
    tiers = [q.tier for q in queries_for(jkh_profile)]
    assert tiers == [Tier.COMPANY, Tier.SECTOR, Tier.MACRO]
    assert '"John Keells Holdings PLC" OR' in queries_for(jkh_profile)[0].text


def test_google_news_splits_outlet_from_headline() -> None:
    items = public.parse_rss(fixture_bytes("google_news_jkh.xml"), channel="google_news", split_outlet=True)
    assert len(items) == 12
    assert all(" - " not in i.title or i.outlet for i in items)
    assert any("Keells" in i.title for i in items)
    assert all(i.outlet for i in items)


def test_outlet_feed_parses_and_strips_html() -> None:
    items = public.parse_rss(fixture_bytes("economynext_feed.xml"), channel="rss:economynext", outlet="economynext")
    assert len(items) == 12
    assert all("<" not in (i.text or "") for i in items)


def test_gdelt_parse() -> None:
    items = public.parse_gdelt(json.loads(fixture_bytes("gdelt_jkh.json")))
    assert items and all(i.channel == "gdelt" and i.outlet for i in items)


def test_cse_announcements_are_official() -> None:
    items = public.parse_cse_announcements(json.loads(fixture_bytes("cse_announcements_jkh.json")), "JKH.N0000")
    assert len(items) == 12
    assert all(i.kind is SourceKind.OFFICIAL for i in items)
    assert "John Keells Holdings Plc: Dealings By Directors" in items[0].title.title()


def test_google_news_channel_queries_each_tier_once(jkh_profile, monkeypatch) -> None:
    calls = []

    class Response:
        content = fixture_bytes("google_news_jkh.xml")

    def fake_get(url, **kwargs):
        calls.append(url)
        return Response()

    monkeypatch.setattr(public, "http_get", fake_get)
    since = datetime(2000, 1, 1)
    public.GoogleNewsChannel().fetch(jkh_profile, since)
    public.GoogleNewsChannel().fetch(jkh_profile, since)     # second run: served from cache
    assert len(calls) == 3
    assert all("gl=LK" in url for url in calls)


def test_key_gated_channels_report_why(monkeypatch) -> None:
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    assert public.NewsApiChannel().status().reason == "NEWSAPI_KEY not set"
    assert public.JinaWebChannel().status().reason == "JINA_API_KEY not set"


SPEC = {"kind": "social", "binary": "definitely-not-installed-xyz",
        "command": ["definitely-not-installed-xyz", "search", "{query}"], "requires_env": ["TWITTER_AUTH_TOKEN"]}
FIELDS = keywords.load_config()["agent_reach_fields"]


def test_agent_reach_reports_not_installed() -> None:
    status = agent_reach.AgentReachChannel("twitter", SPEC, FIELDS).status()
    assert not status.available and status.requires_login
    assert "not installed" in status.reason


def test_agent_reach_reports_not_configured(monkeypatch) -> None:
    monkeypatch.setattr(agent_reach.shutil, "which", lambda binary: "/usr/bin/x")
    monkeypatch.delenv("TWITTER_AUTH_TOKEN", raising=False)
    assert "TWITTER_AUTH_TOKEN" in agent_reach.AgentReachChannel("twitter", SPEC, FIELDS).status().reason


@pytest.mark.parametrize(
    "stdout",
    [
        json.dumps([{"full_text": "John Keells up 5%", "created_at": "Wed Oct 08 10:00:00 +0000 2026",
                     "user": {"username": "trader"}, "favorite_count": 12, "url": "https://x.com/1"}]),
        json.dumps({"data": [{"content": "John Keells up 5%", "timestamp": 1791460800, "author": "trader"}]}),
        '{"text": "John Keells up 5%", "date": "2026-10-08T10:00:00Z"}\nnot json\n',
    ],
)
def test_agent_reach_output_shapes(stdout) -> None:
    items = agent_reach.AgentReachChannel("twitter", SPEC, FIELDS).parse(stdout)
    assert len(items) == 1
    assert items[0].text == "John Keells up 5%"
    assert items[0].published_at.year == 2026
    assert items[0].kind is SourceKind.SOCIAL


# --------------------------------------------------------------------------------------
# Relevance
# --------------------------------------------------------------------------------------

def test_relevance_tiers(jkh_profile) -> None:
    company = relevance.classify(item("John Keells profit up 218-pct"), jkh_profile)
    macro = relevance.classify(item("Central Bank of Sri Lanka holds policy rate"), jkh_profile)
    sector = relevance.classify(item("Colombo tea auction prices firm"), keywords.generate_profile(
        Instrument(symbol="WATA.N0000", name="WATAWALA PLANTATIONS PLC", exchange=Exchange.CSE,
                   asset_class=AssetClass.CSE_EQUITY, currency="LKR"), use_llm=False).model_copy(
        update={"sector_terms": ["Colombo tea auction"]}))
    unrelated = relevance.classify(item("Local football club wins league"), jkh_profile)

    assert company.tier is Tier.COMPANY
    assert macro.tier is Tier.MACRO
    assert sector.tier is Tier.SECTOR
    assert unrelated is None


def test_company_beats_macro_when_both_match(jkh_profile) -> None:
    hit = relevance.classify(item("John Keells shares rise after CBSL policy rate cut",
                                  text="Central Bank of Sri Lanka eased"), jkh_profile)
    assert hit.tier is Tier.COMPANY


def test_short_ticker_must_be_uppercase(jkh_profile) -> None:
    assert relevance.classify(item("JKH to raise capital"), jkh_profile).tier is Tier.COMPANY
    assert relevance.classify(item("visit jkh.example.org for details"), jkh_profile) is None


def test_exclusion_terms_veto() -> None:
    profile = keywords.generate_profile(COMB, use_llm=False).model_copy(update={"company": ["Commercial Bank of Ceylon", "Comb"]})
    assert relevance.classify(item("Best hair comb for curly hair"), profile) is None
    assert relevance.classify(item("Commercial Bank of Ceylon posts record profit"), profile) is not None


def test_sinhala_name_matches() -> None:
    profile = keywords.generate_profile(JKH, use_llm=False).model_copy(update={"local_names": ["ජෝන් කීල්ස්"]})
    hit = relevance.classify(item("ජෝන් කීල්ස් ලාභය ඉහළට"), profile)
    assert hit is not None and hit.tier is Tier.COMPANY


def test_semantic_stage_is_reported_as_skipped_without_laya(jkh_profile) -> None:
    kept, method = relevance.filter_relevant([item("John Keells profit up")], jkh_profile)
    assert method == "keyword" and len(kept) == 1


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize(("text", "lang"), [("John Keells profit", "en"), ("ජෝන් කීල්ස්", "si"), ("ஜோன் கீல்ஸ்", "ta")])
def test_language_detection(text, lang) -> None:
    assert scoring.detect_language(text) == lang


@pytest.fixture
def stub_english(monkeypatch):
    import pandas as pd

    from features import sentiment

    def fake_score(items, *, backend="auto", batch_size=32):
        return pd.DataFrame([{"timestamp": i.timestamp, "symbol": i.symbol, "source": i.source,
                              "polarity": -0.8 if "falls" in i.headline else 0.6, "confidence": 0.9}
                             for i in items])

    monkeypatch.setattr(sentiment, "resolve_backend", lambda preference="auto": "ollama")
    monkeypatch.setattr(sentiment, "score_headlines", fake_score)


def test_local_language_is_left_unscored_without_laya(jkh_profile, stub_english) -> None:
    relevant, _ = relevance.filter_relevant([
        item("John Keells profit rises"),
        item("ජෝන් කීල්ස් ලාභය"),
    ], jkh_profile.model_copy(update={"local_names": ["ජෝන් කීල්ස්"]}))
    scored, backend = scoring.score_items(relevant)

    assert backend == "ollama"
    english = next(s for s in scored if s.language == "en")
    sinhala = next(s for s in scored if s.language == "si")
    assert english.polarity == pytest.approx(0.6)
    assert sinhala.polarity is None


def test_no_backend_leaves_everything_unscored(jkh_profile, monkeypatch) -> None:
    from features import sentiment

    def none_available(preference="auto"):
        raise RuntimeError("no sentiment backend available")

    monkeypatch.setattr(sentiment, "resolve_backend", none_available)
    relevant, _ = relevance.filter_relevant([item("John Keells profit rises")], jkh_profile)
    scored, backend = scoring.score_items(relevant)
    assert backend is None and scored[0].polarity is None


@pytest.mark.parametrize(
    ("answer", "expected"),
    [({"choice": "negative", "probabilities": {"positive": 0.1, "negative": 0.7, "neutral": 0.2}}, (-0.6, 0.8)),
     ({"choice": "positive"}, (0.5, 0.5)),
     ({}, None)],
)
def test_laya_choice_to_polarity(answer, expected) -> None:
    result = scoring._choice_to_polarity(answer)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------

def scored(raw: RawItem, tier: Tier = Tier.COMPANY, polarity: float | None = 0.5, confidence: float = 1.0,
           language: str = "en") -> scoring.ScoredItem:
    return scoring.ScoredItem(relevance.RelevantItem(raw, tier, ("x",)), language, polarity,
                              confidence if polarity is not None else None, "stub")


def test_duplicates_across_channels_collapse_preferring_official() -> None:
    a = scored(item("JKH declares dividend", kind=SourceKind.NEWS, channel="gdelt"))
    b = scored(item("JKH declares dividend", kind=SourceKind.OFFICIAL, channel="cse_announcements"))
    kept = agg.dedupe([a, b])
    assert len(kept) == 1 and kept[0].relevant.item.kind is SourceKind.OFFICIAL


def test_spam_is_dropped() -> None:
    copy = "Buy JKH now guaranteed 10x returns join our group today"
    posts = [scored(item(copy, kind=SourceKind.SOCIAL, text=copy, author=f"bot{i}", url=f"https://x/{i}"))
             for i in range(3)]
    cashtags = scored(item("x", kind=SourceKind.SOCIAL, text="$A $B $C $D $E $F John Keells", author="z"))
    real = scored(item("Thoughtful post", kind=SourceKind.SOCIAL, text="John Keells results look solid this quarter", author="h"))
    kept, dropped = agg.drop_spam([*posts, cashtags, real])
    assert dropped == 4 and kept == [real]


def test_recent_items_outweigh_old_ones() -> None:
    now = utcnow()
    fresh = agg.item_weight(scored(item("a", days_ago=0.1)), now, tau_days=3.0)
    stale = agg.item_weight(scored(item("b", days_ago=10)), now, tau_days=3.0)
    assert fresh > 10 * stale


def test_official_outweighs_social() -> None:
    now = utcnow()
    official = agg.item_weight(scored(item("a", kind=SourceKind.OFFICIAL)), now, tau_days=3.0)
    social = agg.item_weight(scored(item("b", kind=SourceKind.SOCIAL)), now, tau_days=3.0)
    assert official > social


def test_aggregate_tiers_momentum_and_spike() -> None:
    now = utcnow()
    older = [scored(item(f"older {i}", days_ago=8 + i * 0.1), polarity=0.4) for i in range(6)]
    burst = [scored(item(f"burst {i}", days_ago=0.2 + i * 0.05), polarity=-0.7) for i in range(8)]
    macro = [scored(item("cbsl", days_ago=1), tier=Tier.MACRO, polarity=0.2)]
    block = agg.aggregate([*older, *burst, *macro], now=now)

    company = block["tiers"]["company"]
    assert company["items"] == 14
    assert company["momentum"] < 0                  # turned negative recently
    assert company["attention_spike"] > 2           # 8 items in 2 days vs ~0.5/day baseline
    assert block["tiers"]["sector"]["mean_sentiment"] is None
    assert block["overall_sentiment"] is not None


def test_social_sources_carry_no_text() -> None:
    post = scored(item("John Keells to the moon", kind=SourceKind.SOCIAL, url="https://x.com/1",
                       text="John Keells to the moon", author="a"))
    sources = agg.top_sources([post], utcnow(), tau_days=3.0)
    assert sources[0]["title"] is None and sources[0]["url"] == "https://x.com/1"


def test_daily_rows_have_no_text() -> None:
    rows = agg.daily_rows([scored(item("John Keells profit"))], "JKH.N0000")
    assert set(rows[0]) == {"symbol", "date", "tier", "items", "scored", "mean_sentiment"}


# --------------------------------------------------------------------------------------
# Sentiment step end to end
# --------------------------------------------------------------------------------------

class FakeChannel:
    def __init__(self, name, items=None, error=None, available=True, regions=()):
        self.name, self._items, self._error, self._available, self.regions = name, items or [], error, available, regions

    def applies_to(self, profile):
        return not self.regions or profile.exchange in self.regions

    def status(self):
        from news.channels.base import ChannelStatus

        return ChannelStatus(self.name, "news", self._available, None if self._available else "not installed")

    def fetch(self, profile, since, *, limit=40):
        if self._error:
            raise self._error
        return self._items


def test_sentiment_step_survives_a_failing_channel(monkeypatch, stub_english) -> None:
    monkeypatch.setattr("news.llm.chat_json", lambda *a, **k: {})
    channels = [
        FakeChannel("good", [item("John Keells profit rises"), item("John Keells shares falls on results")]),
        FakeChannel("broken", error=TimeoutError("GDELT timed out")),
        FakeChannel("twitter", available=False),
        FakeChannel("crypto_only", [item("Bitcoin rallies")], regions=("binance",)),
    ]
    monkeypatch.setattr(pipeline, "build_channels", lambda: channels)

    ok, block = pipeline.sentiment_step(JKH)
    assert ok
    assert block["channels"]["good"] == {"status": "ok", "items": 2}
    assert block["channels"]["broken"]["status"] == "error"
    assert block["channels"]["twitter"]["status"] == "unavailable"
    assert "crypto_only" not in block["channels"]            # region filter
    assert block["tiers"]["company"]["items"] == 2
    assert block["relevance_method"] == "keyword"
    assert (pipeline.SENTIMENT_DIR).exists()


def test_sentiment_step_fails_when_no_channel_works(monkeypatch, stub_english) -> None:
    monkeypatch.setattr("news.llm.chat_json", lambda *a, **k: {})
    monkeypatch.setattr(pipeline, "build_channels", lambda: [FakeChannel("broken", error=OSError("offline"))])
    ok, detail = pipeline.sentiment_step(JKH)
    assert not ok and "offline" in detail


def test_daily_history_merges_by_day(monkeypatch) -> None:
    import pandas as pd

    row = {"symbol": "JKH.N0000", "date": "2026-10-09", "tier": "company", "items": 1, "scored": 1, "mean_sentiment": 0.1}
    pipeline.store_daily("JKH.N0000", [row])
    pipeline.store_daily("JKH.N0000", [{**row, "items": 3, "mean_sentiment": 0.4}])
    stored = pd.read_parquet(next(pipeline.SENTIMENT_DIR.glob("*.parquet")))
    assert len(stored) == 1 and stored.iloc[0]["items"] == 3


# --------------------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch, tmp_path, stub_english):
    from service import events as events_mod
    from service.api import app
    from service.events import NullProducer

    monkeypatch.setattr(events_mod, "_producer_cache", {"producer": NullProducer()})
    monkeypatch.setattr(active_set, "REGISTRY_PATH", tmp_path / "active.sqlite")
    monkeypatch.setattr("news.llm.chat_json", lambda *a, **k: {})
    monkeypatch.setattr(pipeline, "build_channels",
                        lambda: [FakeChannel("good", [item("John Keells profit rises")])])
    monkeypatch.setattr(active_set, "PIPELINE", [
        ("keywords", active_set._keywords_step), ("sentiment", active_set._sentiment_step)])
    with TestClient(app) as test_client:
        yield test_client


def test_channels_endpoint_lists_everything(client) -> None:
    channels = {c["name"]: c for c in client.get("/analysis/channels").json()["channels"]}
    assert {"google_news", "gdelt", "rss", "cse_announcements", "twitter", "instagram", "linkedin"} <= channels.keys()
    assert channels["google_news"]["available"] is True
    assert channels["instagram"]["requires_login"] is True


def test_keywords_and_sentiment_endpoints(client) -> None:
    active_set.track([JKH])
    active_set.prepare("JKH.N0000")

    profile = client.get("/analysis/keywords/JKH.N0000").json()
    assert "John Keells" in profile["company"]

    edited = client.put("/analysis/keywords/JKH.N0000", json={"exclude": ["jkh.example.org"]}).json()
    assert edited["source"] == "reviewed" and edited["exclude"] == ["jkh.example.org"]

    sentiment = client.get("/analysis/sentiment/JKH.N0000").json()
    assert sentiment["status"] == "ready"
    assert sentiment["sentiment"]["tiers"]["company"]["items"] == 1
    assert sentiment["sentiment"]["scoring_backend"] == "ollama"


def test_keyword_update_rejects_unknown_fields(client) -> None:
    active_set.track([JKH])
    active_set.prepare("JKH.N0000")
    assert client.put("/analysis/keywords/JKH.N0000", json={"symbol": "X"}).status_code == 422


def test_sentiment_for_untracked_symbol_is_404(client) -> None:
    assert client.get("/analysis/sentiment/NOPE.N0000").status_code == 404


# --------------------------------------------------------------------------------------
# Regressions from the first live run
# --------------------------------------------------------------------------------------

def test_foreign_sector_namesakes_are_not_cse_sector_news(jkh_profile) -> None:
    """Live run 2026-10-10: 'diversified holdings' pulled in two US companies."""
    profile = jkh_profile.model_copy(update={"sector_terms": ["diversified conglomerate"]})
    us = item("Compass Diversified Conglomerate declares dividend", text="NYSE-listed")
    lk = item("Sri Lanka's diversified conglomerate sector sees record profits")
    assert relevance.classify(us, profile) is None
    assert relevance.classify(lk, profile).tier is Tier.SECTOR


def test_sri_lankan_outlet_supplies_context(jkh_profile) -> None:
    profile = jkh_profile.model_copy(update={"macro_terms": ["inflation"]})
    local = item("Inflation rises for third month").model_copy(update={"outlet": "EconomyNext"})
    foreign = item("Inflation rises for third month").model_copy(update={"outlet": "CNBC"})
    assert relevance.classify(local, profile).tier is Tier.MACRO
    assert relevance.classify(foreign, profile) is None


def test_crypto_macro_needs_no_country_context() -> None:
    profile = keywords.generate_profile(BTC, use_llm=False)
    hit = relevance.classify(item("SEC crypto rules tighten", text="crypto regulation"), profile)
    assert hit is not None and hit.tier is Tier.MACRO


def test_gdelt_calls_are_spaced(monkeypatch) -> None:
    import time as time_mod

    sleeps = []
    monkeypatch.setattr(time_mod, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(public.GdeltChannel, "_last_call", time_mod.monotonic())
    public.GdeltChannel._throttle()
    assert sleeps and sleeps[0] > 5
