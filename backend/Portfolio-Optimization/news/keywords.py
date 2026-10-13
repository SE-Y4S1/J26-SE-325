"""Per-symbol keyword profiles: generated automatically, correctable by hand.

Hand-writing profiles for 282 CSE securities plus any crypto pair or US stock is not
feasible, so a profile is GENERATED the first time a symbol enters the active set:

  1. catalogue  name variants from the exchange's own listing ("JOHN KEELLS HOLDINGS PLC"
                -> "John Keells Holdings PLC", "John Keells Holdings", "John Keells");
                coin names for crypto ("BTCUSDT" -> "Bitcoin", "BTC").
  2. template   sector vocabulary (configs/news.yaml `sectors`) and the exchange's macro
                vocabulary (`macro`), plus exclusion terms for ambiguous tickers.
  3. llm        optional. The local Ollama model proposes aliases, brands/subsidiaries,
                key people, the sector key and Sinhala/Tamil names. A 3B model can and will
                invent things, so its output is size-limited and filtered, and the profile
                stays `source: auto` until someone reviews it.

A profile saved with `source: reviewed` (from the review screen) is never regenerated over.
Version-controlled corrections in configs/keyword_overrides.yaml are applied on top of
whatever is stored, so a fix committed to the repo always wins.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from data.instruments import Exchange, Instrument
from news.models import KeywordProfile

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "news.yaml"
OVERRIDES_PATH = ROOT / "configs" / "keyword_overrides.yaml"
PROFILE_DIR = ROOT / "artifacts" / "keywords"

# Corporate suffixes stripped to get the name people actually write.
_SUFFIXES = (
    "PLC", "LIMITED", "LTD", "INC", "INCORPORATED", "CORPORATION", "CORP", "COMPANY", "CO",
    "HOLDINGS", "HOLDING", "SA", "AG", "NV", "SE",
)
# Words that keep their capitalisation when the CSE's all-caps names are title-cased.
_KEEP_UPPER = {"PLC", "LTD", "CIC", "HNB", "DFCC", "NDB", "LOLC", "CTC", "SLT", "LB", "PABC",
               "HDFC", "ACL", "LMF", "CDB", "AIA", "JKH", "LTL", "BRC", "UK", "US", "IT"}
_KEEP_LOWER = {"of", "and", "the", "for", "&"}

MAX_LLM_TERMS = {"aliases": 5, "brands_subsidiaries": 8, "key_people": 3, "local_names": 2}


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}


# --------------------------------------------------------------------------------------
# Layer 1: names from the catalogue
# --------------------------------------------------------------------------------------

def title_case(name: str) -> str:
    """'COMMERCIAL BANK OF CEYLON PLC' -> 'Commercial Bank of Ceylon PLC'."""
    words = []
    for index, word in enumerate(name.split()):
        bare = word.strip(".,()")
        if bare.upper() in _KEEP_UPPER:
            words.append(word.upper())
        elif bare.lower() in _KEEP_LOWER and index > 0:
            words.append(word.lower())
        else:
            words.append(word.capitalize() if word.isupper() else word)
    return " ".join(words)


def strip_suffixes(name: str) -> str:
    """Drop trailing corporate suffixes: 'John Keells Holdings PLC' -> 'John Keells'."""
    words = name.replace(",", " ").split()
    while len(words) > 1 and words[-1].strip(".").upper() in _SUFFIXES:
        words.pop()
    return " ".join(words)


def name_variants(instrument: Instrument) -> list[str]:
    """Distinct ways the instrument's name appears in print, most specific first."""
    config = load_config()

    if instrument.exchange is Exchange.BINANCE:
        base = (instrument.provider_id or instrument.symbol).upper()
        return _dedupe([*config.get("coin_names", {}).get(base, []), base])

    # Only re-case names published in all caps (CSE). "NVIDIA Corporation" is already how the
    # press writes it; title-casing would turn it into "Nvidia".
    raw_name = instrument.name.strip()
    full = title_case(raw_name) if raw_name.isupper() else raw_name
    without_plc = re.sub(r"\s+PLC\.?$", "", full, flags=re.IGNORECASE)
    core = strip_suffixes(full)
    variants = [full, without_plc, core]

    ticker = instrument.symbol.split(".")[0].upper()
    if instrument.exchange is Exchange.YAHOO and "=" not in ticker:
        # US tickers are written in prose and as cashtags ("$NVDA").
        variants += [ticker, f"${ticker}"]
    elif instrument.exchange is Exchange.CSE and ticker not in ambiguous_tickers() and len(ticker) >= 3:
        variants.append(ticker)
    return _dedupe(v for v in variants if len(v) >= 2)


@lru_cache(maxsize=1)
def ambiguous_tickers() -> dict[str, list[str]]:
    return {k.upper(): list(v or []) for k, v in (load_config().get("ambiguous_tickers") or {}).items()}


def _dedupe(values) -> list[str]:  # noqa: ANN001 - any iterable of str
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        value = value.strip()
        if value and value.lower() not in seen:
            seen.add(value.lower())
            out.append(value)
    return out


# --------------------------------------------------------------------------------------
# Layer 2: templates
# --------------------------------------------------------------------------------------

def default_sector(instrument: Instrument) -> str:
    """Sector key when nothing better is known."""
    if instrument.exchange is Exchange.BINANCE:
        return "crypto_layer1"
    if instrument.asset_class.value == "forex":
        return "forex"
    if instrument.asset_class.value == "etf":
        return "broad_market_etf"
    return "other"


def template_terms(instrument: Instrument, sector: str) -> tuple[list[str], list[str]]:
    config = load_config()
    sectors = config.get("sectors", {})
    macro = config.get("macro", {}).get(instrument.exchange.value, [])
    return list(sectors.get(sector, []) or []), list(macro)


# --------------------------------------------------------------------------------------
# Layer 3: local LLM expansion
# --------------------------------------------------------------------------------------

_LLM_SYSTEM = (
    "You help build news-search keyword lists for a listed company or crypto asset. "
    "Only include facts you are confident about. Leave a list empty rather than guess. "
    "Reply with ONLY a JSON object with these keys: "
    '"aliases" (other names or abbreviations the press uses for it), '
    '"brands_subsidiaries" (well-known brands or subsidiaries it owns), '
    '"key_people" (current chairperson / CEO only), '
    '"sector" (exactly one key from the list provided), '
    '"local_names" (its name written in Sinhala or Tamil script, if it is Sri Lankan), '
    '"linkedin_slug" (the company\'s LinkedIn company-page slug, or empty).'
)

_GENERIC = {"bank", "hotel", "hotels", "group", "company", "holdings", "plc", "limited", "crypto",
            "coin", "token", "sri lanka", "colombo", "finance", "insurance", "the", "stock"}


def _clean_terms(values: Any, limit: int) -> list[str]:
    """Keep short, specific strings; drop generic words and anything sentence-like."""
    if not isinstance(values, list):
        return []
    cleaned = []
    for value in values:
        if not isinstance(value, str):
            continue
        value = value.strip().strip("\"'")
        if not (2 <= len(value) <= 60) or len(value.split()) > 6:
            continue
        if value.lower() in _GENERIC:
            continue
        cleaned.append(value)
    return _dedupe(cleaned)[:limit]


def llm_expand(instrument: Instrument, known_names: list[str]) -> dict[str, Any] | None:
    """Ask the local model for aliases etc. None when Ollama is unavailable or fails."""
    from news.llm import chat_json

    sector_keys = [k for k in load_config().get("sectors", {}) if k != "other"]
    exchange_label = {"cse": "Colombo Stock Exchange (Sri Lanka)", "binance": "Binance (crypto)",
                      "yahoo": "US / international market"}[instrument.exchange.value]
    user = (
        f"Instrument: {instrument.name} (symbol {instrument.symbol}, {exchange_label}).\n"
        f"Names already known: {', '.join(known_names)}.\n"
        f"Allowed sector keys: {', '.join(sector_keys)}."
    )
    try:
        payload = chat_json(_LLM_SYSTEM, user)
    except Exception as exc:  # noqa: BLE001 - the profile is usable without this layer
        logger.info("LLM keyword expansion for %s skipped: %s", instrument.symbol, exc)
        return None

    sector = payload.get("sector")
    slug = payload.get("linkedin_slug")
    local = _clean_terms(payload.get("local_names"), MAX_LLM_TERMS["local_names"])
    return {
        "aliases": _clean_terms(payload.get("aliases"), MAX_LLM_TERMS["aliases"]),
        "brands_subsidiaries": _clean_terms(payload.get("brands_subsidiaries"), MAX_LLM_TERMS["brands_subsidiaries"]),
        "key_people": _clean_terms(payload.get("key_people"), MAX_LLM_TERMS["key_people"]),
        "sector": sector if isinstance(sector, str) and sector in sector_keys else None,
        # Only keep text actually in Sinhala (U+0D80-0DFF) or Tamil (U+0B80-0BFF) script.
        "local_names": [n for n in local if re.search(r"[඀-෿஀-௿]", n)],
        "linkedin_slug": slug.strip().strip("/") if isinstance(slug, str) and re.fullmatch(r"[A-Za-z0-9\-_/]{2,80}", slug.strip()) else None,
    }


# --------------------------------------------------------------------------------------
# Generate / load / save
# --------------------------------------------------------------------------------------

def generate_profile(instrument: Instrument, *, use_llm: bool = True) -> KeywordProfile:
    """Build a profile from all three layers."""
    names = name_variants(instrument)
    provenance = ["catalogue"]
    sector = default_sector(instrument)
    aliases: list[str] = []
    people_brands: list[str] = []
    local_names: list[str] = []
    linkedin_url = None

    expansion = llm_expand(instrument, names) if use_llm else None
    if expansion:
        provenance.append("llm")
        aliases = expansion["aliases"]
        people_brands = [*expansion["brands_subsidiaries"], *expansion["key_people"]]
        local_names = expansion["local_names"]
        sector = expansion["sector"] or sector
        if expansion["linkedin_slug"]:
            slug = expansion["linkedin_slug"].removeprefix("company/")
            linkedin_url = f"https://www.linkedin.com/company/{slug}"

    sector_terms, macro_terms = template_terms(instrument, sector)
    provenance.append("template")

    ticker = instrument.symbol.split(".")[0].upper()
    if instrument.exchange is Exchange.BINANCE:
        ticker = (instrument.provider_id or ticker).upper()

    return KeywordProfile(
        symbol=instrument.symbol,
        exchange=instrument.exchange.value,
        name=instrument.name,
        sector=sector,
        company=_dedupe([*names, *aliases]),
        people_brands=_dedupe(people_brands),
        sector_terms=sector_terms,
        macro_terms=macro_terms,
        exclude=list(ambiguous_tickers().get(ticker, [])),
        local_names=local_names,
        linkedin_url=linkedin_url,
        source="auto",
        generated_at=datetime.now(timezone.utc),
        provenance=provenance,
    )


def _profile_path(symbol: str) -> Path:
    from data.cache import _safe_symbol

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    return PROFILE_DIR / f"{_safe_symbol(symbol)}.json"


def save_profile(profile: KeywordProfile) -> None:
    path = _profile_path(profile.symbol)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)


def _apply_overrides(profile: KeywordProfile) -> KeywordProfile:
    """Merge configs/keyword_overrides.yaml: lists listed there REPLACE the stored ones."""
    if not OVERRIDES_PATH.exists():
        return profile
    overrides = (yaml.safe_load(OVERRIDES_PATH.read_text(encoding="utf-8")) or {}).get(profile.symbol)
    if not overrides:
        return profile
    allowed = set(KeywordProfile.model_fields) - {"symbol", "exchange", "generated_at", "provenance"}
    updates = {k: v for k, v in overrides.items() if k in allowed}
    return profile.model_copy(update={**updates, "source": "reviewed"})


def load_profile(symbol: str) -> KeywordProfile | None:
    path = _profile_path(symbol)
    if not path.exists():
        return None
    try:
        profile = KeywordProfile(**json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:  # noqa: BLE001 - a corrupt profile is regenerated
        logger.warning("discarding corrupt keyword profile %s: %s", path, exc)
        return None
    return _apply_overrides(profile)


def ensure_profile(instrument: Instrument, *, use_llm: bool = True) -> KeywordProfile:
    """The stored profile, or a freshly generated one. Reviewed profiles are never replaced."""
    existing = load_profile(instrument.symbol)
    if existing is not None:
        return existing
    profile = generate_profile(instrument, use_llm=use_llm)
    save_profile(profile)
    return _apply_overrides(profile)


def update_profile(symbol: str, changes: dict[str, Any]) -> KeywordProfile:
    """Apply a reviewer's edits. The result is marked reviewed and kept across regenerations."""
    current = load_profile(symbol)
    if current is None:
        raise KeyError(f"no keyword profile for {symbol}; track the symbol first")
    editable = {"sector", "company", "people_brands", "sector_terms", "macro_terms", "exclude",
                "local_names", "linkedin_url"}
    unknown = set(changes) - editable
    if unknown:
        raise ValueError(f"not editable: {', '.join(sorted(unknown))}")
    cleaned = {k: (_dedupe(v) if isinstance(v, list) else v) for k, v in changes.items()}
    updated = current.model_copy(update={**cleaned, "source": "reviewed"})
    save_profile(updated)
    return updated
