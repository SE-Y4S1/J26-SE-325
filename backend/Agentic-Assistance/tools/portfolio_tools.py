import contextvars
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from preferences_context import get_current_preferences

try:
    from timing_context import get_current_timing
except ImportError:
    try:
        import sys
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from timing_context import get_current_timing
    except Exception:
        get_current_timing = lambda: None


_request_portfolio_cache: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "_request_portfolio_cache", default=None
)


def init_request_portfolio_cache():
    return _request_portfolio_cache.set({})


def clear_request_portfolio_cache(token):
    if token is not None:
        try:
            _request_portfolio_cache.reset(token)
        except Exception:
            pass


DATA_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "mock_financial_data.json"
)

# Load trusted configuration from repository root .env if not already set in environment
env_candidates = [Path(__file__).resolve().parents[1] / ".env"]
if len(Path(__file__).resolve().parents) > 3:
    env_candidates.append(Path(__file__).resolve().parents[3] / ".env")
for env_candidate in env_candidates:
    if env_candidate.is_file():
        load_dotenv(env_candidate, override=False)

PLATFORM_SERVICE_URL = os.getenv(
    "PORTFOLIO_SERVICE_URL",
    os.getenv("PLATFORM_SERVICE_URL", "http://localhost:8100")
).rstrip("/")

COMPONENT1_SERVICE_URL = os.getenv(
    "PORTFOLIO_OPTIMIZATION_URL",
    os.getenv("COMPONENT1_SERVICE_URL", "http://localhost:8000")
).rstrip("/")

PORTFOLIO_TIMEOUT_SECONDS = float(os.getenv("PORTFOLIO_SERVICE_TIMEOUT", "1.5"))
DEFAULT_PORTFOLIO_ID = os.getenv("PORTFOLIO_ID", "1")
DEFAULT_USER_ID = os.getenv("PLATFORM_USER_ID", os.getenv("DEFAULT_USER_ID", "1"))


def query_component1_health(timeout: float = 1.0) -> dict:
    """
    Query Component 1 (Portfolio-Optimization service) health, model status, and available forecasters.
    Returns status dict without raising exceptions.
    """
    url = f"{COMPONENT1_SERVICE_URL}/health"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "FinTrust-Component4/1.0", "Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return {
                "reachable": True,
                "status": data.get("status", "ok"),
                "model_version": data.get("model_version", "unregistered"),
                "available_forecasters": data.get("available_forecasters", []),
                "kafka_connected": data.get("kafka_connected", False),
                "url": COMPONENT1_SERVICE_URL,
            }
    except Exception as exc:
        return {
            "reachable": False,
            "error": str(exc),
            "model_version": "unavailable",
            "available_forecasters": [],
            "kafka_connected": False,
            "url": COMPONENT1_SERVICE_URL,
        }


def query_live_market_quote(
    symbol: str,
    exchange: str | None = None,
    timeout: float = 3.0,
) -> dict:
    """
    Attempt to retrieve a verified live quote from Component 1's /market/quote endpoint.
    Returns status dict without raising exceptions:
    {
        "available": bool,
        "status_code": int | None,
        "data": dict | None,
        "error": str | None,
        "source": str,
    }
    """
    clean_sym = symbol.strip().upper()
    url = f"{COMPONENT1_SERVICE_URL}/market/quote"
    headers = {
        "User-Agent": "FinTrust-Component4/1.0",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    auth_header = get_auth_header("1")
    headers.update(auth_header)
    payload: dict = {"symbol": clean_sym}
    if exchange:
        payload["exchange"] = exchange
    try:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return {
                "available": True,
                "status_code": resp.status,
                "data": data,
                "error": None,
                "source": f"Component 1 Market Data Service ({COMPONENT1_SERVICE_URL})",
            }
    except urllib.error.HTTPError as exc:
        return {
            "available": False,
            "status_code": exc.code,
            "data": None,
            "error": f"HTTP {exc.code}: {exc.reason}",
            "source": f"Component 1 Market Data Service ({COMPONENT1_SERVICE_URL})",
        }
    except Exception as exc:
        return {
            "available": False,
            "status_code": None,
            "data": None,
            "error": str(exc),
            "source": f"Component 1 Market Data Service ({COMPONENT1_SERVICE_URL})",
        }


# Platform validation invariants (matching backend/Platform/auth/security.py)
PLATFORM_ISSUER = "j26-se-325-platform"
PLATFORM_ALGORITHM = "HS256"
EXAMPLE_SECRET = "change-me-to-a-long-random-string"
MIN_SECRET_LENGTH = 32

KNOWN_PLACEHOLDERS = {
    "your_user_id",
    "user001",
    "user_id",
    "your_id",
    "default",
    "placeholder",
    "none",
    "null",
}


def resolve_user_id(candidate: str | None) -> str:
    """
    Resolve user identifier to a valid numeric user ID string.
    If the candidate is missing, a known placeholder ('your_user_id', 'USER001'),
    or non-numeric, falls back to the configured default user ID.
    """
    if not candidate:
        return DEFAULT_USER_ID
    clean = str(candidate).strip()
    if clean.lower() in KNOWN_PLACEHOLDERS:
        return DEFAULT_USER_ID
    if clean.isdigit() and int(clean) > 0:
        return clean
    return DEFAULT_USER_ID


def validate_platform_secret(secret: str | None) -> bool:
    """
    Validate that the candidate JWT_SECRET strictly satisfies the platform's
    security invariants before minting tokens.
    """
    if not secret:
        return False
    stripped = secret.strip()
    if stripped == EXAMPLE_SECRET:
        return False
    if len(stripped) < MIN_SECRET_LENGTH:
        return False
    return True


def load_mock_data():
    if not DATA_FILE.exists():
        return {}
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def get_auth_header(user_id: str = "1") -> dict:
    """
    Resolve bearer authorization token for platform service.
    Supports direct token override (PORTFOLIO_AUTH_TOKEN) or dynamically minted token
    using the shared repository JWT_SECRET matching the platform's exact validation requirements.
    Never hardcodes or prints secrets.
    """
    # 1. Direct token override if explicitly provided
    direct_token = os.getenv("PORTFOLIO_AUTH_TOKEN")
    if direct_token and direct_token.strip():
        return {"Authorization": f"Bearer {direct_token.strip()}"}

    # 2. Mint token from trusted JWT_SECRET
    jwt_secret = os.getenv("JWT_SECRET")
    if not validate_platform_secret(jwt_secret):
        return {}

    try:
        import jwt

        # Platform requires subject to be a numeric string representing an integer user id
        numeric_sub = resolve_user_id(user_id)
        now = datetime.now(timezone.utc)
        payload = {
            "sub": numeric_sub,
            "iat": now,
            "exp": now + timedelta(hours=1),
            "iss": PLATFORM_ISSUER,
        }
        token = jwt.encode(payload, jwt_secret.strip(), algorithm=PLATFORM_ALGORITHM)
        return {"Authorization": f"Bearer {token}"}
    except Exception:
        return {}


def _http_get(url: str, headers: dict, timeout: float) -> tuple[int | None, object, str | None]:
    """
    Safely execute an HTTP GET request returning (status_code, parsed_json, error_message).
    """
    t_start = time.perf_counter()
    endpoint = urllib.parse.urlparse(url).path or url
    req = urllib.request.Request(url, headers=headers)
    status_code = None
    data = None
    err_msg = None
    is_timeout = False
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status_code = getattr(response, "status", 200)
            raw_bytes = response.read()
            text = raw_bytes.decode("utf-8")
            try:
                data = json.loads(text)
                return status_code, data, None
            except json.JSONDecodeError:
                err_msg = "Malformed JSON returned by platform service"
                return status_code, None, err_msg
    except urllib.error.HTTPError as err:
        status_code = err.code
        try:
            error_body = err.read().decode("utf-8")
            data = json.loads(error_body)
        except Exception:
            data = None
        err_msg = f"HTTP {err.code}: {err.reason}"
        return status_code, data, err_msg
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as err:
        is_timeout = isinstance(err, TimeoutError) or "timed out" in str(err).lower()
        err_msg = f"Service unavailable ({err})"
        return None, None, err_msg
    except Exception as err:
        err_msg = f"Unexpected error querying portfolio service ({err})"
        return None, None, err_msg
    finally:
        duration_ms = (time.perf_counter() - t_start) * 1000
        timing_ctx = get_current_timing()
        if timing_ctx is not None:
            timing_ctx.record_http(
                endpoint=endpoint,
                duration_ms=duration_ms,
                status_code=status_code,
                retries=0,
                is_timeout=is_timeout,
                error=err_msg,
            )


def query_live_portfolio_service(
    portfolio_identifier: str | None = None,
    user_id: str | None = None,
    portfolio_id: str | int | None = None,
    base_url: str = PLATFORM_SERVICE_URL,
    timeout: float = PORTFOLIO_TIMEOUT_SECONDS,
) -> dict:
    """
    Query the platform service for read-only portfolio holdings.

    1. Resolves authenticated user identity (defaulting invalid placeholders to DEFAULT_USER_ID).
    2. If explicit numeric portfolio_id is provided, queries GET /portfolios/{portfolio_id}
       and ensures ownership.
    3. Otherwise, queries GET /portfolios to retrieve the caller's portfolio list.
       - If empty, returns clear no-live-portfolio result.
       - If non-empty, selects the first portfolio ID and queries its details.
    """
    # Disambiguate legacy portfolio_identifier if passed positionally or by keyword
    if user_id is None and portfolio_identifier is not None:
        clean_id = str(portfolio_identifier).strip()
        if clean_id.lower() in KNOWN_PLACEHOLDERS or not clean_id.isdigit():
            user_id = clean_id
        elif portfolio_id is None:
            user_id = DEFAULT_USER_ID
            portfolio_id = clean_id

    resolved_user = resolve_user_id(user_id)
    target_p_id = str(portfolio_id).strip() if (portfolio_id is not None and str(portfolio_id).strip().isdigit()) else None

    # Check user personalization preference: if disabled, do not query platform service
    prefs = get_current_preferences()
    if not prefs.get("personalized_context", True):
        return {
            "available": False,
            "status_code": None,
            "data": None,
            "resolved_user_id": None,
            "portfolio_id": target_p_id,
            "empty_list": False,
            "error": "Personalized financial context is disabled by user preference",
        }

    # Request-scoped cache to prevent redundant duplicate queries within the same request
    req_cache = _request_portfolio_cache.get()
    cache_key = (resolved_user, target_p_id, base_url)
    if req_cache is not None and cache_key in req_cache:
        return req_cache[cache_key]

    headers = {"Accept": "application/json"}
    headers.update(get_auth_header(user_id=resolved_user))

    # Always query GET /portfolios first to retrieve the user's registered portfolio list
    list_url = f"{base_url}/portfolios"
    status_code, data, err = _http_get(list_url, headers=headers, timeout=timeout)

    if status_code != 200:
        res = {
            "available": status_code is not None,
            "status_code": status_code,
            "data": data,
            "resolved_user_id": resolved_user,
            "portfolio_id": target_p_id,
            "empty_list": False,
            "error": err or f"HTTP {status_code}",
        }
        if req_cache is not None:
            req_cache[cache_key] = res
        return res

    def _fetch():
        # If platform returns a list
        if isinstance(data, list):
            if len(data) == 0:
                return {
                    "available": True,
                    "status_code": 200,
                    "data": None,
                    "resolved_user_id": resolved_user,
                    "portfolio_id": target_p_id,
                    "empty_list": True,
                    "error": None,
                }

            # Portfolios exist. Select target or first available portfolio.
            if target_p_id is not None:
                # Check if target_p_id appears in the authenticated user's portfolio list
                matching_p = next((p for p in data if isinstance(p, dict) and str(p.get("id")) == target_p_id), None)
                if matching_p is None:
                    return {
                        "available": True,
                        "status_code": 404,
                        "data": None,
                        "resolved_user_id": resolved_user,
                        "portfolio_id": target_p_id,
                        "empty_list": False,
                        "error": f"Portfolio {target_p_id} not owned by user {resolved_user}",
                    }
                selected_id = target_p_id
            else:
                selected_p = data[0]
                selected_id = selected_p.get("id") if isinstance(selected_p, dict) else None
                if selected_id is None:
                    return {
                        "available": True,
                        "status_code": 200,
                        "data": None,
                        "resolved_user_id": resolved_user,
                        "portfolio_id": None,
                        "empty_list": True,
                        "error": "Unexpected portfolio list item format",
                    }

            detail_url = f"{base_url}/portfolios/{selected_id}"
            detail_status, detail_data, detail_err = _http_get(detail_url, headers=headers, timeout=timeout)
            if detail_status == 200 and isinstance(detail_data, dict):
                return {
                    "available": True,
                    "status_code": 200,
                    "data": detail_data,
                    "resolved_user_id": resolved_user,
                    "portfolio_id": str(selected_id),
                    "empty_list": False,
                    "error": None,
                }
            return {
                "available": detail_status is not None,
                "status_code": detail_status,
                "data": detail_data,
                "resolved_user_id": resolved_user,
                "portfolio_id": str(selected_id),
                "empty_list": False,
                "error": detail_err,
            }

        # Safe handling: if endpoint returned a portfolio dict directly (e.g. single-response mock)
        if isinstance(data, dict):
            p_id = str(data.get("id", target_p_id or "1"))
            return {
                "available": True,
                "status_code": 200,
                "data": data,
                "resolved_user_id": resolved_user,
                "portfolio_id": p_id,
                "empty_list": False,
                "error": None,
            }

        return {
            "available": True,
            "status_code": 200,
            "data": None,
            "resolved_user_id": resolved_user,
            "portfolio_id": target_p_id,
            "empty_list": False,
            "error": err or "Unexpected response format from portfolio endpoint",
        }

    res = _fetch()
    if req_cache is not None:
        req_cache[cache_key] = res
    return res


@tool
def get_portfolio_analysis(
    user_id: str | None = "1",
    portfolio_id: str | None = None,
) -> str:
    """
    Analyze the user's investment portfolio.

    Parameters:
    - user_id: Authenticated user ID (default is '1').
    - portfolio_id: Optional specific portfolio ID. Only provide this if the user
      explicitly asked for a specific portfolio number in their message.
      Do not provide a default portfolio ID for general portfolio queries.
    """
    t_tool_start = time.perf_counter()
    tool_error = None
    try:
        return _execute_portfolio_analysis(user_id=user_id, portfolio_id=portfolio_id)
    except Exception as exc:
        tool_error = str(exc)
        raise
    finally:
        tool_duration_ms = (time.perf_counter() - t_tool_start) * 1000
        timing_ctx = get_current_timing()
        if timing_ctx is not None:
            timing_ctx.record_tool(
                tool_name="get_portfolio_analysis",
                duration_ms=tool_duration_ms,
                success=(tool_error is None),
                retries=0,
                error=tool_error,
            )


def _execute_portfolio_analysis(
    user_id: str | None = "1",
    portfolio_id: str | None = None,
) -> str:
    prefs = get_current_preferences()
    if not prefs.get("personalized_context", True):
        return """
PORTFOLIO ANALYSIS — PERSONALIZED CONTEXT DISABLED

Data Source:
None (Personalized financial context is disabled by user preference).

STATUS:
Personalized financial context is disabled by user preference.
Account-specific portfolio lookup was not performed, and no personalized data was retrieved.

NOTICE:
Under your current Responsible AI settings, access to account-specific investment portfolios and personal asset holdings is disabled. To view your portfolio holdings or allocation, enable 'Personalized financial context' in Responsible AI & Safety settings.
No anonymous or fallback portfolio data has been substituted.
Live holdings: Unverified (Personalized context disabled).
"""

    mock_data = load_mock_data()
    mock_portfolio = mock_data.get("portfolio")
    mock_allocation = mock_portfolio.get("allocation", {}) if mock_portfolio else {}

    user_id_param = str(user_id) if user_id is not None else DEFAULT_USER_ID
    resolved_user = resolve_user_id(user_id_param)

    live_res = query_live_portfolio_service(user_id=user_id_param, portfolio_id=portfolio_id)

    # 1. LIVE SERVICE RETURNED A VALID PORTFOLIO RECORD
    if live_res["available"] and live_res["status_code"] == 200 and isinstance(live_res.get("data"), dict):
        p = live_res["data"]
        p_id = live_res.get("portfolio_id") or p.get("id", "N/A")
        name = p.get("name", "Unnamed Portfolio")
        currency = p.get("base_currency", "USD")
        total_val_raw = p.get("total_value")
        total_value_known = False
        try:
            if total_val_raw is not None:
                total_value = float(total_val_raw)
                total_value_known = True
            else:
                total_value = None
        except (ValueError, TypeError):
            total_value = None

        raw_holdings = p.get("holdings")
        holdings = raw_holdings if isinstance(raw_holdings, list) else []

        has_partial_fields = False
        if holdings:
            holdings_lines = []
            parsed_holdings = []
            calculated_total = 0.0
            all_prices_available = True

            for h in holdings:
                if not isinstance(h, dict):
                    continue
                sym = str(h.get("symbol", "UNKNOWN")).strip().upper()

                # Parse quantity
                qty_raw = h.get("quantity")
                try:
                    qty = float(qty_raw) if qty_raw is not None else None
                except (ValueError, TypeError):
                    qty = None

                # Parse current price
                price_raw = h.get("current_price")
                try:
                    price = float(price_raw) if price_raw is not None else None
                except (ValueError, TypeError):
                    price = None

                if qty is not None and price is not None:
                    val = qty * price
                    calculated_total += val
                    val_str = f"{currency} {val:,.2f}"
                else:
                    val = None
                    all_prices_available = False
                    has_partial_fields = True
                    val_str = "Value: Unavailable"

                cost = h.get("cost_basis")
                if cost is not None:
                    try:
                        cost_str = f"Cost Basis: {currency} {float(cost):,.2f}; "
                    except (ValueError, TypeError):
                        cost_str = "Cost Basis: Unavailable; "
                        has_partial_fields = True
                else:
                    cost_str = ""

                adv = h.get("avg_daily_volume")
                if adv is not None:
                    try:
                        adv_str = f"ADV: {float(adv):,.0f}"
                    except (ValueError, TypeError):
                        adv_str = ""
                else:
                    adv_str = ""

                parsed_holdings.append((sym, qty, price, val, val_str, cost_str, adv_str))

            if not total_value_known and all_prices_available and calculated_total > 0.0:
                total_value = calculated_total
                total_value_known = True

            for sym, qty, price, val, val_str, cost_str, adv_str in parsed_holdings:
                qty_display = f"{qty:,.2f} units" if qty is not None else "Quantity: Unavailable"
                price_display = f"@{currency} {price:,.2f}" if price is not None else "@ Price: Unavailable"

                if val is not None and total_value_known and total_value and total_value > 0:
                    wt = (val / total_value * 100.0)
                    wt_str = f"({wt:.1f}% of portfolio)"
                else:
                    wt_str = "(Weight: Unavailable)"

                cost_adv_part = f"; {cost_str}{adv_str}".rstrip("; ") if (cost_str or adv_str) else ""
                holdings_lines.append(
                    f"- {sym}: {qty_display} {price_display} = {val_str} {wt_str}{cost_adv_part}".strip()
                )
            holdings_text = "\n".join(holdings_lines)

            if total_value_known and total_value is not None:
                total_val_display = f"{currency} {total_value:,.2f}"
            else:
                total_val_display = "Unavailable / Not recorded"
                has_partial_fields = True

            partial_note = "\n- Notice: Some fields (such as valuation, price, or cost basis) are not recorded on the platform and are explicitly marked as Unavailable." if has_partial_fields else ""
            observations_text = f"""- Live holdings and asset valuations retrieved directly from the platform portfolio store.
- Total recorded market value: {total_val_display}.
- Largest recorded positions: {', '.join(h[0] for h in parsed_holdings[:3]) or 'None'}.{partial_note}"""
            lookup_status = "Active live portfolio record retrieved (partial fields marked)" if has_partial_fields else "Active live portfolio record retrieved from platform service."
        else:
            total_val_display = f"{currency} 0.00" if (total_value_known and total_value == 0.0) else (f"{currency} {total_value:,.2f}" if total_value_known and total_value is not None else f"{currency} 0.00")
            holdings_text = "- No holdings recorded in this portfolio. Live API confirms 0 active positions."
            observations_text = f"""- Active portfolio record exists on the platform, but currently contains 0 assets/holdings.
- Total recorded market value: {total_val_display}."""
            lookup_status = "Active live portfolio record retrieved (0 holdings recorded)."

        return f"""
PORTFOLIO ANALYSIS — LIVE PLATFORM RECORD

Data Source:
Live platform portfolio store ({PLATFORM_SERVICE_URL}).

Resolved User ID:
{resolved_user}

Portfolio ID:
{p_id}

Live Lookup Status:
{lookup_status}

Portfolio Name:
{name}

Base Currency:
{currency}

Total Portfolio Value:
{total_val_display}

Holding Count:
{len(holdings)}

RECORDED HOLDINGS:
{holdings_text}

OBSERVATIONS:
{observations_text}

METRIC NOTICE:
The platform portfolio service stores live asset holdings and valuation.
Optimization metrics (such as expected annual return and risk scores) require running Component 1's analytical models and are not stored in the base portfolio record.
"""

    # 2. FALLBACK HANDLING (NOT FOUND, EMPTY LIST, UNAUTHENTICATED, TIMED OUT, OR OFFLINE)
    status_code = live_res.get("status_code")
    empty_list = live_res.get("empty_list", False)

    if empty_list:
        reason = f"No portfolios registered for user on platform service ({PLATFORM_SERVICE_URL})"
        lookup_status = "Empty portfolio list (User has no portfolios on platform)"
        holdings_disclosure = "Live holdings: Unverified (No live portfolio registered on platform)"
        holdings_note = f"No live portfolio was retrieved for user '{resolved_user}' from the platform service (account has no registered portfolios)"
        account_status_sentence = "The platform service confirmed that no portfolios or active holdings are registered for this account."
    elif status_code == 404:
        err_msg = live_res.get("error")
        if err_msg and not err_msg.startswith("HTTP 404"):
            reason = err_msg
            lookup_status = err_msg
        else:
            reason = f"Portfolio not found on platform service ({PLATFORM_SERVICE_URL})"
            lookup_status = "Portfolio not found on platform service"
        holdings_disclosure = "Live holdings: Unverified (No live portfolio record retrieved)"
        holdings_note = f"No live portfolio was retrieved for user '{resolved_user}' from the platform service"
        account_status_sentence = "The platform service confirmed that no matching portfolio was found for this user."
    elif status_code in (401, 403):
        reason = f"Platform service authentication required ({live_res.get('error')})"
        lookup_status = f"Platform service authentication required ({live_res.get('error')})"
        holdings_disclosure = "Live holdings: Unverified (Authentication required)"
        holdings_note = f"Platform authentication required for user '{resolved_user}'"
        account_status_sentence = "Platform authentication credentials are required to access live portfolio records."
    else:
        reason = live_res.get("error") or "Live platform service unreachable"
        lookup_status = f"Service unavailable ({live_res.get('error')})"
        holdings_disclosure = "Live holdings: Unverified (Platform service unreachable or timed out)"
        holdings_note = f"Platform service unreachable for user '{resolved_user}'"
        account_status_sentence = "This does not indicate an empty portfolio; live holdings could not be retrieved due to a platform service connection failure or timeout."

    if mock_portfolio:
        alloc_lines = "\n".join(f"- {k}: {v}% (Mock Baseline)" for k, v in mock_allocation.items())
        return f"""
PORTFOLIO ANALYSIS — FALLBACK DATA (NO LIVE RECORD RETRIEVED)

Data Source:
Local mock baseline (live platform service at {PLATFORM_SERVICE_URL}: {reason}).

STATUS:
NO LIVE PORTFOLIO RECORD RETRIEVED for user ID '{resolved_user}'.
Platform service note: {reason}.

Live Lookup Status:
{lookup_status}

Resolved User Identity:
{resolved_user} (Requested: '{user_id_param}')

FALLBACK NOTICE:
{holdings_note} ({reason}).
{account_status_sentence}
This record is NOT the user's actual portfolio. All metrics, valuations, and allocations below are synthetic baseline estimates for demonstration only. The user does NOT have a verified balance of LKR {mock_portfolio.get("total_value_lkr", 0):,} or active live holdings on record.

User ID:
{resolved_user}

ILLUSTRATIVE MOCK BASELINE ESTIMATES (NOT ACTUAL USER PORTFOLIO):
- Illustrative Baseline Valuation: LKR {mock_portfolio.get("total_value_lkr", 0):,} (Mock Baseline Demonstration Only)
- Illustrative Expected Annual Return: {mock_portfolio.get("expected_annual_return", 0)}% (Synthetic Estimate)
- Illustrative Risk Score: {mock_portfolio.get("risk_score", 0)}/100 (Synthetic Estimate)
- Illustrative Risk Level: {mock_portfolio.get("risk_level", "Unknown")} (Synthetic Estimate)
- Illustrative Diversification Score: {mock_portfolio.get("diversification_score", 0)}/100 (Synthetic Estimate)
- Illustrative Liquidity: {mock_portfolio.get("liquidity", "Unknown")} (Synthetic Estimate)

ILLUSTRATIVE CURRENT ALLOCATION (Mock Baseline Example Only):
{alloc_lines}

ILLUSTRATIVE OBSERVATIONS (Sample Demonstration Only):
- Technology exposure is relatively high in this mock baseline example.
- Portfolio diversification is moderate in this mock baseline example.
- Liquidity is rated high in this mock baseline example.
- Overall portfolio risk is rated moderate in this mock baseline example.

IMPORTANT DISCLAIMER:
Do not describe these synthetic metrics, asset allocations, or observations as the user's actual portfolio.
{holdings_disclosure}.
Live market value: Unverified (No live portfolio record).
"""

    return f"""
PORTFOLIO ANALYSIS — SERVICE UNAVAILABLE

User ID:
{resolved_user}

Resolved User Identity:
{resolved_user} (Requested: '{user_id_param}')

Live Lookup Status:
{lookup_status}

STATUS:
The live platform portfolio service at {PLATFORM_SERVICE_URL} was unreachable ({reason}).
No fallback mock baseline is available.
"""
