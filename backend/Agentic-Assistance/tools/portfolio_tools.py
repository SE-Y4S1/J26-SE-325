import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from preferences_context import get_current_preferences


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

PORTFOLIO_TIMEOUT_SECONDS = float(os.getenv("PORTFOLIO_SERVICE_TIMEOUT", "1.5"))
DEFAULT_PORTFOLIO_ID = os.getenv("PORTFOLIO_ID", "1")
DEFAULT_USER_ID = os.getenv("PLATFORM_USER_ID", os.getenv("DEFAULT_USER_ID", "1"))

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
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status_code = getattr(response, "status", 200)
            raw_bytes = response.read()
            text = raw_bytes.decode("utf-8")
            try:
                data = json.loads(text)
                return status_code, data, None
            except json.JSONDecodeError:
                return status_code, None, "Malformed JSON returned by platform service"
    except urllib.error.HTTPError as err:
        try:
            error_body = err.read().decode("utf-8")
            data = json.loads(error_body)
        except Exception:
            data = None
        return err.code, data, f"HTTP {err.code}: {err.reason}"
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as err:
        return None, None, f"Service unavailable ({err})"
    except Exception as err:
        return None, None, f"Unexpected error querying portfolio service ({err})"


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
    headers = {"Accept": "application/json"}
    headers.update(get_auth_header(user_id=resolved_user))

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

    # Always query GET /portfolios first to retrieve the user's registered portfolio list
    list_url = f"{base_url}/portfolios"
    status_code, data, err = _http_get(list_url, headers=headers, timeout=timeout)

    if status_code != 200:
        return {
            "available": status_code is not None,
            "status_code": status_code,
            "data": data,
            "resolved_user_id": resolved_user,
            "portfolio_id": target_p_id,
            "empty_list": False,
            "error": err or f"HTTP {status_code}",
        }

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


@tool
def get_portfolio_analysis(
    user_id: str = "1",
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
        try:
            total_value = float(total_val_raw) if total_val_raw is not None else 0.0
        except (ValueError, TypeError):
            total_value = 0.0

        raw_holdings = p.get("holdings")
        holdings = raw_holdings if isinstance(raw_holdings, list) else []

        if holdings:
            holdings_lines = []
            parsed_holdings = []
            calculated_total = 0.0
            for h in holdings:
                if not isinstance(h, dict):
                    continue
                sym = str(h.get("symbol", "UNKNOWN")).strip().upper()
                try:
                    qty = float(h.get("quantity") if h.get("quantity") is not None else 0.0)
                except (ValueError, TypeError):
                    qty = 0.0
                try:
                    price = float(h.get("current_price") if h.get("current_price") is not None else 0.0)
                except (ValueError, TypeError):
                    price = 0.0
                val = qty * price
                calculated_total += val

                cost = h.get("cost_basis")
                if cost is not None:
                    try:
                        cost_str = f"Cost Basis: {currency} {float(cost):,.2f}; "
                    except (ValueError, TypeError):
                        cost_str = ""
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

                parsed_holdings.append((sym, qty, price, val, cost_str, adv_str))

            if total_value <= 0.0 and calculated_total > 0.0:
                total_value = calculated_total

            for sym, qty, price, val, cost_str, adv_str in parsed_holdings:
                wt = (val / total_value * 100.0) if total_value > 0 else 0.0
                holdings_lines.append(
                    f"- {sym}: {qty:,.2f} units @ {currency} {price:,.2f} = {currency} {val:,.2f} ({wt:.1f}% of portfolio); {cost_str}{adv_str}".strip()
                )
            holdings_text = "\n".join(holdings_lines)
            observations_text = f"""- Live holdings and asset valuations retrieved directly from the platform portfolio store.
- Total recorded market value: {currency} {total_value:,.2f}.
- Largest recorded positions: {', '.join(h[0] for h in parsed_holdings[:3]) or 'None'}."""
            lookup_status = "Active live portfolio record retrieved from platform service."
        else:
            holdings_text = "- No holdings recorded in this portfolio. Live API confirms 0 active positions."
            observations_text = f"""- Active portfolio record exists on the platform, but currently contains 0 assets/holdings.
- Total recorded market value: {currency} 0.00."""
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
{currency} {total_value:,.2f}

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
    elif status_code in (401, 403):
        reason = f"Platform service authentication required ({live_res.get('error')})"
        lookup_status = f"Platform service authentication required ({live_res.get('error')})"
        holdings_disclosure = "Live holdings: Unverified (Authentication required)"
        holdings_note = f"Platform authentication required for user '{resolved_user}'"
    else:
        reason = live_res.get("error") or "Live platform service unreachable"
        lookup_status = f"Service unavailable ({live_res.get('error')})"
        holdings_disclosure = "Live holdings: Unverified (Service unreachable)"
        holdings_note = f"Platform service unreachable for user '{resolved_user}'"

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
