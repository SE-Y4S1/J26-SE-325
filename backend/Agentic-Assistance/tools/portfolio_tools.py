import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool


DATA_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "mock_financial_data.json"
)

# Load trusted configuration from repository root .env if not already set in environment
for env_candidate in [
    Path(__file__).resolve().parents[1] / ".env",  # backend/Agentic-Assistance/.env
    Path(__file__).resolve().parents[3] / ".env",  # repository root .env
]:
    if env_candidate.is_file():
        load_dotenv(env_candidate, override=False)

PLATFORM_SERVICE_URL = os.getenv(
    "PORTFOLIO_SERVICE_URL",
    os.getenv("PLATFORM_SERVICE_URL", "http://localhost:8100")
).rstrip("/")

PORTFOLIO_TIMEOUT_SECONDS = float(os.getenv("PORTFOLIO_SERVICE_TIMEOUT", "1.5"))
DEFAULT_PORTFOLIO_ID = os.getenv("PORTFOLIO_ID", "1")

# Platform validation invariants (matching backend/Platform/auth/security.py)
PLATFORM_ISSUER = "j26-se-325-platform"
PLATFORM_ALGORITHM = "HS256"
EXAMPLE_SECRET = "change-me-to-a-long-random-string"
MIN_SECRET_LENGTH = 32


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
        numeric_sub = str(int(user_id)) if str(user_id).isdigit() else "1"
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


def query_live_portfolio_service(
    portfolio_identifier: str = DEFAULT_PORTFOLIO_ID,
    base_url: str = PLATFORM_SERVICE_URL,
    timeout: float = PORTFOLIO_TIMEOUT_SECONDS,
) -> dict:
    """
    Query the platform service for read-only portfolio holdings.
    """
    target_id = portfolio_identifier if str(portfolio_identifier).isdigit() else DEFAULT_PORTFOLIO_ID
    url = f"{base_url}/portfolios/{target_id}"

    headers = {"Accept": "application/json"}
    headers.update(get_auth_header(user_id=target_id))

    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status_code = getattr(response, "status", 200)
            raw_bytes = response.read()
            text = raw_bytes.decode("utf-8")
            try:
                data = json.loads(text)
                return {
                    "available": True,
                    "status_code": status_code,
                    "data": data,
                    "error": None,
                }
            except json.JSONDecodeError:
                return {
                    "available": True,
                    "status_code": status_code,
                    "data": None,
                    "error": "Malformed JSON returned by platform service",
                }
    except urllib.error.HTTPError as err:
        try:
            error_body = err.read().decode("utf-8")
            data = json.loads(error_body)
        except Exception:
            data = None
        return {
            "available": True,
            "status_code": err.code,
            "data": data,
            "error": f"HTTP {err.code}: {err.reason}",
        }
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as err:
        return {
            "available": False,
            "status_code": None,
            "data": None,
            "error": f"Service unavailable ({err})",
        }
    except Exception as err:
        return {
            "available": False,
            "status_code": None,
            "data": None,
            "error": f"Unexpected error querying portfolio service ({err})",
        }


@tool
def get_portfolio_analysis(user_id: str = "USER001") -> str:
    """
    Analyze the user's investment portfolio.

    Use this tool when the user asks about:
    portfolio performance, portfolio risk, diversification,
    asset allocation, liquidity, or investment recommendations.
    """

    mock_data = load_mock_data()
    mock_portfolio = mock_data.get("portfolio")
    mock_allocation = mock_portfolio.get("allocation", {}) if mock_portfolio else {}

    live_res = query_live_portfolio_service(portfolio_identifier=user_id)

    # 1. LIVE SERVICE RETURNED A VALID PORTFOLIO RECORD
    if live_res["available"] and live_res["status_code"] == 200 and isinstance(live_res["data"], dict):
        p = live_res["data"]
        p_id = p.get("id", "N/A")
        name = p.get("name", "Unnamed Portfolio")
        currency = p.get("base_currency", "USD")
        total_value = float(p.get("total_value", 0.0))
        holdings = p.get("holdings", [])

        holdings_lines = []
        for h in holdings:
            sym = h.get("symbol", "UNKNOWN")
            qty = float(h.get("quantity", 0.0))
            price = float(h.get("current_price", 0.0))
            val = qty * price
            wt = (val / total_value * 100.0) if total_value > 0 else 0.0
            cost = h.get("cost_basis")
            cost_str = f"Cost Basis: {currency} {float(cost):,.2f}; " if cost is not None else ""
            adv = h.get("avg_daily_volume")
            adv_str = f"ADV: {float(adv):,.0f}" if adv is not None else ""
            holdings_lines.append(
                f"- {sym}: {qty:,.2f} units @ {currency} {price:,.2f} = {currency} {val:,.2f} ({wt:.1f}% of portfolio); {cost_str}{adv_str}".strip()
            )

        holdings_text = "\n".join(holdings_lines) if holdings_lines else "- No holdings recorded"

        return f"""
PORTFOLIO ANALYSIS — LIVE PLATFORM RECORD

Data Source:
Live platform portfolio store ({PLATFORM_SERVICE_URL}).

Portfolio ID:
{p_id}

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
- Live holdings and asset valuations retrieved directly from the platform portfolio store.
- Total recorded market value: {currency} {total_value:,.2f}.
- Largest recorded positions: {', '.join(h.get('symbol', '') for h in holdings[:3]) or 'None'}.

METRIC NOTICE:
The platform portfolio service stores live asset holdings and valuation.
Optimization metrics (such as expected annual return and risk scores) require running Component 1's analytical models and are not stored in the base portfolio record.
"""

    # 2. FALLBACK HANDLING (NOT FOUND, UNAUTHENTICATED, TIMED OUT, OR OFFLINE)
    status_code = live_res.get("status_code")
    if status_code == 404:
        reason = f"Portfolio not found on platform service ({PLATFORM_SERVICE_URL})"
    elif status_code in (401, 403):
        reason = f"Platform service authentication required ({live_res.get('error')})"
    else:
        reason = live_res.get("error") or "Live platform service unreachable"

    if mock_portfolio:
        alloc_lines = "\n".join(f"{k}: {v}%" for k, v in mock_allocation.items())
        return f"""
PORTFOLIO ANALYSIS — FALLBACK DATA

Data Source:
Local mock baseline (live platform service at {PLATFORM_SERVICE_URL}: {reason}).

User ID:
{user_id}

Total Portfolio Value:
LKR {mock_portfolio.get("total_value_lkr", 0):,}

Expected Annual Return:
{mock_portfolio.get("expected_annual_return", 0)}%

Risk Score:
{mock_portfolio.get("risk_score", 0)}/100

Risk Level:
{mock_portfolio.get("risk_level", "Unknown")}

Diversification Score:
{mock_portfolio.get("diversification_score", 0)}/100

Liquidity:
{mock_portfolio.get("liquidity", "Unknown")}

CURRENT ALLOCATION (Mock Baseline):
{alloc_lines}

OBSERVATIONS:
- Technology exposure is relatively high.
- Portfolio diversification is moderate.
- Liquidity is currently high.
- Overall portfolio risk is moderate.

FALLBACK NOTICE:
This record is local mock baseline data because live portfolio data was unavailable ({reason}).
This is NOT a live lookup result. All metrics and allocations above are synthetic baseline estimates.
"""

    return f"""
PORTFOLIO ANALYSIS — SERVICE UNAVAILABLE

User ID:
{user_id}

STATUS:
The live platform portfolio service at {PLATFORM_SERVICE_URL} was unreachable ({reason}).
No fallback mock baseline is available.
"""