import json
from pathlib import Path

from langchain_core.tools import tool


DATA_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "mock_financial_data.json"
)


def load_data():
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


@tool
def get_portfolio_analysis(user_id: str = "USER001") -> str:
    """
    Analyze the user's investment portfolio.

    Use this tool when the user asks about:
    portfolio performance, portfolio risk, diversification,
    asset allocation, liquidity, or investment recommendations.
    """

    data = load_data()

    portfolio = data["portfolio"]
    allocation = portfolio["allocation"]

    return f"""
PORTFOLIO ANALYSIS

User ID: {user_id}

Total Portfolio Value:
LKR {portfolio["total_value_lkr"]:,}

Expected Annual Return:
{portfolio["expected_annual_return"]}%

Risk Score:
{portfolio["risk_score"]}/100

Risk Level:
{portfolio["risk_level"]}

Diversification Score:
{portfolio["diversification_score"]}/100

Liquidity:
{portfolio["liquidity"]}

CURRENT ALLOCATION

Technology: {allocation["Technology"]}%
Banking: {allocation["Banking"]}%
Healthcare: {allocation["Healthcare"]}%
Government Bonds: {allocation["Government Bonds"]}%
Cash: {allocation["Cash"]}%

OBSERVATIONS

- Technology exposure is relatively high.
- Portfolio diversification is moderate.
- Liquidity is currently high.
- Overall portfolio risk is moderate.
"""