from .fraud_tools import get_fraud_analysis
from .portfolio_tools import get_portfolio_analysis
from .audit_tools import get_blockchain_audit


tools = [
    get_fraud_analysis,
    get_portfolio_analysis,
    get_blockchain_audit,
]