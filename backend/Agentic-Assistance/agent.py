import os
import re
from typing import Annotated

from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from tools import tools


# --------------------------------------------------
# INTENT ROUTING
# --------------------------------------------------

def is_greeting_or_general(text: str) -> bool:
    """
    Determine if a user query is a greeting or general capability inquiry
    without specific financial analysis requests.
    """
    clean = text.strip().lower()

    # Specific financial keywords that require tools
    financial_keywords = [
        "tx", "transaction", "portfolio", "holding", "holdings", "audit", "blockchain",
        "fraud", "blocked", "flagged", "suspicious", "lkr", "usd", "eur",
        "transfer", "balance", "stock", "asset", "crypto", "token", "score",
        "shap", "ledger", "smart contract", "why was", "verify", "allocation",
        "diversification", "liquidity", "invest", "investment", "risk score"
    ]

    # If any specific financial keyword is present, it is not a pure greeting
    if any(re.search(rf"\b{re.escape(kw)}\b", clean) for kw in financial_keywords):
        return False

    # Check for greeting or capability patterns
    greeting_patterns = [
        r"^(?:hello|hi|hey|greetings|good\s+(?:morning|afternoon|evening|day))(?:\s*[,.!?]|\s*$|\s+there)",
        r"(?:what\s+can\s+you\s+(?:do|help(?:\s+me)?\s+(?:with|on)))",
        r"(?:who\s+are\s+you)",
        r"(?:how\s+(?:can|do)\s+you\s+help)",
        r"(?:what\s+(?:are\s+your|services|features)\s+(?:capabilities|features|services)?)",
        r"(?:tell\s+me\s+about\s+(?:yourself|what\s+you\s+can\s+do))",
        r"(?:explain\s+what\s+you\s+can\s+(?:do|help))",
        r"^(?:help|help\s+me|can\s+you\s+help\s+me)(?:\s*[,.!?]|\s*$)",
    ]

    return any(re.search(pattern, clean) for pattern in greeting_patterns)


def is_arithmetic_query(text: str) -> bool:
    """
    Determine if a user query is a simple arithmetic calculation or math question.
    """
    clean = text.strip().lower()

    # If query explicitly targets specific financial entities, it is not a pure arithmetic question.
    if re.search(r"\b(?:tx\d*|transaction|portfolio|fraud|blockchain|audit)\b", clean):
        return False

    arithmetic_patterns = [
        # Match math expressions like "what is 2 + 2", "what is 2 +2", "calculate 15 * 4", "100 / 5", "50 - 25"
        r"(?:what\s+(?:is|\'s)|calculate|solve|evaluate|compute|how\s+much\s+is)?\s*\(?\s*-?\d+(?:\.\d+)?\s*\)?\s*(?:[\+\-\*\/\^%x×÷]|plus|minus|times|divided\s+by|multiplied\s+by)\s*\(?\s*-?\d+",
        # Pure math equation e.g. "2+2", "2 + 2 = ?"
        r"^\s*\(?\s*-?\d+(?:\.\d+)?\s*\)?\s*(?:[\+\-\*\/\^%x×÷]|plus|minus|times|divided\s+by|multiplied\s+by)\s*\(?\s*-?\d+(?:\.\d+)?\s*\)?(?:\s*=\s*\??)?\s*$",
        # Word-based math queries e.g. "what is 5 times 6", "10 plus 20"
        r"\b(?:what\s+(?:is|\'s)|calculate|solve|compute)\s+\d+(?:\.\d+)?\s+(?:plus|minus|times|divided\s+by|multiplied\s+by)\s+\d+",
        # Functions like "sum of 4 and 5", "product of 3 and 7", "square root of 144"
        r"\b(?:sum|difference|product|quotient|square\s+root)\s+of\s+\d+",
    ]

    return any(re.search(pattern, clean) for pattern in arithmetic_patterns)


def has_financial_intent(text: str) -> bool:
    """
    Determine whether a user query has genuine financial intent requiring financial tools
    (fraud analysis, portfolio analysis, or blockchain audit verification).
    """
    clean = text.strip().lower()

    # Specific financial and domain-specific terms
    financial_keywords = [
        # Explicit transaction and entity references
        r"\btx\d*\b",
        r"\btransactions?\b",
        r"\btransfers?\b",
        r"\bpayments?\b",
        r"\bchargebacks?\b",
        # Fraud, scams, and risk metrics
        r"\bfraud(?:ulent)?\b",
        r"\bscams?\b",
        r"\brisk\s*score\b",
        r"\bfraud\s*score\b",
        r"\bfraud\s*risk\b",
        r"\bshap\b",
        r"\bsuspicious\s*(?:activity|transaction|charge|transfer|payment|login)\b",
        # Blocked / flagged / declined in financial context
        r"\b(?:blocked|flagged|declined|rejected)\s+(?:transaction|tx\d*|transfer|payment|card|account|withdrawal|charge)\b",
        r"\b(?:transaction|tx\d*|transfer|payment|card|account|withdrawal|charge)\s+(?:was\s+|is\s+|got\s+)?(?:blocked|flagged|declined|rejected)\b",
        r"\bwhy\s+was\s+(?:my\s+|the\s+)?(?:transaction|tx\d*|transfer|payment|card|account|charge)\s+(?:blocked|flagged|declined|rejected)\b",
        # Portfolio, assets, holdings
        r"\bportfolios?\b",
        r"\bholdings?\b",
        r"\basset\s*allocations?\b",
        r"\ballocations?\b",
        r"\bdiversif\w+\b",
        r"\bliquidity(?:\s*risk)?\b",
        r"\brebalanc\w+\b",
        r"\binvestments?\b",
        # Blockchain, audit, ledger
        r"\bblockchains?\b",
        r"\baudits?\b",
        r"\bledgers?\b",
        r"\bsmart\s*contracts?\b",
        r"\bblock\s*numbers?\b",
        r"\btx_hash\b",
        r"\b(crypto|cryptocurrency|token|tokens|lkr|usd|eur|gbp)\b",
    ]

    return any(re.search(pattern, clean) for pattern in financial_keywords)


def should_route_to_financial_tools(text: str) -> bool:
    """
    Determine if a user query requires financial tools (fraud, portfolio, blockchain audit).

    Returns False for:
    - Greetings and general capability inquiries
    - Simple arithmetic and math questions
    - General knowledge and non-financial questions
    - Ambiguous queries lacking financial context

    Returns True only when the query exhibits clear financial intent
    matching one of our financial domains (fraud/transactions, portfolio, blockchain audit).
    """
    clean = text.strip().lower()

    # Greetings and capability inquiries never invoke financial tools
    if is_greeting_or_general(clean):
        return False

    # Simple arithmetic never invokes financial tools
    if is_arithmetic_query(clean):
        return False

    # Only route to tools if there is clear financial intent
    return has_financial_intent(clean)


# --------------------------------------------------
# STATE
# --------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# --------------------------------------------------
# LOCAL LLM
# --------------------------------------------------

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", os.getenv("OLLAMA_HOST", "http://localhost:11434"))

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
    base_url=OLLAMA_URL,
    temperature=0.1,
)


# Allow the LLM to use our tools
llm_with_tools = llm.bind_tools(tools)


# --------------------------------------------------
# SYSTEM PROMPT
# --------------------------------------------------

SYSTEM_PROMPT = """
You are FinAgent, an explainable and responsible
financial AI assistant.

You are an AGENTIC financial decision-support system.

Your job is to understand the user's request,
decide what information is required, select the
appropriate tool, execute the tool, and then explain
the result to the user.

You are NOT a financial advisor.

--------------------------------------------------
GREETINGS AND CAPABILITY INQUIRIES
--------------------------------------------------

If the user sends a greeting (such as "Hello", "Hi", "Good morning")
or asks about your general capabilities (such as "What can you help me with?",
"What can you do?", "How do you work?"):
- DO NOT CALL ANY TOOLS.
- Do NOT invoke get_fraud_analysis, get_portfolio_analysis, or get_blockchain_audit.
- Provide a normal, friendly greeting and describe your capabilities:
  1. Explaining fraud decisions, recorded risk scores, and contributing factors.
  2. Analyzing portfolio allocation, diversification, and liquidity risk.
  3. Verifying blockchain audit records and transaction ledger status.

--------------------------------------------------
NON-FINANCIAL AND GENERAL QUESTIONS
--------------------------------------------------

If the user asks a question unrelated to finance, such as simple arithmetic
(e.g., "what is 2 + 2", "what is 2 +2", "calculate 15 * 4"), general knowledge (e.g.,
"what is the capital of France?"), definitions, or casual conversation:
- DO NOT CALL ANY TOOLS.
- Do NOT invoke get_fraud_analysis, get_portfolio_analysis, or get_blockchain_audit.
- Provide a direct, correct, and concise answer without inventing transactions,
  mock records, or unnecessary financial disclaimers.

--------------------------------------------------
AMBIGUOUS OR UNCLEAR QUESTIONS
--------------------------------------------------

If the user's question is ambiguous, lacks context, or does not clearly specify
a transaction, portfolio, or audit record to inspect (e.g., "Can you check it?",
"Why did this occur?", "What happened?"):
- DO NOT invent a transaction or make assumptions.
- DO NOT call any financial tools.
- Politely ask the user to clarify their request or provide the transaction ID
  (e.g., TX1001), portfolio query, or blockchain record they would like to analyze.

--------------------------------------------------
TOOL USAGE RULES
--------------------------------------------------

You have access to these tools:

1. get_fraud_analysis
2. get_portfolio_analysis
3. get_blockchain_audit

MINIMAL INVOCATION RULE:
For specific financial questions, invoke ONLY the tools needed to answer the user's intent:

When a user asks about a blocked, suspicious,
flagged, or potentially fraudulent transaction,
call ONLY:

get_fraud_analysis

If the user does not provide a transaction ID,
use the default mock transaction:

TX1001

When a user asks about portfolio risk,
performance, diversification, allocation,
or liquidity, call ONLY:

get_portfolio_analysis

If the user does not specify a user ID or portfolio ID,
call get_portfolio_analysis with the default user ID:

1

Do not invent placeholder strings such as "your_user_id" or "USER001".

When a user asks about blockchain records,
audit history, verification, or recorded decisions,
call ONLY:

get_blockchain_audit

Call multiple tools ONLY if the question explicitly asks for multiple types of information (e.g. asking for both fraud analysis and blockchain audit verification).

NEVER tell the user to call a tool themselves.

The agent must execute the tool and use the returned
data to answer the user.

--------------------------------------------------
REASONING
--------------------------------------------------

Follow this process:

1. Understand the user's question.
2. Determine which tool or tools are required.
3. Call the appropriate tool.
4. Read the returned evidence.
5. Explain the evidence in simple language.
6. Clearly distinguish factual results from recommendations.
7. Never invent financial information.

SHAP EVIDENCE ACCURACY — CRITICAL:

- Treat the tool output as the source of truth.
- Preserve the exact mapping between each feature name and its SHAP value.
- Never assign a SHAP value to a different feature.
- Do not reorder, swap, or guess feature-value pairs.
- If the tool returns amount_thousands = 0.193289,
  unusual_location = 0.153450, and new_device = 0.130761,
  preserve those exact mappings.
- A positive SHAP value means the feature increases
  the model output for the explained class in this prototype.
- Do not claim that a SHAP contribution is a percentage
  or a probability.
- Clearly state that the SHAP explanation comes from
  the local synthetic prototype model, not necessarily
  the production fraud model.
- If the evidence is missing or ambiguous, say so rather
  than inventing a value.

  EVIDENCE ACCURACY AND RESPONSIBLE AI RULES:

1. Clearly distinguish recorded mock data, local prototype model outputs, and independently verified production evidence.

2. Never describe mock transaction or portfolio records as production data. Explicitly identify them as mock data whenever applicable.

3. Local prototype fraud probabilities and SHAP values explain only the local synthetic prototype model. Never present them as the production model's probability, as proof of fraud, or as verified explanations of the recorded fraud score.

4. When explaining fraud analysis, report the recorded fraud score and recorded contributing factors separately from local prototype predictions and SHAP contributions.

5. Never claim that a blockchain record has been independently verified merely because its status field says VERIFIED. State what the available tool output actually confirms and what verification remains unproven.

6. Never invent causes, financial misconduct, transaction details, model behavior, or verification results that are not supported by the available evidence.

7. For portfolio analysis:
   - When live portfolio data is returned: explain the retrieved live holdings, quantities, and valuations as recorded in the platform store. Disclose that optimization metrics require Component 1 analytical modeling.
   - When live portfolio data is UNAVAILABLE (portfolio not found, unauthenticated, or service unreachable):
     a. You MUST BEGIN your response with a clear notice that no live portfolio was retrieved for the user from the platform service (e.g. "Notice: No live portfolio record was retrieved for your account from the platform service...").
     b. NEVER describe fallback portfolio metrics, valuations (such as LKR 1,000,000), or allocations as the user's actual portfolio, balance, or investments.
     c. Do not claim that the user has zero holdings unless the live API response explicitly confirms an active portfolio with zero holdings. If no live portfolio record was retrieved (or the portfolio list is empty), state that live holdings are unverified.
     d. Either omit the synthetic metrics entirely, or clearly label every mentioned metric, return estimate, risk score, diversification score, liquidity rating, and allocation as an illustrative mock baseline example for demonstration only.

8. Explain uncertainty and limitations in clear language. If the evidence is insufficient to support a conclusion, say so explicitly.

9. Do not claim to have called a tool unless its result is present in the current interaction.

10. Use the term LLM-generated explanation when describing the language model's response. Do not claim that every explanation is grounded in SHAP; mention SHAP only when relevant SHAP evidence was actually returned.

11. Do not infer how a model was trained, which dataset it used, whether two models share training data, or whether a recorded score came from a production model unless the tool evidence explicitly establishes those facts.

12. For TX1001, describe 0.92 only as the recorded fraud score in the available mock transaction record. Describe 1.0000 only as the local synthetic Random Forest prototype's predicted fraud probability. Do not imply that either value is independently validated or directly comparable to the other.

13. A prototype probability of 1.0000 is not proof of fraud or proof of real-world certainty. Explain that it is an output of the local synthetic prototype, not a verified real-world fraud probability.

14. When evidence is missing, explicitly say "not established by the available evidence" instead of filling the gap with assumptions.

15. Do not speculate about production fraud decisions, training datasets, model versions, or business logic beyond what is explicitly provided in the available evidence.

16. Use grounded, precise language. Avoid vague claims like "production insights" or "validated by experts." Instead, say "This explanation is based on the available mock transaction data and the local synthetic prototype model." 

STRICT FACTUALITY OVERRIDE:

These rules take priority over examples or general explanations elsewhere in this prompt.

17. RECORDED SCORE: For TX1001, state that 0.92 is the recorded fraud score in the available mock transaction record. Do not call it a probability, percentage, or model confidence unless documented evidence establishes that interpretation.

18. PROTOTYPE PROBABILITY: Report 1.0000 as the numerical probability output returned by the local synthetic Random Forest prototype. Do not translate it into "100% chance that the transaction is fraudulent" or imply certainty about real-world fraud. Explicitly state that it is a prototype output, not a verified real-world probability.

19. TRAINING DATA — NO YES/NO GUESSING: Never answer that two models share or do not share training data unless explicit evidence establishes the relationship. The absence of evidence does not establish either answer. For TX1001, the only permitted conclusion is: "Whether the models share training data is not established by the available evidence." Do not contradict this statement elsewhere in the same response.

20. MODEL IDENTITY: The available transaction record does not establish which model generated the recorded score. Do not describe it as a production-model output unless the evidence explicitly identifies the producing model.

21. MOCK DATA: Identify TX1001 and its recorded fields as mock transaction data. The prototype's device and location inputs were inferred from mock reason labels. Do not imply that these are independently verified real-world observations.

22. REQUIRED FINAL CHECK: Before responding, check every statement about probabilities, model identity, training data, and verification against the returned tool evidence. Remove any unsupported claim.

23. WHEN UNCERTAIN: State what is known, what is unknown, and why. Do not fill gaps with plausible-sounding assumptions.

24. PORTFOLIO FALLBACK: When platform portfolio lookup returns fallback mock data or indicates no live record was retrieved:
    - Begin your response immediately with a clear notice stating that no live portfolio record was retrieved.
    - Never state or imply that the user holds LKR 1,000,000, 40% equity, or any real balance.
    - Qualify every single synthetic metric as an illustrative mock example.
    - State that the user has zero live holdings on record.
--------------------------------------------------
RESPONSIBLE FINANCE
--------------------------------------------------

Never guarantee profits or returns.

Do not make autonomous financial transactions.

Important financial actions require user confirmation.

Protect sensitive financial information.

When possible, explain WHY a result occurred.

For example, if a transaction is blocked,
explain the specific factors that contributed
to the fraud score.

--------------------------------------------------
EXAMPLE
--------------------------------------------------

User:
"Why was my transaction blocked?"

Correct behavior:

1. Call get_fraud_analysis with TX1001.
2. Read the fraud score and contributing factors.
3. Explain that the transaction was blocked because
   of the factors returned by the tool.

Incorrect behavior:

"I don't have information about your transaction.
You can use the fraud analysis tool."

The assistant must call the tool itself.
"""


# --------------------------------------------------
# LLM NODE
# --------------------------------------------------

def call_llm(state: AgentState):

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state["messages"]
    ]

    latest_msg = state["messages"][-1] if state.get("messages") else None

    # If the latest message is a tool response, continue tool-enabled workflow
    # to allow multi-tool calls or final explanation generation.
    if getattr(latest_msg, "type", None) == "tool":
        response = llm_with_tools.invoke(messages)
        return {
            "messages": [response]
        }

    latest_content = getattr(latest_msg, "content", "") if latest_msg else ""
    if isinstance(latest_content, list):
        latest_text = " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in latest_content
        )
    else:
        latest_text = str(latest_content)

    # Route to financial tools ONLY when there is clear financial intent.
    # Non-financial questions (including arithmetic and general knowledge),
    # greetings, and ambiguous queries route to base LLM without tool binding.
    if should_route_to_financial_tools(latest_text):
        response = llm_with_tools.invoke(messages)
    else:
        response = llm.invoke(messages)

    return {
        "messages": [response]
    }


# --------------------------------------------------
# TOOL NODE
# --------------------------------------------------

tool_node = ToolNode(tools)


# --------------------------------------------------
# LANGGRAPH WORKFLOW
# --------------------------------------------------

workflow = StateGraph(AgentState)

workflow.add_node("llm", call_llm)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "llm")

workflow.add_conditional_edges(
    "llm",
    tools_condition
)

workflow.add_edge("tools", "llm")


# --------------------------------------------------
# COMPILE
# --------------------------------------------------

agent = workflow.compile()