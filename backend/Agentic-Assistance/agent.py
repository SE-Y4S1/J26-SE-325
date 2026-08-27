from typing import Annotated

from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from tools import tools


# --------------------------------------------------
# STATE
# --------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# --------------------------------------------------
# LOCAL LLM
# --------------------------------------------------

llm = ChatOllama(
    model="llama3.2:3b",
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
TOOL USAGE RULES
--------------------------------------------------

You have access to these tools:

1. get_fraud_analysis
2. get_portfolio_analysis
3. get_blockchain_audit

IMPORTANT:

When a user asks about a blocked, suspicious,
flagged, or potentially fraudulent transaction,
you MUST call:

get_fraud_analysis

If the user does not provide a transaction ID,
use the default mock transaction:

TX1001

When a user asks about portfolio risk,
performance, diversification, allocation,
or liquidity, call:

get_portfolio_analysis

When a user asks about blockchain records,
audit history, verification, or recorded decisions,
call:

get_blockchain_audit

If a question requires multiple types of information,
you may call multiple tools.

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

    response = llm_with_tools.invoke(messages)

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