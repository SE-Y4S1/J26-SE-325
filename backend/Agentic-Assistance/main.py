from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.messages import HumanMessage

from agent import agent
from responsible_ai import run_responsible_ai_checks


app = FastAPI(
    title="FinAgent - Agentic Financial Assistant",
    description="Component 4 - Agentic LLM Assistant & Responsible AI",
    version="0.2.0"
)


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class ChatRequest(BaseModel):
    message: str


# --------------------------------------------------
# ROOT
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "project": "FinAgent",
        "component": "Agentic LLM Assistant & Responsible AI",
        "version": "0.2.0",
        "status": "running"
    }


# --------------------------------------------------
# CHAT
# --------------------------------------------------

@app.post("/assistant/chat")
def chat(request: ChatRequest):

    # ----------------------------------------------
    # Send user message to agent
    # ----------------------------------------------

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(content=request.message)
            ]
        }
    )

    messages = result["messages"]

    # ----------------------------------------------
    # Final answer
    # ----------------------------------------------

    final_message = messages[-1]

    response_text = final_message.content

    # ----------------------------------------------
    # Extract tools used
    # ----------------------------------------------

    tools_used = []

    for message in messages:

        # ToolMessage represents the result
        # returned by a tool
        if getattr(message, "type", None) == "tool":

            tool_name = getattr(message, "name", None)

            if tool_name and tool_name not in tools_used:
                tools_used.append(tool_name)

    # ----------------------------------------------
    # Extract tool evidence
    # ----------------------------------------------

    evidence = []

    for message in messages:

        if getattr(message, "type", None) == "tool":

            tool_name = getattr(message, "name", "unknown_tool")

            tool_output = getattr(message, "content", "")

            evidence.append(
                {
                    "tool": tool_name,
                    "result": tool_output
                }
            )

    # ----------------------------------------------
    # Responsible AI
    # ----------------------------------------------

    responsible_ai = run_responsible_ai_checks(
        request.message,
        response_text
    )

    # ----------------------------------------------
    # Return complete response
    # ----------------------------------------------

    return {

        "answer": response_text,

        "agent_execution": {

            "tools_used": tools_used,

            "number_of_tools": len(tools_used),

            "status": (
                "tools_executed"
                if tools_used
                else "no_tools_required"
            )
        },

        "evidence": evidence,

        "responsible_ai": responsible_ai
    }