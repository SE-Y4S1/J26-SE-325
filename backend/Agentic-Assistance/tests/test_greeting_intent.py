import unittest
from unittest.mock import patch, MagicMock

from langchain_core.messages import HumanMessage, AIMessage

import agent
from agent import is_greeting_or_general, call_llm, SYSTEM_PROMPT


class TestGreetingIntentClassification(unittest.TestCase):
    """Regression tests for greeting and capability intent classification."""

    def test_pure_greetings_identified(self):
        pure_greetings = [
            "Hello",
            "Hello!",
            "Hi",
            "Hi there",
            "Hey",
            "Good morning",
            "Good afternoon",
            "Good evening",
            "Greetings",
        ]
        for greeting in pure_greetings:
            with self.subTest(greeting=greeting):
                self.assertTrue(
                    is_greeting_or_general(greeting),
                    f"Expected '{greeting}' to be recognized as greeting",
                )

    def test_general_capability_questions_identified(self):
        capability_queries = [
            "What can you help me with?",
            "Hello, what can you help me with?",
            "What can you do?",
            "What are your capabilities?",
            "How can you help me?",
            "Who are you?",
            "Tell me about yourself",
            "Explain what you can do",
            "Can you help me?",
            "Help me",
        ]
        for query in capability_queries:
            with self.subTest(query=query):
                self.assertTrue(
                    is_greeting_or_general(query),
                    f"Expected '{query}' to be recognized as general capability question",
                )

    def test_financial_questions_not_classified_as_greetings(self):
        financial_queries = [
            "Why was my transaction blocked?",
            "Why was TX1001 blocked?",
            "Check fraud score for TX1001",
            "Analyze transaction TX1001",
            "Can you check portfolio risk?",
            "What is my portfolio allocation?",
            "Check blockchain audit for TX1001",
            "Verify blockchain record",
            "What is the status of the smart contract ledger?",
            "Hello, why was TX1001 blocked?",
            "Hello! Can you check my portfolio holdings?",
            "Hi, verify the blockchain audit for my transaction.",
        ]
        for query in financial_queries:
            with self.subTest(query=query):
                self.assertFalse(
                    is_greeting_or_general(query),
                    f"Expected '{query}' NOT to be recognized as pure greeting",
                )


class TestAgentGreetingInvocation(unittest.TestCase):
    """Regression tests proving greetings do not invoke financial tools."""

    def test_greeting_routes_to_unbound_llm(self):
        """Pure greeting must route to llm.invoke (no tool binding) and not invoke llm_with_tools."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(
            content="Hello! I am FinAgent. I can help explain fraud decisions, analyze portfolio risk, and check blockchain audit records."
        )
        mock_binding = MagicMock()

        with patch.object(agent, "llm", mock_llm), patch.object(agent, "llm_with_tools", mock_binding):
            state = {
                "messages": [
                    HumanMessage(content="Hello, what can you help me with?")
                ]
            }

            result = call_llm(state)

            self.assertTrue(mock_llm.invoke.called, "Base LLM should be invoked for greeting")
            self.assertFalse(
                mock_binding.invoke.called,
                "Tool-bound LLM must NOT be invoked for general greetings",
            )

            response_msg = result["messages"][0]
            self.assertEqual(
                getattr(response_msg, "tool_calls", []),
                [],
                "Greeting response must contain 0 tool calls",
            )

    def test_financial_query_routes_to_tool_bound_llm(self):
        """Financial query must route to llm_with_tools to allow agentic tool calling."""
        mock_llm = MagicMock()
        mock_binding = MagicMock()
        mock_binding.invoke.return_value = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_fraud_analysis",
                    "args": {"transaction_id": "TX1001"},
                    "id": "call_123",
                }
            ],
        )

        with patch.object(agent, "llm", mock_llm), patch.object(agent, "llm_with_tools", mock_binding):
            state = {
                "messages": [
                    HumanMessage(content="Why was my transaction blocked?")
                ]
            }

            result = call_llm(state)

            self.assertTrue(
                mock_binding.invoke.called,
                "Tool-bound LLM should be invoked for financial questions",
            )
            self.assertFalse(
                mock_llm.invoke.called,
                "Base LLM without tools should not be invoked for financial questions",
            )

            response_msg = result["messages"][0]
            self.assertEqual(len(response_msg.tool_calls), 1)
            self.assertEqual(response_msg.tool_calls[0]["name"], "get_fraud_analysis")

    def test_system_prompt_greeting_and_minimal_tool_rules(self):
        """Ensure SYSTEM_PROMPT contains explicit prohibitions against calling tools for greetings."""
        self.assertIn("GREETINGS AND CAPABILITY INQUIRIES", SYSTEM_PROMPT)
        self.assertIn("DO NOT CALL ANY TOOLS", SYSTEM_PROMPT)
        self.assertIn("MINIMAL INVOCATION RULE", SYSTEM_PROMPT)
        self.assertIn("call ONLY:\n\nget_fraud_analysis", SYSTEM_PROMPT)
        self.assertIn("call ONLY:\n\nget_portfolio_analysis", SYSTEM_PROMPT)
        self.assertIn("call ONLY:\n\nget_blockchain_audit", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
