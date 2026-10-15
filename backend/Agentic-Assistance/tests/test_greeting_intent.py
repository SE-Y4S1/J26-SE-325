import unittest
from unittest.mock import patch, MagicMock

from langchain_core.messages import HumanMessage, AIMessage

import agent
from agent import (
    is_greeting_or_general,
    is_arithmetic_query,
    has_financial_intent,
    should_route_to_financial_tools,
    call_llm,
    SYSTEM_PROMPT,
)


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


class TestArithmeticIntentClassification(unittest.TestCase):
    """Regression tests for arithmetic and math question classification."""

    def test_arithmetic_queries_identified(self):
        arithmetic_queries = [
            "what is 2 + 2",
            "what is 2 +2",
            "what is 2+2",
            "what's 2 + 2",
            "2 + 2",
            "2+2",
            "calculate 15 * 4",
            "calculate 100 / 5",
            "how much is 10 + 20",
            "what is 5 times 8",
            "what is the sum of 10 and 20",
            "square root of 144",
            "50 - 25",
            "100 / 4",
        ]
        for query in arithmetic_queries:
            with self.subTest(query=query):
                self.assertTrue(
                    is_arithmetic_query(query),
                    f"Expected '{query}' to be recognized as arithmetic query",
                )

    def test_financial_questions_not_classified_as_arithmetic(self):
        financial_queries = [
            "Why was my transaction blocked?",
            "Why was TX1001 blocked?",
            "Check fraud score for TX1001",
            "Analyze transaction TX1001",
            "Can you check portfolio risk?",
            "What is my portfolio allocation?",
            "Check blockchain audit for TX1001",
            "Verify blockchain record",
        ]
        for query in financial_queries:
            with self.subTest(query=query):
                self.assertFalse(
                    is_arithmetic_query(query),
                    f"Expected '{query}' NOT to be classified as arithmetic query",
                )


class TestFinancialIntentClassification(unittest.TestCase):
    """Regression tests for financial vs non-financial intent classification."""

    def test_financial_intent_detected_for_financial_queries(self):
        financial_queries = [
            # Fraud and transaction queries
            "Why was my transaction blocked?",
            "Why was TX1001 blocked?",
            "Check fraud score for TX1001",
            "Analyze transaction TX1001",
            "Why was this charge flagged as suspicious?",
            # Portfolio queries
            "Can you check portfolio risk?",
            "What is my portfolio allocation?",
            "What are my current holdings?",
            "How diversified is my portfolio?",
            "Check liquidity risk of my portfolio",
            # Blockchain audit queries
            "Check blockchain audit for TX1001",
            "Verify blockchain record",
            "What is the status of the smart contract ledger?",
            "Check the block number for TX1001",
        ]
        for query in financial_queries:
            with self.subTest(query=query):
                self.assertTrue(
                    has_financial_intent(query),
                    f"Expected '{query}' to be recognized as having financial intent",
                )

    def test_non_financial_queries_do_not_have_financial_intent(self):
        non_financial_queries = [
            # Arithmetic
            "what is 2 + 2",
            "what is 2 +2",
            "calculate 15 * 4",
            # General knowledge
            "What is the capital of France?",
            "Who was Albert Einstein?",
            "Tell me a joke",
            "How does photosynthesis work?",
            "What is the speed of light?",
            "Can you write a poem?",
            # Ambiguous or non-financial questions
            "Can you check it?",
            "Why did this occur?",
            "What happened?",
            "Why was the road blocked?",
        ]
        for query in non_financial_queries:
            with self.subTest(query=query):
                self.assertFalse(
                    has_financial_intent(query),
                    f"Expected '{query}' NOT to have financial intent",
                )


class TestShouldRouteToFinancialTools(unittest.TestCase):
    """Regression tests for should_route_to_financial_tools gate."""

    def test_non_financial_and_greetings_do_not_route_to_tools(self):
        non_tool_queries = [
            "what is 2 + 2",
            "what is 2 +2",
            "calculate 15 * 4",
            "What is the capital of France?",
            "Hello, what can you help me with?",
            "Hi there",
            "Good morning",
            "Who are you?",
            "Tell me a joke",
            "Can you check it?",
            "Why did this occur?",
            "What happened?",
        ]
        for query in non_tool_queries:
            with self.subTest(query=query):
                self.assertFalse(
                    should_route_to_financial_tools(query),
                    f"Expected '{query}' NOT to route to financial tools",
                )

    def test_financial_queries_route_to_tools(self):
        tool_queries = [
            "Why was my transaction blocked?",
            "Check fraud score for TX1001",
            "What is my portfolio allocation?",
            "Check blockchain audit for TX1001",
            "Verify blockchain record",
            "Analyze transaction TX1001",
        ]
        for query in tool_queries:
            with self.subTest(query=query):
                self.assertTrue(
                    should_route_to_financial_tools(query),
                    f"Expected '{query}' to route to financial tools",
                )


class TestAgentInvocationAndToolRouting(unittest.TestCase):
    """Regression tests verifying LLM routing, tool selection, and tool counts."""

    def test_greeting_routes_to_unbound_llm(self):
        """Pure greeting must route to llm.invoke (no tool binding) with 0 tool calls."""
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

    def test_arithmetic_routes_to_unbound_llm_zero_tools(self):
        """Simple arithmetic (e.g. 'what is 2 + 2') must route to llm.invoke with 0 tool calls."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="2 + 2 = 4")
        mock_binding = MagicMock()

        with patch.object(agent, "llm", mock_llm), patch.object(agent, "llm_with_tools", mock_binding):
            for query in ["what is 2 + 2", "what is 2 +2", "calculate 15 * 4"]:
                mock_llm.reset_mock()
                mock_binding.reset_mock()

                state = {
                    "messages": [
                        HumanMessage(content=query)
                    ]
                }

                result = call_llm(state)

                self.assertTrue(
                    mock_llm.invoke.called,
                    f"Base LLM without tools should be invoked for arithmetic query '{query}'",
                )
                self.assertFalse(
                    mock_binding.invoke.called,
                    f"Tool-bound LLM must NOT be invoked for arithmetic query '{query}'",
                )

                response_msg = result["messages"][0]
                self.assertEqual(
                    getattr(response_msg, "tool_calls", []),
                    [],
                    f"Arithmetic query '{query}' must have 0 tool calls",
                )

    def test_general_knowledge_routes_to_unbound_llm_zero_tools(self):
        """General knowledge question must route to llm.invoke with 0 tool calls."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="The capital of France is Paris.")
        mock_binding = MagicMock()

        with patch.object(agent, "llm", mock_llm), patch.object(agent, "llm_with_tools", mock_binding):
            state = {
                "messages": [
                    HumanMessage(content="What is the capital of France?")
                ]
            }

            result = call_llm(state)

            self.assertTrue(
                mock_llm.invoke.called,
                "Base LLM without tools should be invoked for general knowledge",
            )
            self.assertFalse(
                mock_binding.invoke.called,
                "Tool-bound LLM must NOT be invoked for general knowledge",
            )

            response_msg = result["messages"][0]
            self.assertEqual(
                getattr(response_msg, "tool_calls", []),
                [],
                "General knowledge question must contain 0 tool calls",
            )

    def test_ambiguous_question_routes_to_unbound_llm_zero_tools(self):
        """Ambiguous non-financial question must route to llm.invoke to ask clarification without tools."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(
            content="Could you please clarify what you would like me to check?"
        )
        mock_binding = MagicMock()

        with patch.object(agent, "llm", mock_llm), patch.object(agent, "llm_with_tools", mock_binding):
            state = {
                "messages": [
                    HumanMessage(content="Can you check it?")
                ]
            }

            result = call_llm(state)

            self.assertTrue(
                mock_llm.invoke.called,
                "Base LLM should be invoked for ambiguous questions to ask clarification",
            )
            self.assertFalse(
                mock_binding.invoke.called,
                "Tool-bound LLM must NOT be invoked for ambiguous questions lacking financial context",
            )

            response_msg = result["messages"][0]
            self.assertEqual(
                getattr(response_msg, "tool_calls", []),
                [],
                "Ambiguous query must contain 0 tool calls",
            )

    def test_fraud_query_routes_to_fraud_tool(self):
        """Questions about blocked transactions or fraud risk should use get_fraud_analysis."""
        mock_llm = MagicMock()
        mock_binding = MagicMock()
        mock_binding.invoke.return_value = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_fraud_analysis",
                    "args": {"transaction_id": "TX1001"},
                    "id": "call_fraud_1",
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
                "Tool-bound LLM should be invoked for fraud questions",
            )
            self.assertFalse(mock_llm.invoke.called)

            response_msg = result["messages"][0]
            self.assertEqual(len(response_msg.tool_calls), 1)
            self.assertEqual(response_msg.tool_calls[0]["name"], "get_fraud_analysis")

    def test_audit_query_routes_to_audit_tool(self):
        """Blockchain ledger/audit verification questions should use get_blockchain_audit."""
        mock_llm = MagicMock()
        mock_binding = MagicMock()
        mock_binding.invoke.return_value = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_blockchain_audit",
                    "args": {"transaction_id": "TX1001"},
                    "id": "call_audit_1",
                }
            ],
        )

        with patch.object(agent, "llm", mock_llm), patch.object(agent, "llm_with_tools", mock_binding):
            state = {
                "messages": [
                    HumanMessage(content="Check blockchain audit for TX1001")
                ]
            }

            result = call_llm(state)

            self.assertTrue(
                mock_binding.invoke.called,
                "Tool-bound LLM should be invoked for blockchain audit questions",
            )
            self.assertFalse(mock_llm.invoke.called)

            response_msg = result["messages"][0]
            self.assertEqual(len(response_msg.tool_calls), 1)
            self.assertEqual(response_msg.tool_calls[0]["name"], "get_blockchain_audit")

    def test_portfolio_query_routes_to_portfolio_tool(self):
        """Portfolio allocation/risk questions should use get_portfolio_analysis."""
        mock_llm = MagicMock()
        mock_binding = MagicMock()
        mock_binding.invoke.return_value = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_portfolio_analysis",
                    "args": {},
                    "id": "call_port_1",
                }
            ],
        )

        with patch.object(agent, "llm", mock_llm), patch.object(agent, "llm_with_tools", mock_binding):
            state = {
                "messages": [
                    HumanMessage(content="What is my portfolio allocation?")
                ]
            }

            result = call_llm(state)

            self.assertTrue(
                mock_binding.invoke.called,
                "Tool-bound LLM should be invoked for portfolio questions",
            )
            self.assertFalse(mock_llm.invoke.called)

            response_msg = result["messages"][0]
            self.assertEqual(len(response_msg.tool_calls), 1)
            self.assertEqual(response_msg.tool_calls[0]["name"], "get_portfolio_analysis")

    def test_multi_analysis_query_routes_to_multiple_tools(self):
        """Explicit multi-analysis questions should invoke only the requested tools."""
        mock_llm = MagicMock()
        mock_binding = MagicMock()
        mock_binding.invoke.return_value = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_fraud_analysis",
                    "args": {"transaction_id": "TX1001"},
                    "id": "call_fraud_2",
                },
                {
                    "name": "get_blockchain_audit",
                    "args": {"transaction_id": "TX1001"},
                    "id": "call_audit_2",
                },
            ],
        )

        with patch.object(agent, "llm", mock_llm), patch.object(agent, "llm_with_tools", mock_binding):
            state = {
                "messages": [
                    HumanMessage(
                        content="Analyze TX1001 using both fraud analysis and blockchain audit."
                    )
                ]
            }

            result = call_llm(state)

            self.assertTrue(mock_binding.invoke.called)
            response_msg = result["messages"][0]
            self.assertEqual(len(response_msg.tool_calls), 2)
            tool_names = [tc["name"] for tc in response_msg.tool_calls]
            self.assertIn("get_fraud_analysis", tool_names)
            self.assertIn("get_blockchain_audit", tool_names)

    def test_system_prompt_rules(self):
        """Ensure SYSTEM_PROMPT contains explicit prohibitions and rules."""
        self.assertIn("GREETINGS AND CAPABILITY INQUIRIES", SYSTEM_PROMPT)
        self.assertIn("NON-FINANCIAL AND GENERAL QUESTIONS", SYSTEM_PROMPT)
        self.assertIn("AMBIGUOUS OR UNCLEAR QUESTIONS", SYSTEM_PROMPT)
        self.assertIn("DO NOT CALL ANY TOOLS", SYSTEM_PROMPT)
        self.assertIn("MINIMAL INVOCATION RULE", SYSTEM_PROMPT)
        self.assertIn("call ONLY:\n\nget_fraud_analysis", SYSTEM_PROMPT)
        self.assertIn("call ONLY:\n\nget_portfolio_analysis", SYSTEM_PROMPT)
        self.assertIn("call ONLY:\n\nget_blockchain_audit", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
