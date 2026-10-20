import unittest
from unittest.mock import patch, MagicMock

# pyrefly: ignore[missing-import]
from langchain_core.messages import HumanMessage, AIMessage

import agent
from agent import (
    is_greeting_or_general,
    is_thanks_or_appreciation,
    is_farewell,
    is_conversational_acknowledgment,
    get_deterministic_fast_path,
    classify_domain_gate,
    GREETING_FAST_RESPONSE,
    THANKS_FAST_RESPONSE,
    FAREWELL_FAST_RESPONSE,
    CONVERSATIONAL_FAST_RESPONSE,
    OUT_OF_DOMAIN_RESPONSE,
    AMBIGUOUS_CLARIFICATION_RESPONSE,
    is_arithmetic_query,
    has_financial_intent,
    should_route_to_financial_tools,
    call_llm,
    SYSTEM_PROMPT,
    GENERAL_SYSTEM_PROMPT,
)
from tools.portfolio_tools import get_portfolio_analysis
import main
from main import ChatRequest


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


class TestFastPathIntentClassification(unittest.TestCase):
    """Regression tests for thanks, farewell, conversational acknowledgment, and fast path routing."""

    def test_thanks_identified(self):
        thanks_queries = [
            "thanks",
            "thank you",
            "thank you very much",
            "thank you so much",
            "thanks a lot",
            "thx",
            "many thanks",
            "appreciate it",
            "much appreciated",
        ]
        for query in thanks_queries:
            with self.subTest(query=query):
                self.assertTrue(
                    is_thanks_or_appreciation(query),
                    f"Expected '{query}' to be recognized as thanks",
                )

    def test_farewells_identified(self):
        farewell_queries = [
            "bye",
            "goodbye",
            "see you",
            "have a good day",
            "have a nice day",
            "take care",
            "cya",
        ]
        for query in farewell_queries:
            with self.subTest(query=query):
                self.assertTrue(
                    is_farewell(query),
                    f"Expected '{query}' to be recognized as farewell",
                )

    def test_conversational_acknowledgments_identified(self):
        ack_queries = [
            "ok",
            "okay",
            "cool",
            "great",
            "awesome",
            "perfect",
            "good",
            "sounds good",
            "understood",
            "got it",
            "sure",
            "no problem",
        ]
        for query in ack_queries:
            with self.subTest(query=query):
                self.assertTrue(
                    is_conversational_acknowledgment(query),
                    f"Expected '{query}' to be recognized as conversational acknowledgment",
                )

    def test_get_deterministic_fast_path_returns_expected_content(self):
        # Greetings - general capability
        for q in ["Hello", "What can you help me with?", "Who are you?"]:
            self.assertEqual(get_deterministic_fast_path(q), GREETING_FAST_RESPONSE)

        # Contextual greetings specified in Section 4
        self.assertEqual(
            get_deterministic_fast_path("Good morning"),
            "Good morning! What would you like to explore about the stock market or your portfolio?",
        )
        self.assertEqual(
            get_deterministic_fast_path("Hi"),
            "Hi! I'm FinTrust Agent. How can I help you with your investments, portfolio, or financial risk today?",
        )
        self.assertEqual(
            get_deterministic_fast_path("Hi there"),
            "Hi! I'm FinTrust Agent. How can I help you with your investments, portfolio, or financial risk today?",
        )

        # Thanks
        for q in ["Thanks", "Thank you so much", "Appreciate it", "Many thanks"]:
            self.assertEqual(get_deterministic_fast_path(q), THANKS_FAST_RESPONSE)

        # Farewells
        for q in ["Bye", "Goodbye", "Have a nice day", "Take care"]:
            self.assertEqual(get_deterministic_fast_path(q), FAREWELL_FAST_RESPONSE)

        # Acknowledgments
        for q in ["Ok", "Got it", "Cool", "Great", "Sounds good"]:
            self.assertEqual(get_deterministic_fast_path(q), CONVERSATIONAL_FAST_RESPONSE)

    def test_queries_with_financial_intent_never_fast_pathed(self):
        financial_queries = [
            "Hello, why was TX1001 blocked?",
            "Hi, what is my portfolio allocation?",
            "Good morning, check blockchain audit for TX1001",
            "Thanks, can you check fraud for TX1001?",
            "Ok, analyze transaction TX1001",
            "Great, what is my portfolio risk score?",
        ]
        for query in financial_queries:
            with self.subTest(query=query):
                self.assertIsNone(
                    get_deterministic_fast_path(query),
                    f"Financial query '{query}' must NOT be fast-pathed",
                )

    def test_out_of_domain_and_ambiguous_queries_fast_pathed_with_appropriate_responses(self):
        ood_queries = [
            "what is 2 + 2",
            "calculate 15 * 4",
            "What is the capital of France?",
            "Who was Albert Einstein?",
        ]
        for query in ood_queries:
            with self.subTest(query=query):
                self.assertEqual(
                    get_deterministic_fast_path(query),
                    OUT_OF_DOMAIN_RESPONSE,
                    f"Out-of-domain query '{query}' must receive polite domain refusal directly",
                )

        ambiguous_queries = [
            "Can you check it?",
            "What happened?",
            "Why did this occur?",
        ]
        for query in ambiguous_queries:
            with self.subTest(query=query):
                self.assertEqual(
                    get_deterministic_fast_path(query),
                    AMBIGUOUS_CLARIFICATION_RESPONSE,
                    f"Ambiguous query '{query}' must receive clarifying response directly",
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

    def test_greeting_fast_path_zero_tools_zero_llm_calls(self):
        """Pure greetings must resolve deterministically with 0 LLM calls and 0 tool calls."""
        mock_llm = MagicMock()
        mock_binding = MagicMock()

        with patch.object(agent, "llm", mock_llm), patch.object(agent, "llm_with_tools", mock_binding):
            for greeting in ["Hello", "Hi there", "Good morning", "What can you help me with?", "Who are you?"]:
                mock_llm.reset_mock()
                mock_binding.reset_mock()

                state = {
                    "messages": [
                        HumanMessage(content=greeting)
                    ]
                }

                result = call_llm(state)

                # Must NOT call Ollama LLM or financial tool binding
                self.assertFalse(mock_llm.invoke.called, f"Ollama LLM must NOT be called for greeting '{greeting}'")
                self.assertFalse(mock_binding.invoke.called, f"Tools must NOT be bound or called for greeting '{greeting}'")

                response_msg = result["messages"][0]
                self.assertEqual(
                    getattr(response_msg, "tool_calls", []),
                    [],
                    f"Greeting '{greeting}' must contain 0 tool calls",
                )
                self.assertTrue(any(marker in response_msg.content for marker in ["FinTrust Agent", "portfolio", "investments"]))
                self.assertNotIn("SYSTEM PROMPT", response_msg.content)

    def test_thanks_and_conversational_fast_path_zero_tools_zero_llm_calls(self):
        """Thanks, farewells, and conversational acknowledgments must resolve with 0 LLM calls and 0 tools."""
        mock_llm = MagicMock()
        mock_binding = MagicMock()

        with patch.object(agent, "llm", mock_llm), patch.object(agent, "llm_with_tools", mock_binding):
            for conv_msg in ["Thanks", "Thank you very much!", "Appreciate it", "Goodbye", "Have a nice day", "ok", "got it"]:
                mock_llm.reset_mock()
                mock_binding.reset_mock()

                state = {
                    "messages": [
                        HumanMessage(content=conv_msg)
                    ]
                }

                result = call_llm(state)

                self.assertFalse(mock_llm.invoke.called, f"Ollama LLM must NOT be called for conversational msg '{conv_msg}'")
                self.assertFalse(mock_binding.invoke.called, f"Tools must NOT be called for conversational msg '{conv_msg}'")

                response_msg = result["messages"][0]
                self.assertEqual(getattr(response_msg, "tool_calls", []), [])
                self.assertTrue(len(response_msg.content) > 0)

    def test_greeting_with_financial_intent_routes_to_tools(self):
        """Greetings that also ask financial questions must NOT be fast-pathed; they must route to tools."""
        mock_llm = MagicMock()
        mock_binding = MagicMock()
        mock_binding.invoke.return_value = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_fraud_analysis",
                    "args": {"transaction_id": "TX1001"},
                    "id": "call_fraud_mixed",
                }
            ],
        )

        with patch.object(agent, "llm", mock_llm), patch.object(agent, "llm_with_tools", mock_binding):
            state = {
                "messages": [
                    HumanMessage(content="Hello, why was TX1001 blocked?")
                ]
            }

            result = call_llm(state)

            self.assertTrue(mock_binding.invoke.called, "Tool binding must be invoked when financial intent is present")
            self.assertFalse(mock_llm.invoke.called)
            response_msg = result["messages"][0]
            self.assertEqual(len(response_msg.tool_calls), 1)
            self.assertEqual(response_msg.tool_calls[0]["name"], "get_fraud_analysis")

    def test_arithmetic_routes_to_out_of_domain_refusal_zero_tools(self):
        """Pure non-financial arithmetic (e.g. 'what is 2 + 2') must return polite domain refusal directly with 0 tool calls."""
        mock_llm = MagicMock()
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
                self.assertEqual(response_msg.content, OUT_OF_DOMAIN_RESPONSE)

    def test_general_knowledge_routes_to_out_of_domain_refusal_zero_tools(self):
        """General knowledge question must return polite domain refusal with 0 tool calls."""
        mock_llm = MagicMock()
        mock_binding = MagicMock()

        with patch.object(agent, "llm", mock_llm), patch.object(agent, "llm_with_tools", mock_binding):
            state = {
                "messages": [
                    HumanMessage(content="What is the capital of France?")
                ]
            }

            result = call_llm(state)

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
            self.assertEqual(response_msg.content, OUT_OF_DOMAIN_RESPONSE)

    def test_ambiguous_question_routes_to_clarification_zero_tools(self):
        """Ambiguous non-financial question must return clarifying question directly without tools."""
        mock_llm = MagicMock()
        mock_binding = MagicMock()

        with patch.object(agent, "llm", mock_llm), patch.object(agent, "llm_with_tools", mock_binding):
            state = {
                "messages": [
                    HumanMessage(content="Can you check it?")
                ]
            }

            result = call_llm(state)

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
            self.assertEqual(response_msg.content, AMBIGUOUS_CLARIFICATION_RESPONSE)

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
                    HumanMessage(content="Why was transaction TX1001 blocked?")
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

    def test_unresolved_fraud_query_asks_clarification(self):
        """'Why was my transaction blocked?' must ask for clarification instead of defaulting to TX1001."""
        state = {
            "messages": [
                HumanMessage(content="Why was my transaction blocked?")
            ]
        }
        result = call_llm(state)
        response_msg = result["messages"][0]
        self.assertFalse(hasattr(response_msg, "tool_calls") and bool(response_msg.tool_calls))
        self.assertIn("provide the transaction ID", response_msg.content)

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
        self.assertIn("25. NEVER EXPOSE INTERNAL DETAILS", SYSTEM_PROMPT)

    def test_portfolio_fallback_mock_data_clearly_labeled(self):
        """When platform service returns fallback mock data, it must be clearly labeled and not actual user data."""
        with patch("tools.portfolio_tools.query_live_portfolio_service", return_value={
            "available": True,
            "status_code": 404,
            "data": None,
            "resolved_user_id": "1",
            "portfolio_id": None,
            "empty_list": False,
            "error": "Portfolio not found on platform service",
        }):
            output = get_portfolio_analysis.invoke({"user_id": "1"})

            # Must disclose that no live portfolio was retrieved
            self.assertIn("NO LIVE PORTFOLIO RECORD RETRIEVED", output)
            self.assertIn("FALLBACK NOTICE:", output)

            # Must clearly state that this is NOT the user's actual portfolio
            self.assertIn("NOT the user's actual portfolio", output)
            self.assertIn("synthetic baseline estimates for demonstration only", output)

            # Must explicitly label all synthetic metrics as mock baseline
            self.assertIn("Mock Baseline Demonstration Only", output)
            self.assertIn("Synthetic Estimate", output)
            self.assertIn("Mock Baseline Example Only", output)

    def test_main_chat_fast_path_greetings_zero_tools(self):
        """Calling /assistant/chat with greetings must resolve via fast path with 0 tools and no LLM call."""
        with patch.object(agent, "agent") as mock_agent_workflow:
            req = ChatRequest(message="Hello! What can you help me with?")
            res = main.chat(req)

            # Agent workflow must not be called
            self.assertFalse(mock_agent_workflow.invoke.called, "agent.invoke must not be called for greetings")

            # Must return 0 tools
            self.assertEqual(res["agent_execution"]["number_of_tools"], 0)
            self.assertEqual(res["agent_execution"]["tools_used"], [])
            self.assertEqual(res["agent_execution"]["status"], "no_tools_required")
            self.assertEqual(res["evidence"], [])

            # Must have complete responsible AI checks
            self.assertIn("privacy", res["responsible_ai"])
            self.assertIn("safety", res["responsible_ai"])
            self.assertIn("transparency", res["responsible_ai"])

            # Must be friendly fast path answer
            self.assertIn("FinTrust Agent", res["answer"])
            self.assertEqual(res["answer"], GREETING_FAST_RESPONSE)

    def test_main_chat_fast_path_out_of_domain_zero_tools(self):
        """Calling /assistant/chat with out-of-domain query must return polite refusal with 0 tools directly."""
        with patch.object(agent, "agent") as mock_agent_workflow:
            for query in ["What is the capital of France?", "Tell me a joke", "Write a python function"]:
                res = main.chat(ChatRequest(message=query))
                self.assertFalse(mock_agent_workflow.invoke.called, "agent.invoke must not be called for out-of-domain")
                self.assertEqual(res["agent_execution"]["number_of_tools"], 0)
                self.assertEqual(res["agent_execution"]["tools_used"], [])
                self.assertEqual(res["answer"], OUT_OF_DOMAIN_RESPONSE)

    def test_main_chat_fast_path_ambiguous_clarification_zero_tools(self):
        """Calling /assistant/chat with ambiguous query must return clarifying question with 0 tools."""
        with patch.object(agent, "agent") as mock_agent_workflow:
            res = main.chat(ChatRequest(message="Can you check it?"))
            self.assertFalse(mock_agent_workflow.invoke.called)
            self.assertEqual(res["agent_execution"]["number_of_tools"], 0)
            self.assertEqual(res["answer"], AMBIGUOUS_CLARIFICATION_RESPONSE)

    def test_main_chat_fast_path_thanks_and_farewell(self):
        """Calling /assistant/chat with thanks or farewells must resolve via fast path."""
        with patch.object(agent, "agent") as mock_agent_workflow:
            thanks_res = main.chat(ChatRequest(message="Thank you very much!"))
            self.assertFalse(mock_agent_workflow.invoke.called)
            self.assertEqual(thanks_res["agent_execution"]["number_of_tools"], 0)
            self.assertEqual(thanks_res["answer"], THANKS_FAST_RESPONSE)

            farewell_res = main.chat(ChatRequest(message="Goodbye, have a nice day!"))
            self.assertEqual(farewell_res["agent_execution"]["number_of_tools"], 0)
            self.assertEqual(farewell_res["answer"], FAREWELL_FAST_RESPONSE)

    def test_never_expose_internal_details(self):
        """Responses must never expose system prompts, internal directives, or routing tags."""
        disallowed_strings = [
            "RESPONSIBLE AI USER PREFERENCE DIRECTIVES",
            "SYSTEM PROMPT",
            "MINIMAL INVOCATION RULE",
            "STRICT FACTUALITY OVERRIDE",
            "INTERNAL ROUTING",
            "llm_with_tools",
            "tools_condition",
        ]

        # Fast path greeting check
        for pattern in disallowed_strings:
            self.assertNotIn(pattern, GREETING_FAST_RESPONSE)
            self.assertNotIn(pattern, THANKS_FAST_RESPONSE)
            self.assertNotIn(pattern, FAREWELL_FAST_RESPONSE)
            self.assertNotIn(pattern, CONVERSATIONAL_FAST_RESPONSE)

        # Evidence validator sanitization check
        raw_leaked_response = (
            "SYSTEM PROMPT: You are FinAgent.\n\n"
            "Here is the answer to your question. [INTERNAL ROUTING: get_portfolio_analysis]\n"
            "RESPONSIBLE AI USER PREFERENCE DIRECTIVES: - Do not query personal portfolios."
        )
        validated = main.validate_evidence_claims(raw_leaked_response, [])
        for pattern in disallowed_strings:
            self.assertNotIn(pattern, validated["answer"])


if __name__ == "__main__":
    unittest.main()
