"""Unit tests for ConversationMemory and RetrievalAgent execution using unittest."""

import json
import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import ConversationMemory, RetrievalAgent
from tools import create_default_registry


class TestConversationMemory(unittest.TestCase):
    """Verifies that sliding-window memory preserves atomic tool calling turns."""

    def test_memory_sliding_window_preserves_atomic_turns(self) -> None:
        mem = ConversationMemory(max_turns=2)

        # Turn 1
        mem.start_turn("User question 1")
        mem.record_tool_calls([{"id": "call_1", "function": {"name": "calculator", "arguments": '{"expression":"2+2"}'}}])
        mem.record_tool_result("call_1", "calculator", "Result: 4")
        mem.end_turn("Answer 1")

        # Turn 2
        mem.start_turn("User question 2")
        mem.end_turn("Answer 2")

        # Turn 3 (should evict Turn 1, keeping only Turn 2 and Turn 3)
        mem.start_turn("User question 3")
        mem.record_tool_calls([{"id": "call_3", "function": {"name": "get_current_time", "arguments": "{}"}}])
        mem.record_tool_result("call_3", "get_current_time", "Current Time: 2026-10-25")
        mem.end_turn("Answer 3")

        messages = mem.get_messages()

        # Check system prompt
        self.assertEqual(messages[0]["role"], "system")

        # Verify Turn 1 is evicted
        user_queries = [m["content"] for m in messages if m["role"] == "user"]
        self.assertNotIn("User question 1", user_queries)
        self.assertIn("User question 2", user_queries)
        self.assertIn("User question 3", user_queries)

        # Verify Turn 3 tool call and response remain paired atomically
        call_ids = [m.get("tool_call_id") for m in messages if m["role"] == "tool"]
        self.assertIn("call_3", call_ids)
        self.assertNotIn("call_1", call_ids)  # pruned with turn 1

    def test_memory_clear(self) -> None:
        mem = ConversationMemory(max_turns=3)
        mem.start_turn("Hello")
        mem.end_turn("Hi")
        self.assertEqual(len(mem.turns), 1)

        mem.clear()
        self.assertEqual(len(mem.turns), 0)
        self.assertEqual(len(mem.get_messages()), 1)  # Only system prompt remains


class TestRetrievalAgentMock(unittest.TestCase):
    """Verifies agent loop execution in deterministic mock mode."""

    def test_agent_math_tool_execution(self) -> None:
        agent = RetrievalAgent(mock_mode=True)
        response = agent.run("Calculate 25 * 4")
        self.assertTrue("tool-calling loop" in response or "Mock" in response)

        # Inspect memory to verify calculator tool was invoked
        messages = agent.memory.get_messages()
        tool_names = [m.get("name") for m in messages if m["role"] == "tool"]
        self.assertIn("calculator", tool_names)

    def test_agent_rag_search_execution(self) -> None:
        agent = RetrievalAgent(mock_mode=True)
        response = agent.run("Search documents about chunking architecture")

        messages = agent.memory.get_messages()
        tool_names = [m.get("name") for m in messages if m["role"] == "tool"]
        self.assertIn("search_documents", tool_names)


if __name__ == "__main__":
    unittest.main()
