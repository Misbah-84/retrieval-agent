"""Unit tests for the tool execution module using standard library unittest.

Tests cover:
- Mathematical correctness of SafeCalculator
- Security guardrails against arbitrary code execution in SafeCalculator
- Vector similarity search and RAG retrieval
- Time and sandboxed file reading tools
- ToolRegistry execution and argument parsing
"""

import math
import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import (
    calculator,
    create_default_registry,
    get_current_time,
    read_file,
    search_documents,
)
from tools.calculator import SafeCalculatorError, SafeMathEvaluator
from tools.rag_tool import Document, LightweightVectorStore


class TestCalculator(unittest.TestCase):
    """Test suite for the AST-based safe calculator."""

    def setUp(self) -> None:
        self.evaluator = SafeMathEvaluator()

    def test_basic_arithmetic(self) -> None:
        self.assertEqual(self.evaluator.evaluate("2 + 2"), 4)
        self.assertEqual(self.evaluator.evaluate("10 - 4"), 6)
        self.assertEqual(self.evaluator.evaluate("3 * 5"), 15)
        self.assertEqual(self.evaluator.evaluate("20 / 4"), 5.0)
        self.assertEqual(self.evaluator.evaluate("14 % 4"), 2)
        self.assertEqual(self.evaluator.evaluate("2 ** 5"), 32)

    def test_precedence_and_grouping(self) -> None:
        self.assertEqual(self.evaluator.evaluate("2 + 3 * 4"), 14)
        self.assertEqual(self.evaluator.evaluate("(2 + 3) * 4"), 20)
        self.assertEqual(self.evaluator.evaluate("((10 - 2) / 2) + 6"), 10.0)

    def test_whitelisted_math_functions(self) -> None:
        self.assertEqual(self.evaluator.evaluate("sqrt(144)"), 12.0)
        self.assertEqual(self.evaluator.evaluate("abs(-42)"), 42)
        self.assertEqual(self.evaluator.evaluate("round(3.14159, 2)"), 3.14)
        self.assertTrue(math.isclose(self.evaluator.evaluate("sin(pi / 2)"), 1.0))
        self.assertEqual(self.evaluator.evaluate("log10(100)"), 2.0)

    def test_whitelisted_constants(self) -> None:
        self.assertTrue(math.isclose(self.evaluator.evaluate("pi"), math.pi))
        self.assertTrue(math.isclose(self.evaluator.evaluate("e"), math.e))

    def test_division_by_zero_protection(self) -> None:
        with self.assertRaises(SafeCalculatorError):
            self.evaluator.evaluate("10 / 0")

    def test_security_rejection_of_arbitrary_code(self) -> None:
        """Verify that malicious python code strings are rejected at parse time."""
        malicious_inputs = [
            "__import__('os').system('echo hacked')",
            "open('test.txt', 'w')",
            "eval('2 + 2')",
            "exec('x = 1')",
            "lambda x: x + 1",
            "[x for x in [1, 2, 3]]",
            "globals()",
            "locals()",
        ]
        for payload in malicious_inputs:
            with self.assertRaises(SafeCalculatorError):
                self.evaluator.evaluate(payload)

    def test_calculator_tool_wrapper(self) -> None:
        res = calculator("sqrt(16) * 5")
        self.assertIn("Result: 20", res)

        err_res = calculator("invalid + * syntax")
        self.assertIn("Calculator error", err_res)


class TestRAGRetriever(unittest.TestCase):
    """Test suite for the in-memory vector store."""

    def test_vector_store_retrieval(self) -> None:
        store = LightweightVectorStore(use_openai=False)
        docs = [
            Document(id="1", title="Quantum Computing", content="Qubits and superposition principles."),
            Document(id="2", title="Python Concurrency", content="Asyncio, threading, and multiprocessing."),
            Document(id="3", title="RAG Chunking", content="Splitting text into 512 token chunks with overlap."),
        ]
        store.add_documents(docs)

        results = store.search("chunking overlap", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].document.id, "3")
        self.assertIn("512 token chunks", results[0].document.content)

    def test_search_documents_tool(self) -> None:
        output = search_documents("equipment stipend policy", top_k=1)
        self.assertTrue("ACME AI Labs" in output or "stipend" in output.lower())


class TestOtherTools(unittest.TestCase):
    """Test suite for time and file tools."""

    def test_get_current_time(self) -> None:
        res = get_current_time("UTC")
        self.assertIn("Current Time:", res)
        self.assertIn("Timezone: UTC", res)

    def test_read_file_security_sandbox(self) -> None:
        # Relative file inside project
        valid_res = read_file("requirements.txt", max_lines=5)
        self.assertIn("openai", valid_res)

        # Attempted path traversal attack
        traversal_res = read_file("../../../../windows/system32/cmd.exe")
        self.assertTrue("Security error" in traversal_res or "Access denied" in traversal_res)


class TestToolRegistry(unittest.TestCase):
    """Test suite for registry dispatch."""

    def test_registry_dispatch(self) -> None:
        reg = create_default_registry()
        self.assertTrue(reg.has_tool("calculator"))
        self.assertTrue(reg.has_tool("search_documents"))
        self.assertTrue(reg.has_tool("get_current_time"))
        self.assertTrue(reg.has_tool("read_file"))

        # Execute with dict
        res = reg.execute("calculator", {"expression": "100 / 4"})
        self.assertFalse(res.is_error)
        self.assertIn("Result: 25", res.output)

        # Execute with JSON string
        res2 = reg.execute("calculator", '{"expression": "7 * 8"}')
        self.assertFalse(res2.is_error)
        self.assertIn("Result: 56", res2.output)

        # Unknown tool
        unknown_res = reg.execute("non_existent_tool", {})
        self.assertTrue(unknown_res.is_error)


if __name__ == "__main__":
    unittest.main()
