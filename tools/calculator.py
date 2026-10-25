"""Safe calculator tool for mathematical evaluation using AST parsing.

Prevents arbitrary code execution by strictly validating the Abstract Syntax Tree (AST)
and only permitting whitelisted mathematical operators and functions.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable, Dict

# Whitelisted binary operators
BINARY_OPS: Dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Whitelisted unary operators
UNARY_OPS: Dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Whitelisted mathematical functions from the math module
ALLOWED_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "pow": math.pow,
    "degrees": math.degrees,
    "radians": math.radians,
}

# Whitelisted mathematical constants
ALLOWED_CONSTANTS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

# Safety limits to prevent denial-of-service via huge numbers
MAX_POWER_EXPONENT = 1_000
MAX_POWER_BASE = 1_000_000


class SafeCalculatorError(Exception):
    """Exception raised when a mathematical expression is unsafe or invalid."""
    pass


class SafeMathEvaluator:
    """Evaluates mathematical expressions safely using Python's AST."""

    def evaluate(self, expression: str) -> float | int:
        """Parse and safely evaluate a mathematical expression string.

        Args:
            expression: Math expression to compute (e.g., '(12 * 4) + sqrt(144)').

        Returns:
            The evaluated integer or float result.

        Raises:
            SafeCalculatorError: If expression contains disallowed operations,
                                 syntax errors, or causes arithmetic errors.
        """
        if not expression or not expression.strip():
            raise SafeCalculatorError("Expression is empty.")

        cleaned = expression.strip()

        try:
            tree = ast.parse(cleaned, mode="eval")
        except SyntaxError as e:
            raise SafeCalculatorError(f"Syntax error in expression: {e.msg}") from e

        return self._eval_node(tree.body)

    def _eval_node(self, node: ast.AST) -> float | int:
        # Numeric constants (Python 3.8+ uses ast.Constant)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise SafeCalculatorError(f"Disallowed literal type: {type(node.value).__name__}")

        # Named constants (e.g., pi, e)
        if isinstance(node, ast.Name):
            if node.id in ALLOWED_CONSTANTS:
                return ALLOWED_CONSTANTS[node.id]
            raise SafeCalculatorError(f"Disallowed or unknown identifier: '{node.id}'")

        # Binary operations (e.g. a + b, a * b, a ** b)
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in BINARY_OPS:
                raise SafeCalculatorError(f"Unsupported binary operator: {op_type.__name__}")

            left_val = self._eval_node(node.left)
            right_val = self._eval_node(node.right)

            # Prevent resource exhaustion via massive exponentiation
            if op_type is ast.Pow:
                if abs(right_val) > MAX_POWER_EXPONENT or abs(left_val) > MAX_POWER_BASE:
                    raise SafeCalculatorError(
                        f"Exponentiation exceeds safety bounds (base <= {MAX_POWER_BASE}, exp <= {MAX_POWER_EXPONENT})."
                    )

            # Zero division protection
            if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right_val == 0:
                raise SafeCalculatorError("Division by zero is undefined.")

            try:
                result = BINARY_OPS[op_type](left_val, right_val)
                return result
            except OverflowError:
                raise SafeCalculatorError("Arithmetic operation resulted in an overflow.")
            except Exception as e:
                raise SafeCalculatorError(f"Error executing operation: {str(e)}")

        # Unary operations (e.g. -x, +x)
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in UNARY_OPS:
                raise SafeCalculatorError(f"Unsupported unary operator: {op_type.__name__}")
            operand_val = self._eval_node(node.operand)
            return UNARY_OPS[op_type](operand_val)

        # Function calls (e.g. sqrt(16), log(100, 10))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise SafeCalculatorError("Only simple function calls are allowed (e.g., sqrt(16)).")

            func_name = node.func.id
            if func_name not in ALLOWED_FUNCTIONS:
                raise SafeCalculatorError(f"Function '{func_name}' is not in the allowed functions whitelist.")

            if node.keywords:
                raise SafeCalculatorError("Keyword arguments are not allowed in function calls.")

            args = [self._eval_node(arg) for arg in node.args]

            try:
                func = ALLOWED_FUNCTIONS[func_name]
                return func(*args)
            except ValueError as e:
                raise SafeCalculatorError(f"Math error in {func_name}(): {str(e)}")
            except OverflowError:
                raise SafeCalculatorError(f"Overflow error in {func_name}().")
            except Exception as e:
                raise SafeCalculatorError(f"Error calling {func_name}(): {str(e)}")

        # Any other AST node is rejected
        raise SafeCalculatorError(f"Disallowed expression node type: {type(node).__name__}")


_evaluator = SafeMathEvaluator()


def calculator(expression: str) -> str:
    """Evaluates mathematical expressions safely without using arbitrary eval().

    Args:
        expression: A mathematical expression string, e.g. "sqrt(256) * 12 + 45".

    Returns:
        A formatted string with the result, or an error message explaining the issue.
    """
    try:
        result = _evaluator.evaluate(expression)
        # Format cleanly: if float is whole number, display as int/clean float
        if isinstance(result, float) and result.is_integer():
            return f"Result: {int(result)}"
        if isinstance(result, float):
            return f"Result: {round(result, 6)}"
        return f"Result: {result}"
    except SafeCalculatorError as err:
        return f"Calculator error: {str(err)}"
    except Exception as err:
        return f"Unexpected error evaluating expression: {str(err)}"


# Tool specification for OpenAI / Anthropic function calling
CALCULATOR_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "Safely evaluates mathematical expressions. Supports addition (+), subtraction (-), "
            "multiplication (*), division (/), modulo (%), powers (**), and functions like "
            "sqrt, abs, round, sin, cos, tan, log, log10, exp, floor, ceil, pi, and e."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate, e.g., '(45 * 12) + sqrt(144)'.",
                }
            },
            "required": ["expression"],
        },
    },
}
