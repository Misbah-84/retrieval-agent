"""Tool registry and dispatcher module.

Manages tool registration, JSON schema generation for LLM function calling,
and safe execution with error catching and latency profiling.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .calculator import CALCULATOR_TOOL_DEFINITION, calculator
from .other_tools import (
    READ_FILE_TOOL_DEFINITION,
    TIME_TOOL_DEFINITION,
    get_current_time,
    read_file,
)
from .rag_tool import (
    RAG_TOOL_DEFINITION,
    initialize_rag,
    search_documents,
    vector_store,
)


@dataclass
class ToolExecutionResult:
    """Encapsulates the output and metadata of a tool execution."""
    tool_name: str
    arguments: Dict[str, Any]
    output: str
    execution_time_ms: float
    is_error: bool = False


class ToolRegistry:
    """Registry maintaining available tools, schemas, and dispatch execution."""

    def __init__(self) -> None:
        self._tools: Dict[str, Callable[..., Any]] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        func: Callable[..., Any],
        schema: Dict[str, Any],
    ) -> None:
        """Registers a new tool function and its corresponding OpenAI schema."""
        self._tools[name] = func
        self._schemas[name] = schema

    def has_tool(self, name: str) -> bool:
        """Check if a tool exists in the registry."""
        return name in self._tools

    def get_tool_names(self) -> List[str]:
        """Returns list of registered tool names."""
        return list(self._tools.keys())

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """Returns the list of tool definition schemas formatted for OpenAI tool calling."""
        return list(self._schemas.values())

    def execute(self, name: str, arguments: str | Dict[str, Any]) -> ToolExecutionResult:
        """Safely executes a registered tool by name with provided arguments.

        Args:
            name: The tool name to invoke.
            arguments: Either a JSON string or pre-parsed dictionary of arguments.

        Returns:
            ToolExecutionResult with the output, latency, and status.
        """
        start_time = time.perf_counter()

        # Parse arguments if passed as JSON string from LLM
        parsed_args: Dict[str, Any] = {}
        if isinstance(arguments, str):
            try:
                parsed_args = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as err:
                elapsed = (time.perf_counter() - start_time) * 1000
                return ToolExecutionResult(
                    tool_name=name,
                    arguments={"raw_input": arguments},
                    output=f"Error: Malformed JSON arguments: {str(err)}",
                    execution_time_ms=elapsed,
                    is_error=True,
                )
        elif isinstance(arguments, dict):
            parsed_args = arguments

        if name not in self._tools:
            elapsed = (time.perf_counter() - start_time) * 1000
            return ToolExecutionResult(
                tool_name=name,
                arguments=parsed_args,
                output=f"Error: Tool '{name}' is not recognized. Available tools: {', '.join(self.get_tool_names())}",
                execution_time_ms=elapsed,
                is_error=True,
            )

        func = self._tools[name]
        try:
            raw_output = func(**parsed_args)
            output_str = str(raw_output)
            is_error = False
        except TypeError as err:
            output_str = f"Argument error calling tool '{name}': {str(err)}"
            is_error = True
        except Exception as err:
            output_str = f"Execution error in tool '{name}': {str(err)}"
            is_error = True

        elapsed = (time.perf_counter() - start_time) * 1000
        return ToolExecutionResult(
            tool_name=name,
            arguments=parsed_args,
            output=output_str,
            execution_time_ms=elapsed,
            is_error=is_error,
        )


def create_default_registry() -> ToolRegistry:
    """Creates and returns the standard ToolRegistry populated with project tools."""
    registry = ToolRegistry()
    registry.register("search_documents", search_documents, RAG_TOOL_DEFINITION)
    registry.register("calculator", calculator, CALCULATOR_TOOL_DEFINITION)
    registry.register("get_current_time", get_current_time, TIME_TOOL_DEFINITION)
    registry.register("read_file", read_file, READ_FILE_TOOL_DEFINITION)
    return registry


__all__ = [
    "ToolRegistry",
    "ToolExecutionResult",
    "create_default_registry",
    "calculator",
    "search_documents",
    "get_current_time",
    "read_file",
    "initialize_rag",
    "vector_store",
]
