"""Core AI Retrieval Agent implementation with native function calling and conversation memory.

Features:
- OpenAI tool-calling loop (Model -> Tool Execution -> Observation -> Decision)
- Atomic Turn-aware Conversation Memory (preserves tool call/response pairs during sliding-window pruning)
- Configurable callback hooks for logging and UI rendering
- Built-in Mock/Offline demonstration mode for testing without an API key
- Robust error handling for rate limits, API timeouts, and malformed inputs
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI, OpenAIError

from tools import ToolExecutionResult, ToolRegistry, create_default_registry


DEFAULT_SYSTEM_PROMPT = """You are a helpful, precise, and transparent AI engineering assistant.
You have access to a set of specialized tools:
1. `search_documents(query)`: Retrieves verified engineering architecture docs, evaluation guides, and internal company policies from the vector store.
2. `calculator(expression)`: Safely computes mathematical and scientific expressions.
3. `get_current_time(timezone)`: Retrieves real-time temporal and calendar data.
4. `read_file(file_path)`: Reads local project documentation or configuration files.

GUIDELINES:
- Always use `search_documents` when asked about technical architecture, chunking, RAG evaluations, or company policies rather than hallucinating.
- Always use `calculator` for math computations instead of estimating.
- Always use `get_current_time` if asked for the current date, time, day of the week, or time-sensitive calculations.
- Maintain a professional, structured, and concise tone. When presenting retrieved information, cite the relevant document source title.
"""


@dataclass
class ConversationTurn:
    """Represents an atomic interaction turn in the dialogue.

    An atomic turn includes:
    - User message
    - Any intermediate assistant tool-call messages and corresponding tool responses
    - Final assistant response message
    """
    user_message: Dict[str, Any]
    intermediate_messages: List[Dict[str, Any]] = field(default_factory=list)
    assistant_message: Optional[Dict[str, Any]] = None

    def to_messages_list(self) -> List[Dict[str, Any]]:
        """Converts the turn into a sequential list of message dicts for the API."""
        msgs = [self.user_message]
        msgs.extend(self.intermediate_messages)
        if self.assistant_message:
            msgs.append(self.assistant_message)
        return msgs


class ConversationMemory:
    """Sliding-window conversation memory with atomic turn preservation.

    Ensures that when pruning older history, tool_calls are never separated
    from their matching 'tool' response messages, preventing API 400 Bad Request errors.
    """

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT, max_turns: int = 5) -> None:
        self.system_prompt: str = system_prompt
        self.max_turns: int = max_turns
        self.turns: List[ConversationTurn] = []
        self._current_turn: Optional[ConversationTurn] = None

    def start_turn(self, user_content: str) -> None:
        """Starts a new atomic dialogue turn with a user query."""
        user_msg = {"role": "user", "content": user_content}
        self._current_turn = ConversationTurn(user_message=user_msg)

    def record_tool_calls(self, tool_calls: List[Any], content: Optional[str] = None) -> None:
        """Records an assistant message requesting tool calls."""
        if self._current_turn is None:
            return

        serialized_tool_calls = []
        for tc in tool_calls:
            if hasattr(tc, "model_dump"):
                serialized_tool_calls.append(tc.model_dump())
            elif isinstance(tc, dict):
                serialized_tool_calls.append(tc)
            else:
                serialized_tool_calls.append({
                    "id": getattr(tc, "id", "call_id"),
                    "type": "function",
                    "function": {
                        "name": getattr(tc.function, "name", ""),
                        "arguments": getattr(tc.function, "arguments", "{}"),
                    },
                })

        msg = {
            "role": "assistant",
            "content": content or "",
            "tool_calls": serialized_tool_calls,
        }
        self._current_turn.intermediate_messages.append(msg)

    def record_tool_result(self, tool_call_id: str, name: str, result: str) -> None:
        """Records a tool's observation response."""
        if self._current_turn is None:
            return

        msg = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": result,
        }
        self._current_turn.intermediate_messages.append(msg)

    def end_turn(self, assistant_content: str) -> None:
        """Finalizes the turn with the assistant's final textual response."""
        if self._current_turn is None:
            return

        self._current_turn.assistant_message = {
            "role": "assistant",
            "content": assistant_content,
        }
        self.turns.append(self._current_turn)
        self._current_turn = None

        # Apply sliding window: retain only the most recent `max_turns`
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def clear(self) -> None:
        """Resets conversation history back to initial state."""
        self.turns.clear()
        self._current_turn = None

    def get_messages(self) -> List[Dict[str, Any]]:
        """Returns the full ordered message list suitable for an OpenAI chat completion."""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        # Add completed turns
        for turn in self.turns:
            messages.extend(turn.to_messages_list())

        # Add active turn in-flight if applicable
        if self._current_turn is not None:
            messages.extend(self._current_turn.to_messages_list())

        return messages

    def get_stats(self) -> Dict[str, Any]:
        """Returns summary statistics for the conversation state."""
        total_messages = len(self.get_messages())
        return {
            "stored_turns": len(self.turns),
            "max_turns": self.max_turns,
            "total_messages": total_messages,
        }


class AgentCallback:
    """Callback interface for tracking reasoning and tool execution in real-time."""

    def on_thought(self, thought: str) -> None:
        """Called when the agent generates reasoning or status thoughts."""
        pass

    def on_tool_call(self, tool_name: str, arguments: Dict[str, Any], call_id: str) -> None:
        """Called immediately before a tool is invoked."""
        pass

    def on_tool_result(self, result: ToolExecutionResult) -> None:
        """Called immediately after a tool finishes execution."""
        pass

    def on_final_response(self, response: str) -> None:
        """Called when the final response is produced."""
        pass

    def on_error(self, error_message: str) -> None:
        """Called on encountering a critical error."""
        pass


class RetrievalAgent:
    """Production-ready Tool-Calling AI Agent."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        tool_registry: Optional[ToolRegistry] = None,
        memory: Optional[ConversationMemory] = None,
        max_iterations: int = 8,
        callback: Optional[AgentCallback] = None,
        mock_mode: bool = False,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.tools = tool_registry or create_default_registry()
        self.memory = memory or ConversationMemory()
        self.max_iterations = max_iterations
        self.callback = callback or AgentCallback()
        self.mock_mode = mock_mode or (not bool(self.api_key) and not os.getenv("OPENAI_API_KEY"))

        if not self.mock_mode:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None

    def run(self, user_prompt: str) -> str:
        """Executes the agent loop on the provided user prompt.

        Args:
            user_prompt: The user query or task instruction.

        Returns:
            The agent's final text answer.
        """
        if not user_prompt or not user_prompt.strip():
            return "Please provide a non-empty question or instruction."

        # Start conversation turn
        self.memory.start_turn(user_prompt.strip())

        if self.mock_mode:
            return self._run_mock_loop(user_prompt)

        return self._run_openai_loop()

    def _run_openai_loop(self) -> str:
        """Standard OpenAI function-calling iterative execution loop."""
        tools_schema = self.tools.get_openai_tools()

        for iteration in range(1, self.max_iterations + 1):
            messages = self.memory.get_messages()

            try:
                self.callback.on_thought(f"Querying model '{self.model}' (Turn iteration {iteration}/{self.max_iterations})...")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools_schema if tools_schema else None,
                    tool_choice="auto" if tools_schema else None,
                )
            except OpenAIError as e:
                err_msg = f"OpenAI API Error: {str(e)}"
                self.callback.on_error(err_msg)
                self.memory.end_turn(f"I encountered an error communicating with the model: {err_msg}")
                return err_msg
            except Exception as e:
                err_msg = f"Unexpected execution error: {str(e)}"
                self.callback.on_error(err_msg)
                self.memory.end_turn(f"An unexpected error occurred: {err_msg}")
                return err_msg

            choice = response.choices[0]
            message = choice.message

            # Check if the model decided to call tools
            if message.tool_calls:
                self.callback.on_thought(f"Model selected {len(message.tool_calls)} tool(s) to execute.")
                self.memory.record_tool_calls(message.tool_calls, content=message.content)

                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    raw_args = tool_call.function.arguments

                    # Parse arguments dictionary for callback display
                    try:
                        parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception:
                        parsed_args = {"raw_input": raw_args}

                    self.callback.on_tool_call(func_name, parsed_args, tool_call.id)

                    # Execute the tool safely via registry
                    exec_result = self.tools.execute(func_name, raw_args)
                    self.callback.on_tool_result(exec_result)

                    # Append tool observation back to conversation history
                    self.memory.record_tool_result(
                        tool_call_id=tool_call.id,
                        name=func_name,
                        result=exec_result.output,
                    )

                # Continue loop: feed observations back into LLM
                continue

            # No tool calls: final answer produced
            final_content = message.content or "[Empty response from model]"
            self.memory.end_turn(final_content)
            self.callback.on_final_response(final_content)
            return final_content

        # Reached max iterations without terminal answer
        fallback_msg = (
            f"Agent reached the maximum iteration limit ({self.max_iterations}) without concluding. "
            "Please refine your query or increase max_iterations."
        )
        self.memory.end_turn(fallback_msg)
        self.callback.on_thought(fallback_msg)
        return fallback_msg

    def _run_mock_loop(self, user_prompt: str) -> str:
        """Deterministic offline mode for testing tools, logging, and memory without an API key."""
        self.callback.on_thought("[Mock Mode Active] Simulating tool-calling reasoning loop...")
        prompt_lower = user_prompt.lower()

        # Heuristic 1: Math calculation detection
        math_match = re.search(r"(\d+[\s\+\-\*\/\%\^\(\)\.\w]+|\bsqrt\([^)]+\)|\blog\([^)]+\))", user_prompt)
        has_calc_intent = any(k in prompt_lower for k in ["calculate", "math", "+", "*", "sqrt", "divide", "multiply"])

        # Heuristic 2: Search documents intent
        search_keywords = ["chunking", "rag", "eval", "policy", "remote", "stipend", "vector", "architecture", "guardrail"]
        wants_search = any(k in prompt_lower for k in search_keywords) or "search" in prompt_lower

        # Heuristic 3: Time intent
        wants_time = any(k in prompt_lower for k in ["time", "clock", "date", "today", "day"])

        # Heuristic 4: Read file intent
        wants_file = "read" in prompt_lower or "file" in prompt_lower

        executed_any = False

        if wants_time:
            self.callback.on_thought("Determined intent: temporal lookup via 'get_current_time'.")
            call_id = "mock_call_time_01"
            args = {"timezone": "UTC"}
            self.callback.on_tool_call("get_current_time", args, call_id)
            res = self.tools.execute("get_current_time", args)
            self.callback.on_tool_result(res)
            self.memory.record_tool_calls([{"id": call_id, "type": "function", "function": {"name": "get_current_time", "arguments": json.dumps(args)}}])
            self.memory.record_tool_result(call_id, "get_current_time", res.output)
            executed_any = True

        if wants_search:
            self.callback.on_thought("Determined intent: knowledge retrieval via 'search_documents'.")
            call_id = "mock_call_rag_02"
            # Extract query keywords
            query_clean = user_prompt.replace("search", "").replace("for", "").strip()
            args = {"query": query_clean, "top_k": 2}
            self.callback.on_tool_call("search_documents", args, call_id)
            res = self.tools.execute("search_documents", args)
            self.callback.on_tool_result(res)
            self.memory.record_tool_calls([{"id": call_id, "type": "function", "function": {"name": "search_documents", "arguments": json.dumps(args)}}])
            self.memory.record_tool_result(call_id, "search_documents", res.output)
            executed_any = True

        if has_calc_intent and math_match:
            self.callback.on_thought("Determined intent: mathematical evaluation via 'calculator'.")
            call_id = "mock_call_calc_03"
            expression = math_match.group(0).strip()
            args = {"expression": expression}
            self.callback.on_tool_call("calculator", args, call_id)
            res = self.tools.execute("calculator", args)
            self.callback.on_tool_result(res)
            self.memory.record_tool_calls([{"id": call_id, "type": "function", "function": {"name": "calculator", "arguments": json.dumps(args)}}])
            self.memory.record_tool_result(call_id, "calculator", res.output)
            executed_any = True

        if wants_file and not executed_any:
            self.callback.on_thought("Determined intent: file inspection via 'read_file'.")
            call_id = "mock_call_file_04"
            args = {"file_path": "data/knowledge_base.json", "max_lines": 15}
            self.callback.on_tool_call("read_file", args, call_id)
            res = self.tools.execute("read_file", args)
            self.callback.on_tool_result(res)
            self.memory.record_tool_calls([{"id": call_id, "type": "function", "function": {"name": "read_file", "arguments": json.dumps(args)}}])
            self.memory.record_tool_result(call_id, "read_file", res.output)
            executed_any = True

        # Generate synthesized mock response
        if executed_any:
            final_resp = (
                f"[Offline/Mock Agent Response]\n\n"
                f"I processed your query ('{user_prompt}') using the tool-calling loop. "
                f"The tools were successfully invoked and recorded in the conversation memory buffer. "
                f"To enable live OpenAI reasoning, provide an `OPENAI_API_KEY` in your `.env` file."
            )
        else:
            final_resp = (
                f"[Offline/Mock Agent Response] Hello! I received your query: '{user_prompt}'. "
                f"Currently running in Mock Mode because no OPENAI_API_KEY is configured. "
                f"Try asking: 'What is the equipment stipend?', 'Calculate sqrt(256) * 10', or 'What time is it?'."
            )

        self.memory.end_turn(final_resp)
        self.callback.on_final_response(final_resp)
        return final_resp
