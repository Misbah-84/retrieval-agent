"""CLI Chat Interface for Retrieval Agent.

Provides an interactive terminal interface powered by `rich`, featuring:
- Real-time display of the agent's reasoning steps and tool execution
- Formatted tool invocation panels with parameter inspection and execution latency
- Interactive slash commands (/help, /tools, /memory, /clear, /exit)
- Automatic detection of OpenAI API keys with seamless offline fallback
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from rich import print as rprint
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Ensure current directory is on PYTHONPATH
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Load .env file
load_dotenv(CURRENT_DIR / ".env")

from agent import AgentCallback, ConversationMemory, RetrievalAgent
from tools import ToolExecutionResult, create_default_registry, initialize_rag

console = Console()


class RichTerminalCallback(AgentCallback):
    """Formats and prints agent activity directly to the terminal using Rich."""

    def __init__(self, console_instance: Console) -> None:
        self.console = console_instance

    def on_thought(self, thought: str) -> None:
        self.console.print(f"[dim cyan]› {thought}[/dim cyan]")

    def on_tool_call(self, tool_name: str, arguments: Dict[str, Any], call_id: str) -> None:
        formatted_args = json.dumps(arguments, indent=2)
        panel_content = Text()
        panel_content.append("Tool: ", style="bold magenta")
        panel_content.append(f"{tool_name}\n", style="bold white")
        panel_content.append(f"Arguments:\n{formatted_args}", style="yellow")

        self.console.print(
            Panel(
                panel_content,
                title=f"[bold yellow]⚙ Executing Tool Call: {tool_name}[/bold yellow]",
                border_style="yellow",
                expand=False,
            )
        )

    def on_tool_result(self, result: ToolExecutionResult) -> None:
        style = "red" if result.is_error else "green"
        status_text = "FAILED" if result.is_error else "SUCCESS"

        content = (
            f"[bold {style}]Status:[/] {status_text}  "
            f"[dim](Latency: {result.execution_time_ms:.2f} ms)[/dim]\n\n"
            f"[bold]Observation:[/]\n{result.output}"
        )

        self.console.print(
            Panel(
                content,
                title=f"[{style}]✔ Tool Output: {result.tool_name}[/{style}]",
                border_style=style,
                expand=False,
            )
        )

    def on_final_response(self, response: str) -> None:
        # Final answer is formatted in the main loop
        pass

    def on_error(self, error_message: str) -> None:
        self.console.print(f"[bold red]✖ Agent Error:[/] {error_message}")


def display_banner(model_name: str, mock_mode: bool, max_turns: int) -> None:
    """Renders the application banner and status."""
    title_text = Text("⚡ RETRIEVAL AGENT ⚡", style="bold cyan")
    subtitle = "Production-Quality Function Calling & RAG AI Agent"

    status_table = Table.grid(padding=(0, 2))
    status_table.add_column(style="bold white")
    status_table.add_column(style="cyan")

    mode_label = "[bold red]OFFLINE / MOCK MODE[/bold red]" if mock_mode else f"[bold green]LIVE ({model_name})[/bold green]"
    status_table.add_row("Execution Mode:", mode_label)
    status_table.add_row("Memory Window:", f"{max_turns} turns (atomic preservation)")
    status_table.add_row("Tools Loaded:", "search_documents, calculator, get_current_time, read_file")
    status_table.add_row("Commands:", "Type [bold yellow]/help[/bold yellow] for interactive commands")

    banner_panel = Panel(
        status_table,
        title=f"{title_text.markup} - {subtitle}",
        border_style="cyan",
        padding=(1, 2),
    )
    console.print(banner_panel)


def print_help() -> None:
    """Displays available CLI commands."""
    table = Table(title="Interactive Agent Commands", border_style="cyan")
    table.add_column("Command", style="bold yellow")
    table.add_column("Description", style="white")

    table.add_row("/help", "Show this help menu")
    table.add_row("/tools", "List all registered tools and their input parameter schemas")
    table.add_row("/memory", "Display active conversation memory buffer and message turns")
    table.add_row("/clear", "Reset conversation memory and start a fresh dialogue")
    table.add_row("/exit, /quit", "Exit the retrieval agent CLI")

    console.print(table)


def print_tools(agent: RetrievalAgent) -> None:
    """Displays registered tools and descriptions in a formatted table."""
    table = Table(title="Registered Agent Tools", border_style="magenta")
    table.add_column("Tool Name", style="bold magenta")
    table.add_column("Description", style="white")
    table.add_column("Required Parameters", style="cyan")

    for schema in agent.tools.get_openai_tools():
        func_def = schema.get("function", {})
        name = func_def.get("name", "Unknown")
        desc = func_def.get("description", "No description provided.")
        params = func_def.get("parameters", {}).get("required", [])
        param_str = ", ".join(params) if params else "None"
        table.add_row(name, desc, param_str)

    console.print(table)


def print_memory(agent: RetrievalAgent) -> None:
    """Displays current conversation buffer."""
    stats = agent.memory.get_stats()
    messages = agent.memory.get_messages()

    console.print(
        Panel(
            f"Stored Turns: {stats['stored_turns']} / {stats['max_turns']}\n"
            f"Total Messages in Context: {stats['total_messages']}",
            title="[bold blue]Conversation Memory Diagnostics[/bold blue]",
            border_style="blue",
        )
    )

    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        role_style = {
            "system": "dim cyan",
            "user": "bold cyan",
            "assistant": "bold green",
            "tool": "bold yellow",
        }.get(role, "white")

        content_preview = msg.get("content", "")
        if "tool_calls" in msg and msg["tool_calls"]:
            calls = [f"{c['function']['name']}({c['function']['arguments']})" for c in msg["tool_calls"]]
            content_preview = f"[Tool Calls: {', '.join(calls)}]"

        if len(content_preview) > 160:
            content_preview = content_preview[:160] + "..."

        console.print(f"  [{role_style}][{i}] {role.upper()}:[/{role_style}] {content_preview}")


def run_cli() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Retrieval Agent CLI")
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI model identifier (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force mock/offline demonstration mode",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=int(os.getenv("MAX_MEMORY_TURNS", "5")),
        help="Number of dialogue turns to keep in memory window",
    )
    parser.add_argument(
        "--openai-embeddings",
        action="store_true",
        help="Use OpenAI embeddings for RAG instead of local TF-IDF",
    )

    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    mock_mode = args.mock or not bool(api_key)

    # Initialize RAG embeddings
    initialize_rag(use_openai_embeddings=args.openai_embeddings and bool(api_key), api_key=api_key)

    # Instantiate Tool Registry, Memory, and Agent
    registry = create_default_registry()
    memory = ConversationMemory(max_turns=args.turns)
    callback = RichTerminalCallback(console)

    agent = RetrievalAgent(
        model=args.model,
        api_key=api_key,
        tool_registry=registry,
        memory=memory,
        callback=callback,
        mock_mode=mock_mode,
    )

    display_banner(args.model, mock_mode, args.turns)

    if mock_mode:
        console.print(
            "[yellow]Note: No OPENAI_API_KEY found or --mock specified. Running in Mock/Offline mode.\n"
            "Try sample queries like: 'What is the equipment stipend policy?', 'Calculate sqrt(256) * 12', or 'What time is it?'\n"
            "Add your API key to .env to enable live GPT-4o-mini model reasoning.[/yellow]\n"
        )

    # Interactive Chat Loop
    while True:
        try:
            user_input = console.input("\n[bold cyan]You > [/bold cyan]").strip()

            if not user_input:
                continue

            # Command handling
            if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                console.print("[yellow]Exiting Retrieval Agent. Goodbye![/yellow]")
                break

            if user_input.lower() == "/help":
                print_help()
                continue

            if user_input.lower() == "/tools":
                print_tools(agent)
                continue

            if user_input.lower() == "/memory":
                print_memory(agent)
                continue

            if user_input.lower() == "/clear":
                agent.memory.clear()
                console.print("[green]Conversation memory cleared successfully.[/green]")
                continue

            # Run agent
            console.print("")
            response = agent.run(user_input)

            # Display final response
            console.print(
                Panel(
                    Markdown(response),
                    title="[bold green]Agent Response[/bold green]",
                    border_style="green",
                    padding=(1, 2),
                )
            )

        except KeyboardInterrupt:
            console.print("\n[yellow]Session interrupted. Exiting...[/yellow]")
            break
        except EOFError:
            break


if __name__ == "__main__":
    run_cli()
