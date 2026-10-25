"""Additional useful tools for the agent: real-time clock and sandboxed file reader.

These tools provide the agent with temporal context and safe local filesystem inspection.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def get_current_time(timezone: str = "UTC") -> str:
    """Returns the current date, time, day of the week, and timezone.

    Args:
        timezone: The IANA timezone name (e.g., 'UTC', 'America/New_York', 'Europe/London').
                  Defaults to 'UTC'.

    Returns:
        Formatted string containing the current temporal information.
    """
    try:
        tz = ZoneInfo(timezone.strip())
    except ZoneInfoNotFoundError:
        # Fallback to UTC if timezone is not recognized
        tz = ZoneInfo("UTC")
        timezone = "UTC (fallback from invalid timezone)"
    except Exception:
        tz = datetime.timezone.utc
        timezone = "UTC"

    now = datetime.datetime.now(tz)
    weekday = now.strftime("%A")
    formatted_date = now.strftime("%Y-%m-%d %H:%M:%S")

    return (
        f"Current Time: {formatted_date}\n"
        f"Day of Week: {weekday}\n"
        f"Timezone: {timezone}\n"
        f"ISO-8601: {now.isoformat()}"
    )


# Project root path sandbox (defaults to parent directory of tools)
SAFE_BASE_DIR = Path(__file__).resolve().parent.parent


def read_file(file_path: str, max_lines: int = 50) -> str:
    """Safely reads lines from a text or markdown file within the project sandbox.

    Args:
        file_path: Relative or absolute path to the file to inspect.
        max_lines: Maximum number of lines to return (prevents context blowout).

    Returns:
        The content of the file or an error message if unauthorized/missing.
    """
    try:
        target_path = Path(file_path)
        if not target_path.is_absolute():
            target_path = (SAFE_BASE_DIR / target_path).resolve()
        else:
            target_path = target_path.resolve()

        # Security sandbox check: prevent directory traversal
        try:
            target_path.relative_to(SAFE_BASE_DIR)
        except ValueError:
            return (
                f"Security error: Access denied. Cannot read files outside project sandbox: {SAFE_BASE_DIR}"
            )

        if not target_path.exists():
            return f"File error: '{file_path}' does not exist."

        if not target_path.is_file():
            return f"File error: '{file_path}' is not a regular file."

        # Read lines up to max_lines
        lines = []
        with open(target_path, "r", encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f):
                if idx >= max_lines:
                    lines.append(f"\n... [Truncated: showing first {max_lines} lines of {target_path.name}]")
                    break
                lines.append(line)

        return "".join(lines) if lines else f"[Empty file: {target_path.name}]"
    except Exception as e:
        return f"Error reading file '{file_path}': {str(e)}"


# Tool specifications for OpenAI / Anthropic function calling
TIME_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": (
            "Returns the current date, time, weekday, and ISO timestamp for a specified timezone "
            "(defaults to UTC). Use this for temporal reasoning, checking current date/time, or scheduling."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone string, e.g. 'UTC', 'America/New_York', 'Europe/Paris'. Default is 'UTC'.",
                    "default": "UTC",
                }
            },
            "required": [],
        },
    },
}

READ_FILE_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Safely reads content from a file inside the project workspace. "
            "Use this to inspect local documentation, configuration, or data files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative path to the file inside the project repository.",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum lines of content to read (default: 50).",
                    "default": 50,
                },
            },
            "required": ["file_path"],
        },
    },
}
