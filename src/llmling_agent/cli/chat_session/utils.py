from __future__ import annotations

import httpx

from llmling_agent.chat_session.exceptions import ChatSessionConfigError


def format_error(error: Exception) -> str:
    """Format error message for display."""
    # Known error types we want to handle specially
    match error:
        case ChatSessionConfigError():
            return f"Chat session error: {error}"
        case ValueError() if "token" in str(error):
            return "Connection interrupted"
        case httpx.ReadError():
            return "Connection lost. Please try again."
        case GeneratorExit():
            return "Response stream interrupted"
        case _:
            return f"Error: {error}"
