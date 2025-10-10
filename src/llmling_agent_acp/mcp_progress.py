"""ACP implementation of MCP progress notification handler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from llmling_agent_acp.session import ACPSession

logger = get_logger(__name__)


def create_acp_progress_handler(acp_session: ACPSession):
    """Create progress handler function for MCP to ACP bridging."""

    async def handle_progress(
        tool_name: str,
        tool_call_id: str,
        tool_input: dict,
        progress: float,
        total: float | None = None,
        message: str | None = None,
    ) -> None:
        """Handle MCP progress and convert to ACP in_progress update."""
        try:
            from llmling_agent_acp.converters import format_tool_call_for_acp

            # Create content from progress message
            output = message if message else f"Progress: {progress}"
            if total:
                output += f"/{total}"

            # Create ACP tool call progress notification
            notification = format_tool_call_for_acp(
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=output,
                session_id=acp_session.session_id,
                status="in_progress",
                tool_call_id=tool_call_id,
            )

            # Send notification via ACP session
            await acp_session.client.session_update(notification)

        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to convert MCP progress to ACP notification: %s", e)

    return handle_progress
