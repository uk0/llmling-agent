"""ACP notification helper for clean session update API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from acp.schema import (
    AgentPlan,
    ContentToolCallContent,
    FileEditToolCallContent,
    SessionNotification,
    TerminalToolCallContent,
    TextContentBlock,
    ToolCallLocation,
    ToolCallProgress,
    ToolCallStart,
)
from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Sequence

    from acp.acp_types import ToolCallKind, ToolCallStatus
    from acp.schema import PlanEntry
    from llmling_agent_acp.session import ACPSession

logger = get_logger(__name__)


ContentType = Sequence[
    ContentToolCallContent | FileEditToolCallContent | TerminalToolCallContent | str
]


class ACPNotifications:
    """Clean API for creating and sending ACP session notifications.

    Provides convenient methods for common notification patterns,
    handling both creation and sending in a single call.
    """

    def __init__(self, session: ACPSession) -> None:
        """Initialize notifications helper.

        Args:
            session: ACP session containing connection and session_id
        """
        self.session = session

    async def tool_call_start(
        self,
        tool_call_id: str,
        title: str,
        *,
        kind: ToolCallKind | None = None,
        locations: Sequence[ToolCallLocation] | None = None,
        content: ContentType | None = None,
        raw_input: dict[str, Any] | None = None,
    ) -> None:
        """Send a tool call start notification.

        Args:
            tool_call_id: Tool call identifier
            title: Optional title for the start notification
            kind: Optional tool call kind
            locations: Optional sequence of file/path locations
            content: Optional sequence of content blocks
            raw_input: Optional raw input data
        """
        start = ToolCallStart(
            tool_call_id=tool_call_id,
            status="pending",
            title=title,
            kind=kind,
            locations=locations,
            content=[
                ContentToolCallContent(content=TextContentBlock(text=i))
                if isinstance(i, str)
                else i
                for i in content or []
            ],
            raw_input=raw_input,
        )
        notification = SessionNotification(
            session_id=self.session.session_id, update=start
        )
        await self.session.client.session_update(notification)

    async def tool_call_progress(
        self,
        tool_call_id: str,
        status: ToolCallStatus,
        *,
        title: str | None = None,
        raw_output: str | None = None,
        locations: Sequence[ToolCallLocation] | None = None,
        content: ContentType | None = None,
    ) -> None:
        """Send a generic progress notification.

        Args:
            tool_call_id: Tool call identifier
            status: Progress status
            title: Optional title for the progress update
            raw_output: Optional raw output text
            locations: Optional sequence of file/path locations
            content: Optional sequence of content blocks or strings to display
        """
        progress = ToolCallProgress(
            tool_call_id=tool_call_id,
            status=status,
            title=title,
            raw_output=raw_output,
            locations=locations,
            content=[
                ContentToolCallContent(content=TextContentBlock(text=i))
                if isinstance(i, str)
                else i
                for i in content or []
            ],
        )
        notification = SessionNotification(
            session_id=self.session.session_id, update=progress
        )
        await self.session.client.session_update(notification)

    async def file_edit_progress(
        self,
        tool_call_id: str,
        path: str,
        old_text: str,
        new_text: str,
        *,
        line: int | None = None,
        status: ToolCallStatus = "completed",
        title: str | None = None,
    ) -> None:
        """Send a notification with file edit content.

        Args:
            tool_call_id: Tool call identifier
            path: File path being edited
            old_text: Original file content
            new_text: New file content
            line: Line number being edited
            status: Progress status (default: 'completed')
            title: Optional title
        """
        file_edit_content = FileEditToolCallContent(
            path=path,
            old_text=old_text,
            new_text=new_text,
        )
        locations = [ToolCallLocation(path=path, line=line)]
        await self.tool_call_progress(
            tool_call_id=tool_call_id,
            status=status,
            title=title,
            locations=locations,
            content=[file_edit_content],
        )

    async def terminal_progress(
        self,
        tool_call_id: str,
        terminal_id: str,
        *,
        status: ToolCallStatus = "completed",
        title: str | None = None,
        raw_output: str | None = None,
    ) -> None:
        """Send a notification with terminal content.

        Args:
            tool_call_id: Tool call identifier
            terminal_id: Terminal identifier
            status: Progress status (default: 'completed')
            title: Optional title
            raw_output: Optional raw output text
        """
        terminal_content = TerminalToolCallContent(terminal_id=terminal_id)
        await self.tool_call_progress(
            tool_call_id=tool_call_id,
            status=status,
            title=title,
            raw_output=raw_output,
            content=[terminal_content],
        )

    async def plan(self, entries: list[PlanEntry]) -> None:
        """Send a plan notification.

        Args:
            entries: List of plan entries to send
        """
        plan = AgentPlan(entries=entries)
        notification = SessionNotification(
            session_id=self.session.session_id, update=plan
        )
        await self.session.client.session_update(notification)
