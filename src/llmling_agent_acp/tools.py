"""ACP (Agent Client Protocol) tool bridge for llmling-agent tool execution.

This module provides integration between llmling-agent tools and the ACP protocol,
enabling seamless tool execution with proper streaming, permission handling, and
file system integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from acp import RequestPermissionRequest
from acp.acp_types import (
    ContentToolCallContent,
    TerminalToolCallContent,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
)
from acp.schema import (
    AllowedOutcome,
    PermissionOption,
    SessionNotification,
    ToolCallLocation,
    ToolCallStart as ToolCallStartSchema,
    ToolCallUpdate,
)
from llmling_agent.log import get_logger
from llmling_agent.tools.base import Tool
from llmling_agent_acp.converters import _determine_tool_kind, format_tool_call_for_acp


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from acp import Client
    from acp.acp_types import ToolCallStatus


logger = get_logger(__name__)

OPTIONS = [
    PermissionOption(
        option_id="allow_once",
        name="Allow Once",
        kind="allow_once",
    ),
    PermissionOption(
        option_id="allow_session",
        name="Allow for Session",
        kind="allow_always",
    ),
]


# Create a wrapper that intercepts file operations
class ACPFileSystemTool:
    """Wrapper for file system operations through ACP client."""

    def __init__(self, original_tool: Tool, bridge: ACPToolBridge):
        self.original_tool = original_tool
        self.bridge = bridge
        self.name = original_tool.name
        self.description = original_tool.description

    async def __call__(self, **kwargs: Any) -> Any:
        # Intercept file operations and route through ACP client
        modified_kwargs = await self._process_file_params(kwargs)
        return await self.original_tool.execute(**modified_kwargs)

    async def _process_file_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Process parameters to handle file operations through ACP."""
        # This is a simplified implementation
        # In practice, you'd need more sophisticated file handling
        return params


class ACPToolBridge:
    """Bridges llmling tools with ACP tool call system.

    This bridge handles:
    - Tool execution with ACP streaming updates
    - File system operations through ACP client
    - Permission requests for sensitive operations
    - Tool call progress and status reporting
    """

    def __init__(self, client: Client) -> None:
        """Initialize ACP tool bridge.

        Args:
            client: External library Client interface for operations
        """
        self.client = client
        self._active_tools: dict[str, Any] = {}

    async def execute_tool(
        self,
        tool: Tool,
        params: dict[str, Any],
        session_id: str,
    ) -> AsyncGenerator[SessionNotification, None]:
        """Execute a tool and stream ACP notifications.

        Args:
            tool: Tool instance to execute
            params: Parameters to pass to the tool
            session_id: ACP session identifier

        Yields:
            SessionNotification objects for tool execution updates
        """
        tool_call_id = f"{tool.name}_{hash(str(params))}"

        try:
            # Send initial tool call notification
            yield await _create_tool_start_notification(
                tool, params, session_id, tool_call_id
            )

            # Store active tool
            self._active_tools[tool_call_id] = {
                "tool": tool,
                "params": params,
                "session_id": session_id,
                "status": "running",
            }

            result = await tool.execute(**params)
            yield await _create_tool_completion_notification(
                tool, params, result, session_id, tool_call_id
            )

        except Exception as e:
            logger.exception("Tool execution failed: %s", tool.name)
            yield await _create_tool_error_notification(
                tool, params, str(e), session_id, tool_call_id
            )

        finally:
            # Clean up active tool
            self._active_tools.pop(tool_call_id, None)

    def wrap_tool_for_filesystem(self, tool: Tool) -> Tool:
        """Wrap tool to use ACP filesystem operations.

        Args:
            tool: Original tool to wrap

        Returns:
            Wrapped tool that uses ACP filesystem
        """
        wrapper = ACPFileSystemTool(tool, self)
        return Tool.from_callable(wrapper)

    async def request_tool_permission(
        self,
        tool: Tool,
        params: dict[str, Any],
        session_id: str,
        operation_type: str = "execute",
    ) -> bool:
        """Request permission to execute a tool.

        Args:
            tool: Tool requiring permission
            params: Tool parameters
            session_id: ACP session ID
            operation_type: Type of operation (execute, file_access, etc.)

        Returns:
            True if permission granted, False otherwise
        """
        try:
            # Create tool call update for permission request
            tool_call = ToolCallUpdate(
                tool_call_id=f"{tool!s}_permission_{hash(str(params))}",
                title=f"Execute {tool.name}",
                status="pending",
                kind=_determine_tool_kind(tool.name),
                raw_input=params,
            )

            # Create permission options
            options = [
                PermissionOption(option_id="allow", name="Allow", kind="allow_once"),
                PermissionOption(option_id="deny", name="Deny", kind="reject_once"),
            ]

            # If it's a file operation, add more specific options
            if operation_type == "file_access":
                options.extend(OPTIONS)

            request = RequestPermissionRequest(
                session_id=session_id,
                tool_call=tool_call,
                options=options,
            )

            response = await self.client.request_permission(request)

            return isinstance(
                response.outcome, AllowedOutcome
            ) and response.outcome.option_id in ["allow", "allow_once", "allow_session"]

        except Exception:
            logger.exception("Failed to request tool permission")
            return False

    def get_active_tools(self) -> dict[str, Any]:
        """Get currently active/running tools.

        Returns:
            Dictionary of active tool call IDs and their info
        """
        return self._active_tools.copy()

    async def cancel_tool(self, tool_call_id: str) -> bool:
        """Cancel a running tool execution.

        Args:
            tool_call_id: ID of tool call to cancel

        Returns:
            True if tool was cancelled, False if not found
        """
        tool_info = self._active_tools.get(tool_call_id)
        if not tool_info:
            return False

        # Mark as cancelled
        tool_info["status"] = "cancelled"

        # In a real implementation, you might need to interrupt the tool execution
        # For now, we'll just mark it as cancelled
        logger.info("Cancelled tool call %s", tool_call_id)
        return True


async def _create_tool_start_notification(
    tool: Tool,
    params: dict[str, Any],
    session_id: str,
    tool_call_id: str,
) -> SessionNotification:
    """Create tool start notification.

    Args:
        tool: Tool being executed
        params: Tool parameters
        session_id: ACP session ID
        tool_call_id: Unique tool call identifier

    Returns:
        SessionNotification for tool start
    """
    locations = _extract_file_locations(params)
    tool_kind = _determine_tool_kind(tool.name)

    tool_call = ToolCallStart(
        tool_call_id=tool_call_id,
        title=f"Execute {tool.name}",
        status="in_progress",
        kind=tool_kind,
        locations=locations,
        raw_input=params,
    )

    return SessionNotification(session_id=session_id, update=tool_call)


async def create_tool_progress_update(
    tool_call_id: str,
    progress_message: str,
    session_id: str,
) -> SessionNotification:
    """Create a progress update for a running tool.

    Args:
        tool_call_id: ID of the tool call
        progress_message: Progress message to send
        session_id: ACP session ID

    Returns:
        SessionNotification with progress update
    """
    content = TextContentBlock(text=progress_message)
    update = ToolCallProgress(
        tool_call_id=tool_call_id,
        status="in_progress",
        content=[ContentToolCallContent(content=content)],
    )

    return SessionNotification(session_id=session_id, update=update)


async def _create_tool_completion_notification(
    tool: Tool,
    params: dict[str, Any],
    result: Any,
    session_id: str,
    tool_call_id: str,
) -> SessionNotification:
    """Create tool completion notification.

    Args:
        tool: Tool that was executed
        params: Tool parameters
        result: Tool execution result
        session_id: ACP session ID
        tool_call_id: Unique tool call identifier

    Returns:
        SessionNotification for tool completion
    """
    return format_tool_call_for_acp(
        tool_name=tool.name,
        tool_input=params,
        tool_output=result,
        session_id=session_id,
        status="completed",
    )


async def _create_tool_error_notification(
    tool: Tool,
    params: dict[str, Any],
    error: str,
    session_id: str,
    tool_call_id: str,
) -> SessionNotification:
    """Create tool error notification.

    Args:
        tool: Tool that failed
        params: Tool parameters
        error: Error message
        session_id: ACP session ID
        tool_call_id: Unique tool call identifier

    Returns:
        SessionNotification for tool error
    """
    return format_tool_call_for_acp(
        tool_name=tool.name,
        tool_input=params,
        tool_output=f"Error: {error}",
        session_id=session_id,
        status="failed",
    )


def _extract_file_locations(params: dict[str, Any]) -> list[ToolCallLocation]:
    """Extract file locations from tool parameters.

    Args:
        params: Tool parameters to analyze

    Returns:
        List of file locations found in parameters
    """
    # Common file parameter names
    file_param_names = [
        "path",
        "file_path",
        "filepath",
        "filename",
        "file",
        "input_path",
        "output_path",
        "source",
        "destination",
        "cwd",  # Working directory for terminal operations
    ]
    return [
        ToolCallLocation(path=param_value)
        for param_name, param_value in params.items()
        if param_name.lower() in file_param_names
        and isinstance(param_value, str)
        and ("/" in param_value or "\\" in param_value or "." in param_value)
    ]


def _is_terminal_tool(tool_name: str) -> bool:
    """Check if a tool is a terminal operation tool.

    Args:
        tool_name: Name of the tool to check

    Returns:
        True if tool is terminal-related
    """
    terminal_tool_names = {
        "run_command",
        "get_command_output",
        "create_terminal",
        "wait_for_terminal_exit",
        "kill_terminal",
        "release_terminal",
        "run_command_with_timeout",
    }
    return tool_name in terminal_tool_names


async def create_terminal_tool_call_notification(
    tool_name: str,
    params: dict[str, Any],
    session_id: str,
    tool_call_id: str,
    status: ToolCallStatus = "in_progress",
) -> SessionNotification:
    """Create a terminal tool call notification with embedded terminal.

    Args:
        tool_name: Name of terminal tool
        params: Tool parameters
        session_id: ACP session ID
        tool_call_id: Tool call identifier
        status: Tool call status

    Returns:
        SessionNotification with terminal tool call
    """
    # Extract terminal ID if present
    terminal_id = params.get("terminal_id")

    locations = _extract_file_locations(params)

    tool_call = ToolCallStartSchema(
        tool_call_id=tool_call_id,
        title=f"Terminal: {tool_name}",
        status=status,
        kind="execute",  # Terminal operations are execution type
        locations=locations,
        raw_input=params,
    )

    # Add terminal content if we have a terminal ID
    if terminal_id:
        tool_call.content = [TerminalToolCallContent(terminal_id=terminal_id)]

    return SessionNotification(session_id=session_id, update=tool_call)


class ACPToolRegistry:
    """Registry for managing ACP-compatible tools.

    Provides centralized management of tools with ACP integration,
    including permission handling, filesystem operations, and
    execution monitoring.
    """

    def __init__(self, bridge: ACPToolBridge) -> None:
        """Initialize tool registry.

        Args:
            bridge: ACP tool bridge for execution
        """
        self.bridge = bridge
        self._tools: dict[str, Tool] = {}
        self._tool_permissions: dict[str, dict[str, bool]] = {}

    def register_tool(
        self,
        tool: Tool,
        *,
        requires_permission: bool = False,
        filesystem_access: bool = False,
    ) -> None:
        """Register a tool with the registry.

        Args:
            tool: Tool to register
            requires_permission: Whether tool requires user permission
            filesystem_access: Whether tool needs filesystem access
        """
        tool_name = str(tool)
        self._tools[tool_name] = tool
        self._tool_permissions[tool_name] = {
            "requires_permission": requires_permission,
            "filesystem_access": filesystem_access,
        }

        logger.info("Registered tool %s", tool_name)

    def get_tool(self, name: str) -> Tool | None:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    async def execute_tool(
        self,
        name: str,
        params: dict[str, Any],
        session_id: str,
    ) -> AsyncGenerator[SessionNotification, None]:
        """Execute a registered tool.

        Args:
            name: Tool name to execute
            params: Tool parameters
            session_id: ACP session ID

        Yields:
            SessionNotification objects for tool execution
        """
        tool = self._tools.get(name)
        if not tool:
            logger.error("Tool %s not found", name)
            return

        permissions = self._tool_permissions.get(name, {})

        # Check if permission is required
        if permissions.get("requires_permission", False):
            permission_granted = await self.bridge.request_tool_permission(
                tool, params, session_id, "execute"
            )
            if not permission_granted:
                logger.info("Permission denied for tool %s", name)
                return

        # Handle terminal tools specially
        if _is_terminal_tool(name):
            async for notification in self._execute_terminal_tool(
                tool, params, session_id
            ):
                yield notification
        else:
            # Execute regular tool through bridge
            async for notification in self.bridge.execute_tool(tool, params, session_id):
                yield notification

    async def _execute_terminal_tool(
        self,
        tool: Tool,
        params: dict[str, Any],
        session_id: str,
    ) -> AsyncGenerator[SessionNotification, None]:
        """Execute a terminal tool with special handling.

        Args:
            tool: Terminal tool to execute
            params: Tool parameters
            session_id: ACP session ID

        Yields:
            SessionNotification objects for terminal tool execution
        """
        tool_call_id = f"{tool.name}_{hash(str(params))}"

        try:
            # Send initial terminal tool notification
            yield await create_terminal_tool_call_notification(
                tool.name, params, session_id, tool_call_id, "in_progress"
            )

            # Execute the terminal tool
            result = await tool.execute(**params)

            # Send completion notification
            yield await _create_tool_completion_notification(
                tool, params, result, session_id, tool_call_id
            )

        except Exception as e:
            logger.exception("Terminal tool execution failed: %s", tool.name)
            yield await _create_tool_error_notification(
                tool, params, str(e), session_id, tool_call_id
            )
