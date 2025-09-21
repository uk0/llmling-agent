"""Client implementation for integrating llmling-agent with ACP (Agent Client Protocol).

This module provides client implementations that work with the external acp library,
specifically implementing the Client protocol interface for filesystem operations,
permissions, and session updates.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from acp import (
    Client,
    ReadTextFileResponse,
    RequestPermissionResponse,
)
from acp.schema import RequestPermissionOutcome1, RequestPermissionOutcome2

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from acp import (
        ReadTextFileRequest,
        RequestPermissionRequest,
        SessionNotification,
        WriteTextFileRequest,
    )

logger = get_logger(__name__)


class DefaultACPClient:
    """Default implementation of ACP Client interface for basic operations.

    This provides a basic client implementation that can be used for testing
    or as a base for more sophisticated client implementations.
    """

    def __init__(self, *, allow_file_operations: bool = False) -> None:
        """Initialize default ACP client.

        Args:
            allow_file_operations: Whether to allow file read/write operations
        """
        self.allow_file_operations = allow_file_operations
        self._session_updates: list[SessionNotification] = []

    async def requestPermission(
        self, params: RequestPermissionRequest
    ) -> RequestPermissionResponse:
        """Default permission handler - grants all permissions.

        Args:
            params: Permission request parameters

        Returns:
            Permission response granting access
        """
        logger.info("Permission requested for %s", params.toolCall.title or "operation")

        # Default: grant permission for the first option
        if params.options:
            id_ = params.options[0].optionId
            outcome = RequestPermissionOutcome2(outcome="selected", optionId=id_)
            return RequestPermissionResponse(outcome=outcome)

        # No options - deny
        outcome = RequestPermissionOutcome1(outcome="cancelled")
        return RequestPermissionResponse(outcome=outcome)

    async def sessionUpdate(self, params: SessionNotification) -> None:
        """Handle session update notifications.

        Args:
            params: Session update notification
        """
        msg = "Session update for %s: %s"
        logger.debug(msg, params.sessionId, params.update.sessionUpdate)
        self._session_updates.append(params)

    async def writeTextFile(self, params: WriteTextFileRequest) -> None:
        """Write text to file (if allowed).

        Args:
            params: File write request parameters
        """
        if not self.allow_file_operations:
            msg = "File operations not allowed"
            raise RuntimeError(msg)

        try:
            path = Path(params.path)
            path.write_text(params.content, encoding="utf-8")
            logger.info("Wrote file %s", params.path)
        except Exception:
            logger.exception("Failed to write file %s", params.path)
            raise

    async def readTextFile(self, params: ReadTextFileRequest) -> ReadTextFileResponse:
        """Read text from file (if allowed).

        Args:
            params: File read request parameters

        Returns:
            File content response
        """
        if not self.allow_file_operations:
            msg = "File operations not allowed"
            raise RuntimeError(msg)

        try:
            path = Path(params.path)

            if not path.exists():
                msg = f"File not found: {params.path}"
                raise FileNotFoundError(msg)  # noqa: TRY301

            content = path.read_text(encoding="utf-8")

            # Apply line filtering if requested
            if params.line is not None or params.limit is not None:
                lines = content.splitlines()
                start_line = (params.line - 1) if params.line else 0
                end_line = start_line + params.limit if params.limit else len(lines)
                content = "\n".join(lines[start_line:end_line])

            logger.info("Read file %s", params.path)
            return ReadTextFileResponse(content=content)

        except Exception:
            logger.exception("Failed to read file %s", params.path)
            raise

    async def createTerminal(self, params: Any) -> Any:
        """Create terminal (not implemented).

        Args:
            params: Terminal creation parameters

        Returns:
            Terminal creation response
        """
        msg = "Terminal operations not implemented"
        raise NotImplementedError(msg)

    async def terminalOutput(self, params: Any) -> Any:
        """Get terminal output (not implemented).

        Args:
            params: Terminal output request parameters

        Returns:
            Terminal output response
        """
        msg = "Terminal operations not implemented"
        raise NotImplementedError(msg)

    async def releaseTerminal(self, params: Any) -> None:
        """Release terminal (not implemented).

        Args:
            params: Terminal release parameters
        """
        msg = "Terminal operations not implemented"
        raise NotImplementedError(msg)

    async def waitForTerminalExit(self, params: Any) -> Any:
        """Wait for terminal exit (not implemented).

        Args:
            params: Terminal wait parameters

        Returns:
            Terminal exit response
        """
        msg = "Terminal operations not implemented"
        raise NotImplementedError(msg)

    async def killTerminal(self, params: Any) -> None:
        """Kill terminal (not implemented).

        Args:
            params: Terminal kill parameters
        """
        msg = "Terminal operations not implemented"
        raise NotImplementedError(msg)

    def get_session_updates(self) -> list[SessionNotification]:
        """Get all received session updates.

        Returns:
            List of session update notifications
        """
        return self._session_updates.copy()

    def clear_session_updates(self) -> None:
        """Clear all stored session updates."""
        self._session_updates.clear()


class FileSystemACPClient(DefaultACPClient):
    """ACP client with enhanced filesystem operations and permission handling.

    This client provides more sophisticated file handling with proper permission
    checks and user interaction for sensitive operations.
    """

    def __init__(
        self,
        *,
        allowed_paths: list[str] | None = None,
        require_permission: bool = True,
    ) -> None:
        """Initialize filesystem ACP client.

        Args:
            allowed_paths: List of allowed file paths/patterns
            require_permission: Whether to require permission for file operations
        """
        super().__init__(allow_file_operations=True)
        self.allowed_paths = allowed_paths or []
        self.require_permission = require_permission

    def _is_path_allowed(self, path: str) -> bool:
        """Check if a path is allowed for operations.

        Args:
            path: File path to check

        Returns:
            True if path is allowed
        """
        if not self.allowed_paths:
            return True  # No restrictions if no allowed paths specified

        path_obj = Path(path).resolve()

        for allowed in self.allowed_paths:
            allowed_path = Path(allowed).resolve()
            try:
                path_obj.relative_to(allowed_path)
            except ValueError:
                continue
            else:
                return True

        return False

    async def writeTextFile(self, params: WriteTextFileRequest) -> None:
        """Write text to file with permission checking.

        Args:
            params: File write request parameters
        """
        if not self._is_path_allowed(params.path):
            msg = f"Path not allowed: {params.path}"
            raise PermissionError(msg)

        await super().writeTextFile(params)

    async def readTextFile(self, params: ReadTextFileRequest) -> ReadTextFileResponse:
        """Read text from file with permission checking.

        Args:
            params: File read request parameters

        Returns:
            File content response
        """
        if not self._is_path_allowed(params.path):
            msg = f"Path not allowed: {params.path}"
            raise PermissionError(msg)

        return await super().readTextFile(params)


# Type alias for compatibility
ACPClientInterface = Client
