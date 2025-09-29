"""Permission MCP server for handling ACP permission requests.

This module provides a specialized MCP server that handles permission requests
for ACP tools, showing UI dialogs to users and returning permission decisions.
"""

from __future__ import annotations

import asyncio
import contextlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from threading import Thread
from typing import TYPE_CHECKING, Any

import anyenv

from acp import RequestPermissionRequest
from acp.schema import AllowedOutcome, DeniedOutcome, PermissionOption, ToolCallUpdate
from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from acp import Client

logger = get_logger(__name__)

OPTIONS = [
    PermissionOption(option_id="allow_once", name="Allow", kind="allow_once"),
    PermissionOption(option_id="allow_always", name="Always Allow", kind="allow_always"),
    PermissionOption(option_id="reject_once", name="Reject", kind="reject_once"),
]


class PermissionMCPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for permission MCP server."""

    def __init__(self, client: Client, session_id: str, *args: Any, **kwargs: Any):
        self.client = client
        self.session_id = session_id
        super().__init__(*args, **kwargs)

    def log_message(self, fmt: str, *args: Any) -> None:
        """Override to use our logger instead of stderr."""
        logger.debug(fmt, args)

    def do_POST(self) -> None:
        """Handle POST requests for MCP communication."""
        try:
            if self.path != "/mcp":
                self.send_error(404, "Not found")
                return

            # Read request body
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            if not body:
                self.send_error(400, "Empty request body")
                return

            # Parse MCP request
            try:
                request_data = anyenv.load_json(body.decode(), return_type=dict)
            except anyenv.JsonDumpError as e:
                logger.exception("Failed to parse JSON request")
                self.send_error(400, f"Invalid JSON: {e}")
                return

            # Handle MCP request
            response = asyncio.run(self._handle_mcp_request(request_data))

            # Send response
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            response_json = anyenv.dump_json(response)
            self.wfile.write(response_json.encode())

        except Exception as e:
            logger.exception("Error handling POST request")
            with contextlib.suppress(Exception):
                self.send_error(500, f"Internal server error: {e}")

    async def _handle_mcp_request(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Handle MCP protocol request."""
        method = request_data.get("method", "")
        params = request_data.get("params", {})
        request_id = request_data.get("id")

        try:
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "acp-permission", "version": "1.0.0"},
                    },
                }

            if method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": [
                            {
                                "name": "permission",
                                "description": "Handle permission requests for ACP tools",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "tool_name": {"type": "string"},
                                        "input": {"type": "object"},
                                        "tool_use_id": {"type": "string"},
                                    },
                                    "required": ["tool_name", "input"],
                                },
                            }
                        ]
                    },
                }

            if method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})

                if tool_name == "permission":
                    result = await self._handle_permission_request(arguments)
                    data = anyenv.dump_json(result)
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": [{"type": "text", "text": data}]},
                    }
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32602, "message": f"Unknown tool: {tool_name}"},
                }

        except Exception as e:
            logger.exception("Error handling MCP request")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32603, "message": f"Internal error: {e}"},
            }
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

    async def _handle_permission_request(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle permission request and show UI dialog."""
        try:
            tool_name = arguments["tool_name"]
            tool_input = arguments["input"]
            name = f"{tool_name}_{hash(str(tool_input))}"
            tool_use_id = arguments.get("tool_use_id", name)
            logger.info("Handling permission request for tool: %s", tool_name)
            # Create tool call info for permission request
            tool_call = ToolCallUpdate(
                tool_call_id=tool_use_id,
                title=f"Execute {tool_name}",
                status="pending",
                raw_input=tool_input,
            )

            # Create and send permission request
            permission_request = RequestPermissionRequest(
                session_id=self.session_id,
                tool_call=tool_call,
                options=OPTIONS,
            )

            # This shows the permission UI dialog
            response = await self.client.request_permission(permission_request)

            # Process response
            if isinstance(response.outcome, AllowedOutcome):
                return {
                    "behavior": "allow",
                    "updatedInput": tool_input,
                    "updatedPermissions": [],
                }
            if isinstance(response.outcome, DeniedOutcome):
                return {
                    "behavior": "deny",
                    "message": "Permission denied by user",
                }

        except Exception as e:
            logger.exception("Failed to handle permission request")
            return {"behavior": "deny", "message": f"Permission request failed: {e}"}
        else:
            return {"behavior": "deny", "message": "Unknown permission response"}


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Threading HTTP server for handling multiple requests."""

    daemon_threads = True
    allow_reuse_address = True


class PermissionMCPServer:
    """MCP server that handles permission requests for ACP tools.

    This server provides a single 'permission' tool that:
    1. Receives permission requests for other tools
    2. Shows UI dialogs via ACP client.request_permission()
    3. Returns allow/deny decisions back to the calling system
    """

    def __init__(self, client: Client, session_id: str):
        """Initialize permission MCP server.

        Args:
            client: ACP client for showing permission dialogs
            session_id: ACP session ID for permission requests
        """
        self.client = client
        self.session_id = session_id
        self.server: HTTPServer | None = None
        self.server_thread: Thread | None = None
        self._url: str | None = None
        self._port: int | None = None

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> tuple[str, int]:
        """Start the HTTP server for permission handling.

        Args:
            host: Host to bind to (default: localhost)
            port: Port to bind to (0 for automatic port selection)

        Returns:
            Tuple of (url, port) for the running server
        """

        # Create handler factory with client and session_id
        def handler_factory(*args: Any, **kwargs: Any) -> PermissionMCPHandler:
            return PermissionMCPHandler(self.client, self.session_id, *args, **kwargs)

        # Create and start server
        self.server = ThreadingHTTPServer((host, port), handler_factory)
        self._port = self.server.server_address[1]
        self._url = f"http://{host}:{self._port}"

        # Start server in background thread
        self.server_thread = Thread(
            target=self.server.serve_forever,
            name=f"PermissionMCPServer-{self.session_id}",
            daemon=True,
        )
        self.server_thread.start()
        msg = "Permission MCP server started for session %s at %s"
        logger.info(msg, self.session_id, self._url)
        return self._url, self._port

    async def stop(self) -> None:
        """Stop the permission MCP server."""
        if self.server:
            logger.info("Stopping permission MCP server for session %s", self.session_id)
            self.server.shutdown()
            self.server.server_close()

            if self.server_thread and self.server_thread.is_alive():
                self.server_thread.join(timeout=5.0)

            self.server = None
            self.server_thread = None
            self._url = None
            self._port = None

    @property
    def url(self) -> str | None:
        """Get the server URL if running."""
        return self._url

    @property
    def port(self) -> int | None:
        """Get the server port if running."""
        return self._port

    @property
    def is_running(self) -> bool:
        """Check if the server is currently running."""
        return (
            self.server is not None
            and self.server_thread is not None
            and self.server_thread.is_alive()
        )

    def get_mcp_server_config(self) -> dict[str, Any]:
        """Get MCP server configuration for session setup.

        Returns:
            MCP server configuration dict for use in newSession
        """
        if not self.is_running:
            msg = "Permission MCP server is not running"
            raise RuntimeError(msg)

        return {
            "name": "acp-permission",
            "type": "http",
            "url": f"{self._url}/mcp",
            "headers": {"x-acp-session-id": self.session_id},
        }
