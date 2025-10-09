"""Terminal handle implementation. NOTE: not integrated yet."""

from __future__ import annotations

from typing import TYPE_CHECKING

from acp.schema import (
    KillTerminalCommandResponse,
    ReleaseTerminalResponse,
    TerminalOutputResponse,
    WaitForTerminalExitResponse,
)


if TYPE_CHECKING:
    from acp.connection import Connection


class TerminalHandle:
    """Handle for a terminal session."""

    def __init__(self, terminal_id: str, session_id: str, conn: Connection) -> None:
        self.id = terminal_id
        self._session_id = session_id
        self._conn = conn

    async def current_output(self) -> TerminalOutputResponse:
        dct = {"sessionId": self._session_id, "terminalId": self.id}
        resp = await self._conn.send_request("terminal/output", dct)
        return TerminalOutputResponse.model_validate(resp)

    async def wait_for_exit(self) -> WaitForTerminalExitResponse:
        dct = {"sessionId": self._session_id, "terminalId": self.id}
        method = "terminal/wait_for_exit"
        resp = await self._conn.send_request(method, dct)
        return WaitForTerminalExitResponse.model_validate(resp)

    async def kill(self) -> KillTerminalCommandResponse:
        dct = {"sessionId": self._session_id, "terminalId": self.id}
        resp = await self._conn.send_request("terminal/kill", dct)
        payload = resp if isinstance(resp, dict) else {}
        return KillTerminalCommandResponse.model_validate(payload)

    async def release(self) -> ReleaseTerminalResponse:
        dct = {"sessionId": self._session_id, "terminalId": self.id}
        resp = await self._conn.send_request("terminal/release", dct)
        payload = resp if isinstance(resp, dict) else {}
        return ReleaseTerminalResponse.model_validate(payload)
