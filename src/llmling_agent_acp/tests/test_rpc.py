from __future__ import annotations

import asyncio
import contextlib
import json
from typing import TYPE_CHECKING

import pytest

from acp import (
    Agent,
    AgentSideConnection,
    CancelNotification,
    Client,
    ClientSideConnection,
    InitializeRequest,
    InitializeResponse,
    NewSessionRequest,
    NewSessionResponse,
    PromptResponse,
    ReadTextFileRequest,
    ReadTextFileResponse,
    RequestPermissionResponse,
    SessionNotification,
    SetSessionModeRequest,
    WriteTextFileRequest,
)
from acp.schema import AgentMessageChunk, TextContentBlock, UserMessageChunk


if TYPE_CHECKING:
    from acp import (
        LoadSessionRequest,
        PromptRequest,
        RequestPermissionRequest,
    )

# --------------------- Test Utilities ---------------------


class _Server:
    def __init__(self) -> None:
        self._server: asyncio.AbstractServer | None = None
        self.server_reader: asyncio.StreamReader | None = None
        self.server_writer: asyncio.StreamWriter | None = None
        self.client_reader: asyncio.StreamReader | None = None
        self.client_writer: asyncio.StreamWriter | None = None

    async def __aenter__(self):
        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            self.server_reader = reader
            self.server_writer = writer

        self._server = await asyncio.start_server(handle, host="127.0.0.1", port=0)
        host, port = self._server.sockets[0].getsockname()[:2]
        self.client_reader, self.client_writer = await asyncio.open_connection(host, port)

        # wait until server side is set
        for _ in range(100):
            if self.server_reader and self.server_writer:
                break
            await asyncio.sleep(0.01)
        assert self.server_reader
        assert self.server_writer
        assert self.client_reader
        assert self.client_writer
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.client_writer:
            self.client_writer.close()
            with contextlib.suppress(Exception):
                await self.client_writer.wait_closed()
        if self.server_writer:
            self.server_writer.close()
            with contextlib.suppress(Exception):
                await self.server_writer.wait_closed()
        if self._server:
            self._server.close()
            await self._server.wait_closed()


# --------------------- Test Doubles -----------------------


class _TestClient(Client):
    __test__ = False  # prevent pytest from collecting this class

    def __init__(self) -> None:
        self.permission_outcomes: list[dict] = []
        self.files: dict[str, str] = {}
        self.notifications: list[SessionNotification] = []
        self.ext_calls: list[tuple[str, dict]] = []
        self.ext_notes: list[tuple[str, dict]] = []

    async def requestPermission(
        self, params: RequestPermissionRequest
    ) -> RequestPermissionResponse:
        outcome = (
            self.permission_outcomes.pop()
            if self.permission_outcomes
            else {"outcome": "cancelled"}
        )
        return RequestPermissionResponse.model_validate({"outcome": outcome})

    async def writeTextFile(self, params: WriteTextFileRequest) -> None:
        self.files[str(params.path)] = params.content

    async def readTextFile(self, params: ReadTextFileRequest) -> ReadTextFileResponse:
        content = self.files.get(str(params.path), "default content")
        return ReadTextFileResponse(content=content)

    async def sessionUpdate(self, params: SessionNotification) -> None:
        self.notifications.append(params)

    # Optional terminal methods (not implemented in this test client)
    async def createTerminal(self, params) -> None:  # pragma: no cover - placeholder
        pass

    async def terminalOutput(self, params) -> None:  # pragma: no cover - placeholder
        pass

    async def releaseTerminal(self, params) -> None:  # pragma: no cover - placeholder
        pass

    async def waitForTerminalExit(self, params) -> None:  # pragma: no cover - placeholder
        pass

    async def killTerminal(self, params) -> None:  # pragma: no cover - placeholder
        pass

    async def extMethod(self, method: str, params: dict) -> dict:
        self.ext_calls.append((method, params))
        return {"ok": True, "method": method}

    async def extNotification(self, method: str, params: dict) -> None:
        self.ext_notes.append((method, params))


class _TestAgent(Agent):
    __test__ = False  # prevent pytest from collecting this class

    def __init__(self) -> None:
        self.prompts: list[PromptRequest] = []
        self.cancellations: list[str] = []
        self.ext_calls: list[tuple[str, dict]] = []
        self.ext_notes: list[tuple[str, dict]] = []

    async def initialize(self, params: InitializeRequest) -> InitializeResponse:
        # Avoid serializer warnings by omitting defaults
        return InitializeResponse(
            protocol_version=params.protocol_version,
            agent_capabilities=None,
            auth_methods=[],
        )

    async def newSession(self, params: NewSessionRequest) -> NewSessionResponse:
        return NewSessionResponse(session_id="test-session-123")

    async def loadSession(self, params: LoadSessionRequest) -> None:
        return None

    async def authenticate(self, params) -> None:
        return None

    async def prompt(self, params: PromptRequest) -> PromptResponse:
        self.prompts.append(params)
        return PromptResponse(stop_reason="end_turn")

    async def cancel(self, params: CancelNotification) -> None:
        self.cancellations.append(params.session_id)

    async def setSessionMode(self, params):
        return {}

    async def extMethod(self, method: str, params: dict) -> dict:
        self.ext_calls.append((method, params))
        return {"ok": True, "method": method}

    async def extNotification(self, method: str, params: dict) -> None:
        self.ext_notes.append((method, params))


# ------------------------ Tests --------------------------


@pytest.mark.asyncio
async def test_initialize_and_new_session():
    async with _Server() as s:
        assert s.client_writer is not None
        assert s.client_reader is not None
        assert s.server_writer is not None
        assert s.server_reader is not None
        agent = _TestAgent()
        client = _TestClient()
        # server side is agent; client side is client
        agent_conn = ClientSideConnection(
            lambda _conn: client, s.client_writer, s.client_reader
        )
        _client_conn = AgentSideConnection(
            lambda _conn: agent, s.server_writer, s.server_reader
        )

        resp = await agent_conn.initialize(InitializeRequest(protocol_version=1))
        assert isinstance(resp, InitializeResponse)
        assert resp.protocol_version == 1

        new_sess = await agent_conn.newSession(
            NewSessionRequest(mcp_servers=[], cwd="/test")
        )
        assert new_sess.session_id == "test-session-123"


@pytest.mark.asyncio
async def test_bidirectional_file_ops():
    async with _Server() as s:
        assert s.client_writer is not None
        assert s.client_reader is not None
        assert s.server_writer is not None
        assert s.server_reader is not None
        agent = _TestAgent()
        client = _TestClient()
        client.files["/test/file.txt"] = "Hello, World!"
        _agent_conn = ClientSideConnection(
            lambda _conn: client, s.client_writer, s.client_reader
        )
        client_conn = AgentSideConnection(
            lambda _conn: agent, s.server_writer, s.server_reader
        )

        # Agent asks client to read
        res = await client_conn.readTextFile(
            ReadTextFileRequest(session_id="sess", path="/test/file.txt")
        )
        assert res.content == "Hello, World!"

        # Agent asks client to write
        await client_conn.writeTextFile(
            WriteTextFileRequest(
                session_id="sess", path="/test/file.txt", content="Updated"
            )
        )
        assert client.files["/test/file.txt"] == "Updated"


@pytest.mark.asyncio
async def test_cancel_notification_and_capture_wire():
    async with _Server() as s:
        assert s.client_writer is not None
        assert s.client_reader is not None
        assert s.server_writer is not None
        assert s.server_reader is not None
        # Build only agent-side (server) connection. Client side: reader to inspect wire
        agent = _TestAgent()
        client = _TestClient()
        agent_conn = ClientSideConnection(
            lambda _conn: client, s.client_writer, s.client_reader
        )
        _client_conn = AgentSideConnection(
            lambda _conn: agent, s.server_writer, s.server_reader
        )

        # Send cancel notification from client-side connection to agent
        await agent_conn.cancel(CancelNotification(session_id="test-123"))

        # Read raw line from server peer (it will be consumed by agent recv loop quickly).
        # Instead, wait a brief moment and assert agent recorded it.
        for _ in range(50):
            if agent.cancellations:
                break
            await asyncio.sleep(0.01)
        assert agent.cancellations == ["test-123"]


@pytest.mark.asyncio
async def test_session_notifications_flow():
    async with _Server() as s:
        agent = _TestAgent()
        client = _TestClient()
        assert s.client_writer is not None
        assert s.client_reader is not None
        assert s.server_writer is not None
        assert s.server_reader is not None
        _agent_conn = ClientSideConnection(
            lambda _conn: client, s.client_writer, s.client_reader
        )
        client_conn = AgentSideConnection(
            lambda _conn: agent, s.server_writer, s.server_reader
        )

        # Agent -> Client notifications
        content = TextContentBlock(text="Hello")
        await client_conn.sessionUpdate(
            SessionNotification(
                session_id="sess", update=AgentMessageChunk(content=content)
            )
        )
        content = TextContentBlock(text="World")
        await client_conn.sessionUpdate(
            SessionNotification(
                session_id="sess", update=UserMessageChunk(content=content)
            )
        )

        # Wait for async dispatch
        for _ in range(50):
            if len(client.notifications) >= 2:  # noqa: PLR2004
                break
            await asyncio.sleep(0.01)
        assert len(client.notifications) >= 2  # noqa: PLR2004
        assert client.notifications[0].session_id == "sess"


@pytest.mark.asyncio
async def test_concurrent_reads():
    async with _Server() as s:
        agent = _TestAgent()
        client = _TestClient()
        assert s.client_writer is not None
        assert s.client_reader is not None
        assert s.server_writer is not None
        assert s.server_reader is not None
        for i in range(5):
            client.files[f"/test/file{i}.txt"] = f"Content {i}"
        _agent_conn = ClientSideConnection(
            lambda _conn: client, s.client_writer, s.client_reader
        )
        client_conn = AgentSideConnection(
            lambda _conn: agent, s.server_writer, s.server_reader
        )

        async def read_one(i: int):
            return await client_conn.readTextFile(
                ReadTextFileRequest(session_id="sess", path=f"/test/file{i}.txt")
            )

        results = await asyncio.gather(*(read_one(i) for i in range(5)))
        for i, res in enumerate(results):
            assert res.content == f"Content {i}"


@pytest.mark.asyncio
async def test_invalid_params_results_in_error_response():
    async with _Server() as s:
        # Only start agent-side (server) so we can inject raw request from client socket
        agent = _TestAgent()
        assert s.client_writer is not None
        assert s.client_reader is not None
        assert s.server_writer is not None
        assert s.server_reader is not None
        _server_conn = AgentSideConnection(
            lambda _conn: agent, s.server_writer, s.server_reader
        )

        # Send initialize with wrong param type (protocolVersion should be int)
        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "oops"},
        }
        s.client_writer.write((json.dumps(req) + "\n").encode())
        await s.client_writer.drain()

        # Read response
        line = await asyncio.wait_for(s.client_reader.readline(), timeout=1)
        resp = json.loads(line)
        assert resp["id"] == 1
        assert "error" in resp
        invalid_params_code = -32602
        assert resp["error"]["code"] == invalid_params_code


@pytest.mark.asyncio
async def test_method_not_found_results_in_error_response():
    async with _Server() as s:
        assert s.client_writer is not None
        assert s.client_reader is not None
        assert s.server_writer is not None
        assert s.server_reader is not None
        agent = _TestAgent()
        _server_conn = AgentSideConnection(
            lambda _conn: agent, s.server_writer, s.server_reader
        )

        req = {"jsonrpc": "2.0", "id": 2, "method": "unknown/method", "params": {}}
        s.client_writer.write((json.dumps(req) + "\n").encode())
        await s.client_writer.drain()

        line = await asyncio.wait_for(s.client_reader.readline(), timeout=1)
        resp = json.loads(line)
        assert resp["id"] == 2  # noqa: PLR2004
        method_not_found_code = -32601
        assert resp["error"]["code"] == method_not_found_code


@pytest.mark.asyncio
async def test_set_session_mode_and_extensions():
    async with _Server() as s:
        assert s.client_writer is not None
        assert s.client_reader is not None
        assert s.server_writer is not None
        assert s.server_reader is not None
        agent = _TestAgent()
        client = _TestClient()
        agent_conn = ClientSideConnection(
            lambda _conn: client, s.client_writer, s.client_reader
        )
        _client_conn = AgentSideConnection(
            lambda _conn: agent, s.server_writer, s.server_reader
        )

        # setSessionMode
        resp = await agent_conn.setSessionMode(
            SetSessionModeRequest(session_id="sess", mode_id="yolo")
        )
        # Either empty object or typed response depending on implementation
        assert resp is None or resp.__class__.__name__ == "SetSessionModeResponse"

        # extMethod
        res = await agent_conn.extMethod("ping", {"x": 1})
        assert res.get("ok") is True

        # extNotification
        await agent_conn.extNotification("note", {"y": 2})
        # allow dispatch
        await asyncio.sleep(0.05)
        assert agent.ext_notes
        assert agent.ext_notes[-1][0] == "note"


@pytest.mark.asyncio
async def test_ignore_invalid_messages():
    async with _Server() as s:
        assert s.client_writer is not None
        assert s.client_reader is not None
        assert s.server_writer is not None
        assert s.server_reader is not None
        agent = _TestAgent()
        _server_conn = AgentSideConnection(
            lambda _conn: agent, s.server_writer, s.server_reader
        )

        # Message without id and method
        msg1 = {"jsonrpc": "2.0"}
        s.client_writer.write((json.dumps(msg1) + "\n").encode())
        await s.client_writer.drain()

        # Message without jsonrpc and without id/method
        msg2 = {"foo": "bar"}
        s.client_writer.write((json.dumps(msg2) + "\n").encode())
        await s.client_writer.drain()

        # Should not receive any response lines
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(s.client_reader.readline(), timeout=0.1)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
