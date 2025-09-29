"""Test fixtures for ACP tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from acp import (
    Agent,
    DefaultACPClient,
    InitializeResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PromptResponse,
)


if TYPE_CHECKING:
    from acp import (
        InitializeRequest,
        LoadSessionRequest,
        NewSessionRequest,
        PromptRequest,
    )
    from acp.schema import CancelNotification


class TestAgent(Agent):
    """Test agent implementation for ACP testing."""

    def __init__(self) -> None:
        self.prompts: list[PromptRequest] = []
        self.cancellations: list[str] = []
        self.ext_calls: list[tuple[str, dict]] = []
        self.ext_notes: list[tuple[str, dict]] = []

    async def initialize(self, params: InitializeRequest) -> InitializeResponse:
        return InitializeResponse(
            protocol_version=params.protocol_version,
            agent_capabilities=None,
        )

    async def new_session(self, params: NewSessionRequest) -> NewSessionResponse:
        return NewSessionResponse(session_id="test-session-123")

    async def load_session(self, params: LoadSessionRequest) -> LoadSessionResponse:
        return LoadSessionResponse()

    async def authenticate(self, params) -> None:
        return None

    async def prompt(self, params: PromptRequest) -> PromptResponse:
        self.prompts.append(params)
        return PromptResponse(stop_reason="end_turn")

    async def cancel(self, params: CancelNotification) -> None:
        self.cancellations.append(params.session_id)

    async def set_session_mode(self, params):
        return {}

    async def set_session_model(self, params):
        return {}

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.ext_calls.append((method, params))
        return {"ok": True, "method": method}

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        self.ext_notes.append((method, params))


@pytest.fixture
def test_client() -> DefaultACPClient:
    """Create a fresh test client for each test."""
    return DefaultACPClient(allow_file_operations=True, use_real_files=False)


@pytest.fixture
def test_agent() -> TestAgent:
    """Create a fresh test agent for each test."""
    return TestAgent()
