"""Integration tests for agent pipeline functions."""

from __future__ import annotations

import json
from typing import Any

from pydantic_ai.models.test import TestModel
import pytest

from llmling_agent.functional import run_agent_pipeline, run_agent_pipeline_sync
from llmling_agent.models import (
    AgentConfig,
    AgentsManifest,
    InlineResponseDefinition,
    ResponseField,
    SystemPrompt,
)


@pytest.fixture
def pipeline_manifest(test_model: Any) -> AgentsManifest:
    """Create test manifest for pipeline testing."""
    # Create basic response type for simple text responses
    basic_response = InlineResponseDefinition(
        description="Basic test result",
        fields={"message": ResponseField(type="str", description="Test message")},
    )

    # Create structured response type
    structured_response = InlineResponseDefinition(
        description="Structured test result",
        fields={
            "success": ResponseField(type="bool", description="Operation success"),
            "data": ResponseField(type="str", description="Result data"),
            "score": ResponseField(type="int", description="Numeric score"),
        },
    )

    # Create a separate TestModel for structured responses
    structured_model = TestModel(
        custom_result_text=json.dumps({"success": True, "data": "Test data", "score": 42})
    )

    return AgentsManifest(
        responses={
            "BasicResult": basic_response,
            "StructuredResult": structured_response,
        },
        agents={
            "test_agent": AgentConfig(
                name="test_agent",
                model=test_model,  # Regular TestModel for text responses
                result_type="BasicResult",
                system_prompts=["You are a test agent"],
            ),
            "structured_agent": AgentConfig(
                name="structured_agent",
                model=structured_model,  # Use structured TestModel
                result_type="StructuredResult",
                system_prompts=["You provide structured responses"],
            ),
        },
    )


@pytest.mark.asyncio
class TestAgentPipeline:
    """Test async pipeline functionality."""

    async def test_simple_prompt(self, pipeline_manifest: AgentsManifest) -> None:
        """Test basic prompt execution."""
        result = await run_agent_pipeline(
            "test_agent",
            "Hello!",
            pipeline_manifest,
            output_format="text",
        )
        assert isinstance(result, str)
        assert result == "Test response"  # From TestModel fixture

    async def test_system_prompt(self, pipeline_manifest: AgentsManifest) -> None:
        """Test using SystemPrompt."""
        prompt = SystemPrompt(type="text", value="Test instruction")
        result = await run_agent_pipeline(
            "test_agent",
            prompt,
            pipeline_manifest,
            output_format="text",
        )
        assert result == "Test response"

    async def test_multiple_prompts(self, pipeline_manifest: AgentsManifest) -> None:
        """Test sequence of prompts."""
        result = await run_agent_pipeline(
            "test_agent",
            ["First", "Second"],
            pipeline_manifest,
            output_format="text",
        )
        assert result == "Test response"  # Last response

    async def test_structured_output(self, pipeline_manifest: AgentsManifest) -> None:
        """Test structured response handling."""
        result = await run_agent_pipeline(
            "structured_agent",
            "Process this",
            pipeline_manifest,
            output_format="json",
        )
        assert "success" in result
        assert "data" in result
        assert "score" in result

    async def test_streaming(self, pipeline_manifest: AgentsManifest) -> None:
        """Test streaming mode."""
        chunks = [
            chunk
            async for chunk in await run_agent_pipeline(
                "test_agent",
                "Hello!",
                pipeline_manifest,
                stream=True,
            )
        ]
        assert chunks  # Should have received chunks
        assert "".join(chunks) == "Test response"

    async def test_error_handling_raise(self, pipeline_manifest: AgentsManifest) -> None:
        """Test error handling with raise mode."""
        with pytest.raises(ValueError):  # noqa: PT011
            await run_agent_pipeline(
                "nonexistent_agent",
                "Hello!",
                pipeline_manifest,
                error_handling="raise",
            )

    async def test_error_handling_return(self, pipeline_manifest: AgentsManifest) -> None:
        """Test error handling with return mode."""
        result: Any = await run_agent_pipeline(
            "nonexistent_agent",
            "Hello!",
            pipeline_manifest,
            error_handling="return",
        )
        assert isinstance(result, str)
        assert result.startswith("Error:")

    async def test_with_capabilities(self, pipeline_manifest: AgentsManifest) -> None:
        """Test with capability overrides."""
        result = await run_agent_pipeline(
            "test_agent",
            "Hello!",
            pipeline_manifest,
            capabilities={"can_delegate_tasks": True},
            output_format="text",
        )
        assert result == "Test response"

    async def test_tool_control(self, pipeline_manifest: AgentsManifest) -> None:
        """Test tool control options."""
        # Test with specific tool enabled
        result = await run_agent_pipeline(
            "test_agent",
            "Hello!",
            pipeline_manifest,
            tool_choice="some_tool",
            output_format="text",
        )
        assert result == "Test response"

        # Test with tools disabled
        result = await run_agent_pipeline(
            "test_agent",
            "Hello!",
            pipeline_manifest,
            tool_choice=False,
            output_format="text",
        )
        assert result == "Test response"


def test_sync_pipeline(pipeline_manifest: AgentsManifest) -> None:
    """Test synchronous pipeline version."""
    result = run_agent_pipeline_sync(
        "test_agent",
        "Hello!",
        pipeline_manifest,
        output_format="text",
    )
    assert result == "Test response"


def test_sync_pipeline_streaming_error(pipeline_manifest: AgentsManifest) -> None:
    """Test that streaming raises error in sync version."""
    with pytest.raises(ValueError, match="Streaming not supported"):
        run_agent_pipeline_sync(
            "test_agent",
            "Hello!",
            pipeline_manifest,
            stream=True,
        )
