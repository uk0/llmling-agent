from typing import Any

from pydantic import BaseModel
import pytest
import yamling

from llmling_agent import Agent
from llmling_agent.models.agents import AgentsManifest


class Result(BaseModel):
    is_positive: bool


AGENT_CONFIG = """
agents:
    summarizer:
        model: openai:gpt-4o-mini
        system_prompts:
            - Summarize text in a structured way.
"""


@pytest.mark.asyncio
async def test_structured_response():
    manifest = AgentsManifest[Any].model_validate(yamling.load_yaml(AGENT_CONFIG))
    async with Agent[None].open_agent(manifest, "summarizer") as agent:
        result = await agent.run("I love this new feature!", result_type=Result)
        summary = result.data
        assert summary.is_positive
