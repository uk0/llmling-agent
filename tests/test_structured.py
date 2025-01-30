from pydantic import BaseModel
import pytest

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
    manifest = AgentsManifest.from_yaml(AGENT_CONFIG)
    async with manifest.pool as pool:
        agent = pool.get_agent("summarizer", return_type=Result)
        result = await agent.run("I love this new feature!")
        assert result.data.is_positive
