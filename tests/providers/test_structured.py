from __future__ import annotations

from pydantic import BaseModel

from llmling_agent import AgentsManifest


class Result(BaseModel):
    """Structured response result."""

    is_positive: bool


AGENT_CONFIG = """
agents:
    summarizer:
        model: openai:gpt-5-nano
        system_prompts:
            - Summarize text in a structured way.
"""


async def test_structured_response():
    manifest = AgentsManifest.from_yaml(AGENT_CONFIG)
    async with manifest.pool as pool:
        agent = pool.get_agent("summarizer", return_type=Result)
        result = await agent.run("I love this new feature!")
        assert result.data.is_positive
