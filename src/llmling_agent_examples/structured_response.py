"""Example of structured responses defined both in code and YAML."""

from typing import Any

from pydantic import BaseModel

from llmling_agent import Agent
from llmling_agent.models.agents import AgentsManifest


class PythonResult(BaseModel):
    """Structured response defined in Python."""

    main_point: str
    is_positive: bool


AGENT_CONFIG = """
# Define response type in YAML
responses:
    YamlResult:
        type: inline
        description: "Sentiment analysis result"
        fields:
            sentiment:
                type: str
                description: "Overall sentiment"
            confidence:
                type: float
                description: "Confidence score"
            mood:
                type: str
                description: "Detected mood"
                constraints:
                    min_length: 3
                    max_length: 20

agents:
    summarizer:
        model: openai:gpt-4o-mini
        system_prompts:
            - Summarize text in a structured way.

    analyzer:
        model: openai:gpt-4o-mini
        system_prompts:
            - Analyze text sentiment and mood.
        # Use YAML-defined response type
        result_type: YamlResult
"""


async def example_structured_response():
    """Show both ways of defining structured responses."""
    manifest = AgentsManifest[Any].from_yaml(AGENT_CONFIG)

    # Example 1: Python-defined structure
    async with Agent[Any].open_agent(
        manifest, "summarizer", result_type=PythonResult
    ) as summarizer:
        result = await summarizer.run("I love this new feature!")
        summary = result.data
        print("\nPython-defined Response:")
        print(f"Main point: {summary.main_point}")
        print(f"Is positive: {summary.is_positive}")

    # Example 2: YAML-defined structure
    # NOTE: this is not recommended for programmatic usage and is just a demo. Use this
    # only for complete YAML workflows, otherwise your linter wont like what you are doin.
    async with Agent[Any].open_agent(manifest, "analyzer") as analyzer:
        result = await analyzer.run("I'm really excited about this project!")
        analysis = result.data
        print("\nYAML-defined Response:")
        # Type checkers cant deal with dynamically generated Models, so we have to
        # git-ignore
        print(f"Sentiment: {analysis.sentiment}")  # type: ignore
        print(f"Confidence: {analysis.confidence:.2f}")  # type: ignore
        print(f"Mood: {analysis.mood}")  # type: ignore


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_structured_response())
