"""Example showing agent function discovery and execution.

This example demonstrates:
- Using agents as function decorators
- Automatic function discovery
- Dependency injection
- Execution order control
- Function result handling
"""

from llmling_agent.agent import Agent
from llmling_agent.running import agent_function


AGENT_CONFIG = """
agents:
  analyzer:
    name: Data Analyzer
    model: openai:gpt-4o-mini
    system_prompts:
      - You are a data analyst specializing in business metrics.
      - Focus on key trends, patterns, and notable changes.

  writer:
    name: Technical Writer
    model: openai:gpt-4o-mini
    system_prompts:
      - You are a business writer creating clear executive summaries.
      - Focus on actionable insights and bottom-line impact.
"""

DATA = """
Monthly Sales Data (2023):
Jan: $12,500
Feb: $15,300
Mar: $18,900
Apr: $14,200
May: $16,800
Jun: $21,500
"""


@agent_function
async def analyze_data(analyzer: Agent):
    """First step: Analyze the data."""
    result = await analyzer.run(f"Analyze this sales data and identify trends:\n{DATA}")
    return result.data


@agent_function(depends_on="analyze_data")
async def summarize_analysis(writer: Agent, analyze_data: str):
    """Second step: Create an executive summary."""
    prompt = "Create a brief executive summary of this sales analysis:\n{analyze_data}"
    result = await writer.run(prompt)
    return result.data


async def run():
    from llmling_agent.running import run_agents_async

    results = await run_agents_async(config_path, parallel=True)
    print("Analysis:", results["analyze_data"])
    print("Summary:", results["summarize_analysis"])


if __name__ == "__main__":
    import asyncio
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp:
        tmp.write(AGENT_CONFIG)
        tmp.flush()
        config_path = tmp.name
        asyncio.run(run())
