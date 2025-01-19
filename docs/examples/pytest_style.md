# PyTest-Style Agent Functions

This example demonstrates a pytest-inspired way to work with agents:

- Using agents as function decorators
- Automatic function discovery
- Dependency injection
- Execution order control
- Function result handling

## Configuration

Our `pytest_agents.yml` defines two agents for analysis and writing:

```yaml
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
```

## Implementation

Here's how we use the pytest-style decorators:

```python
from llmling_agent.agent import Agent
from llmling_agent.running import agent_function, run_agents_async

# Sample data to analyze
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
    results = await run_agents_async("pytest_agents.yml", parallel=True)
    print("Analysis:", results["analyze_data"])
    print("Summary:", results["summarize_analysis"])


if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
```

## How It Works

1. Functions are decorated with `@agent_function`
2. Type hints specify which agent to inject (`analyzer: Agent`)
3. Dependencies are declared in the decorator (`depends_on="analyze_data"`)
4. Results from one function can be injected into another
5. All functions are discovered and executed in the correct order

Key Features:

- Automatic agent injection based on type hints
- Function dependency resolution
- Parallel execution where possible
- Results passed automatically between functions

This provides a clean, declarative way to orchestrate multi-agent workflows, similar to how pytest fixtures work.
