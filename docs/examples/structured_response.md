# Structured Responses: Python vs YAML

This example demonstrates two ways to define structured responses in LLMling-agent:

- Using Python Pydantic models
- Using YAML response definitions
- Type validation and constraints
- Agent integration with structured outputs

## Configuration

Our `structured_agents.yml` defines response types and agents:

```yaml
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
```

## Implementation

Here's how to use both approaches:

```python
from pydantic import BaseModel
from llmling_agent import Agent
from llmling_agent.models.agents import AgentsManifest


class PythonResult(BaseModel):
    """Structured response defined in Python."""
    main_point: str
    is_positive: bool


async def example_structured_response():
    manifest = AgentsManifest[Any].from_yaml("structured_agents.yml")

    # Example 1: Python-defined structure
    async with Agent[Any].open_agent(
        manifest,
        "summarizer",
        result_type=PythonResult
    ) as summarizer:
        result = await summarizer.run("I love this new feature!")
        summary = result.data
        print("\nPython-defined Response:")
        print(f"Main point: {summary.main_point}")
        print(f"Is positive: {summary.is_positive}")

    # Example 2: YAML-defined structure
    # Note: For programmatic use, Python definitions are recommended
    async with Agent[Any].open_agent(manifest, "analyzer") as analyzer:
        result = await analyzer.run("I'm really excited about this project!")
        analysis = result.data
        print("\nYAML-defined Response:")
        # Type hints aren't available for dynamic models
        print(f"Sentiment: {analysis.sentiment}")  # type: ignore
        print(f"Confidence: {analysis.confidence:.2f}")  # type: ignore
        print(f"Mood: {analysis.mood}")  # type: ignore


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_structured_response())
```

## How It Works

1. Python-defined Responses:

- Use Pydantic models
- Full IDE support and type checking
- Best for programmatic use
- Inline field documentation

2. YAML-defined Responses:

- Define in configuration
- Include validation constraints
- Best for configuration-driven workflows
- Self-documenting fields

Example Output:
```
Python-defined Response:
Main point: User expresses enthusiasm for new feature
Is positive: true

YAML-defined Response:
Sentiment: positive
Confidence: 0.95
Mood: excited
```

This demonstrates:

- Two ways to define structured outputs
- Validation and constraints
- Integration with type system
- Trade-offs between approaches
