# Getting Started

LLMling Agent is a framework for creating and managing LLM-powered agents. It integrates with LLMling's resource system and provides structured interactions with language models.

## Zero Configuration Usage

The fastest way to start is using the quickstart command which creates a temporary agent:

```bash
# Start an ephemeral chat session (requires uv)
uvx llmling-agent quickstart openai:gpt-4o-mini

# Enable streaming mode
uvx llmling-agent quickstart openai:gpt-4o-mini --stream
```

This creates a temporary agent with basic capabilities - no configuration needed! Just set your OpenAI API key:
```bash
export OPENAI_API_KEY=your-key-here
```

## Functional API

LLMling Agent provides both sync and async interfaces for quick model interactions:

```python
from llmling_agent.functional import run_with_model, run_with_model_sync

# Async usage
async def main():
    # Simple completion
    result = await run_with_model(
        "Analyze this text",
        model="openai:gpt-4o-mini"
    )
    print(result)

    # With structured output
    from pydantic import BaseModel

    class Analysis(BaseModel):
        summary: str
        sentiment: float

    result = await run_with_model(
        "Analyze the sentiment",
        model="openai:gpt-4o-mini",
        result_type=Analysis
    )
    print(f"Summary: {result.summary}")
    print(f"Sentiment: {result.sentiment}")

# Sync usage (convenience wrapper)
result = run_with_model_sync(
    "Quick question",
    model="openai:gpt-4o-mini"
)
```

## Configuration-Based Usage

For persistent agents, you can either use separate configuration files or inline configuration.

### Separate Configuration Files

Create two configuration files:
1. An agent configuration defining the agents and their behavior
2. An environment configuration defining available tools and resources (see [LLMling documentation](https://github.com/phil65/llmling))

Here's a minimal example:

```yaml
# agents.yml - Agent configuration
agents:
  system_checker:
    model: openai:gpt-4o-mini
    role: assistant
    environment: env_system.yml  # Reference to environment file
    system_prompts:
      - "You help users check their system status."
```

```yaml
# env_system.yml - Environment configuration (LLMling format)
tools:
  get_system_info:
    import_path: platform.platform
    description: "Get system platform information"
  get_memory:
    import_path: psutil.virtual_memory
    description: "Get memory usage information"
```

### Quick Start Configuration

Alternatively, use the init command for a working starter configuration:

```bash
# Create a basic agent configuration
llmling-agent init agents.yml

# Chat with the default agent
llmling-agent chat simple_agent
```

This creates a minimal working configuration:

```yaml
# agents.yml - Created by init command
agents:
  simple_agent:
    description: "Basic agent with minimal configuration"
    model: openai:gpt-4o-mini
    role: assistant
    environment:  # Inline environment configuration
      type: inline
      config:
        tools:
          open_webpage:
            import_path: webbrowser.open
            description: "Open URL in browser"
        resources:
          help_text:
            type: text
            content: "Basic help text for the agent"
    system_prompts:
      - "You are a helpful assistant."
```

The `init` command creates this basic setup that you can extend. You can:
- Define more agents in the same file
- Add tools and resources in separate environment files
- Configure structured responses
- Define custom roles and capabilities

> **Note**: For details about environment configuration (tools, resources, etc.),
> see the [LLMling documentation](https://github.com/phil65/llmling).

### Using Configured Agents

Once you have a configuration:

1. Add it to LLMling Agent:
```bash
llmling-agent add my-config agents.yml
```

2. Start chatting:
```bash
llmling-agent chat simple_agent  # or system_checker
```

3. Or use it in code:
```python
from llmling_agent import LLMlingAgent

async with LLMlingAgent.open_agent("agents.yml", "simple_agent") as agent:
    result = await agent.run("How can you help me?")
    print(result.data)
```

## Next Steps

- Learn about [Key Concepts](https://github.com/phil65/llmling-agent/blob/main/docs/key_concepts.md)
- Explore [Agent Configuration](https://github.com/phil65/llmling-agent/blob/main/docs/agent_config.md)
- Try the [Web Interface](https://github.com/phil65/llmling-agent/blob/main/docs/webui.md)
- See [Running Agents](https://github.com/phil65/llmling-agent/blob/main/docs/running_agents.md) for more usage patterns
- Check the [Command Reference](https://github.com/phil65/llmling-agent/blob/main/docs/commands.md) for CLI options

> **Note**: Make sure you have the required API keys set up for your chosen models.
> See our [configuration guide](https://github.com/phil65/llmling-agent/blob/main/docs/agent_config.md#model-configuration) for details.
