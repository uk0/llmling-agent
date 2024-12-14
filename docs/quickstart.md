# Getting Started

LLMling Agent is a framework for creating and managing LLM-powered agents. It integrates with LLMling's resource system and provides structured interactions with language models.

## Quick Start

### Creating Your First Configuration

LLMling Agent provides commands to help you create your first configuration:

```bash
# Create a basic agent configuration
llmling-agent init agents.yml

# Or use the interactive wizard for more options (EXPERIMENTAL)
llmling-agent init agents.yml --interactive
```

The basic configuration includes:
- A simple agent with common tools
- Example system prompts
- Basic environment configuration

You can use this as a starting point and customize it for your needs.


### Basic Usage

The simplest way to use LLMling Agent is through its command-line interface:

```bash
# Start an interactive chat with an agent
llmling-agent chat my-agent

# Run an agent with a specific prompt
llmling-agent run my-agent "What is the current system status?"
```

### First Agent Configuration

Agents are defined in YAML configuration files. You'll need two files:
1. An agent configuration defining the agents and their behavior
2. An environment configuration defining available tools and resources (see [LLMling documentation](https://llmling.readthedocs.io) for details)

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

### Running Your First Agent

1. Save both configuration files:
   - `agents.yml` - Agent configuration
   - `env_system.yml` - Environment configuration

2. Add the agent configuration to LLMling Agent:
```bash
llmling-agent add my-config agents.yml
```

3. Start chatting with your agent:
```bash
llmling-agent chat system_checker
```

4. Or run it programmatically:
```python
from llmling_agent import LLMlingAgent

async with LLMlingAgent.open_agent("agents.yml", "system_checker") as agent:
    result = await agent.run("How much memory is available?")
    print(result.data)
```
