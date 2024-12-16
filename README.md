# LLMling-Agent

[![PyPI License](https://img.shields.io/pypi/l/llmling-agent.svg)](https://pypi.org/project/llmling-agent/)
[![Package status](https://img.shields.io/pypi/status/llmling-agent.svg)](https://pypi.org/project/llmling-agent/)
[![Daily downloads](https://img.shields.io/pypi/dd/llmling-agent.svg)](https://pypi.org/project/llmling-agent/)
[![Weekly downloads](https://img.shields.io/pypi/dw/llmling-agent.svg)](https://pypi.org/project/llmling-agent/)
[![Monthly downloads](https://img.shields.io/pypi/dm/llmling-agent.svg)](https://pypi.org/project/llmling-agent/)
[![Distribution format](https://img.shields.io/pypi/format/llmling-agent.svg)](https://pypi.org/project/llmling-agent/)
[![Wheel availability](https://img.shields.io/pypi/wheel/llmling-agent.svg)](https://pypi.org/project/llmling-agent/)
[![Python version](https://img.shields.io/pypi/pyversions/llmling-agent.svg)](https://pypi.org/project/llmling-agent/)
[![Implementation](https://img.shields.io/pypi/implementation/llmling-agent.svg)](https://pypi.org/project/llmling-agent/)
[![Releases](https://img.shields.io/github/downloads/phil65/llmling-agent/total.svg)](https://github.com/phil65/llmling-agent/releases)
[![Github Contributors](https://img.shields.io/github/contributors/phil65/llmling-agent)](https://github.com/phil65/llmling-agent/graphs/contributors)
[![Github Discussions](https://img.shields.io/github/discussions/phil65/llmling-agent)](https://github.com/phil65/llmling-agent/discussions)
[![Github Forks](https://img.shields.io/github/forks/phil65/llmling-agent)](https://github.com/phil65/llmling-agent/forks)
[![Github Issues](https://img.shields.io/github/issues/phil65/llmling-agent)](https://github.com/phil65/llmling-agent/issues)
[![Github Issues](https://img.shields.io/github/issues-pr/phil65/llmling-agent)](https://github.com/phil65/llmling-agent/pulls)
[![Github Watchers](https://img.shields.io/github/watchers/phil65/llmling-agent)](https://github.com/phil65/llmling-agent/watchers)
[![Github Stars](https://img.shields.io/github/stars/phil65/llmling-agent)](https://github.com/phil65/llmling-agent/stars)
[![Github Repository size](https://img.shields.io/github/repo-size/phil65/llmling-agent)](https://github.com/phil65/llmling-agent)
[![Github last commit](https://img.shields.io/github/last-commit/phil65/llmling-agent)](https://github.com/phil65/llmling-agent/commits)
[![Github release date](https://img.shields.io/github/release-date/phil65/llmling-agent)](https://github.com/phil65/llmling-agent/releases)
[![Github language count](https://img.shields.io/github/languages/count/phil65/llmling-agent)](https://github.com/phil65/llmling-agent)
[![Github commits this week](https://img.shields.io/github/commit-activity/w/phil65/llmling-agent)](https://github.com/phil65/llmling-agent)
[![Github commits this month](https://img.shields.io/github/commit-activity/m/phil65/llmling-agent)](https://github.com/phil65/llmling-agent)
[![Github commits this year](https://img.shields.io/github/commit-activity/y/phil65/llmling-agent)](https://github.com/phil65/llmling-agent)
[![Package status](https://codecov.io/gh/phil65/llmling-agent/branch/main/graph/badge.svg)](https://codecov.io/gh/phil65/llmling-agent/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyUp](https://pyup.io/repos/github/phil65/llmling-agent/shield.svg)](https://pyup.io/repos/github/phil65/llmling-agent/)

[Read the documentation!](https://phil65.github.io/llmling-agent/)

# Getting Started

LLMling Agent is a framework for creating and managing LLM-powered agents. It integrates with LLMling's resource system and provides structured interactions with language models.


## Quick Start

The fastest way to start chatting with an AI:
```bash
# Start an ephemeral chat session (requires uv)
uvx llmling-agent quickstart openai:gpt-4o-mini
```

This creates a temporary agent ready for chat - no configuration needed!
LLMling-Agent is Pydantic-ai based, so all pydantic-ai models can be used.
The according API keys need to be set as environment variables.

For persistent agents, you can use:

```bash
# Create a basic agent configuration
llmling-agent init agents.yml

# Or use the interactive wizard (EXPERIMENTAL)
llmling-agent init agents.yml --interactive
```

This creates a basic agent configuration file that you can customize. The interactive mode will guide you through setting up your agents.

### Basic Usage

The simplest way to use LLMling Agent is through its command-line interface:

```bash
# Start an interactive chat with an agent
llmling-agent chat my-agent

# Run an agent with a specific prompt
llmling-agent run my-agent "What is the current system status?"
```

## Features

### Dynamic Environment

LLMling Agent allows the AI to modify its own environment (when permitted):
- Register new tools on the fly
- Load and analyze resources
- Install Python packages
- Create new tools from code

These capabilities can be controlled via roles and permissions to ensure safe operation.

### Interactive Chat Sessions

The chat interface provides rich features:
```bash
# Start a chat session
llmling-agent chat my-agent

# Available during chat:
/list-tools              # See available tools
/register-tool os.getcwd # Add new tools on the fly
/list-resources         # View accessible resources
/show-resource config   # Examine resource content
/enable-tool tool_name  # Enable/disable tools
/set-model gpt-4       # Switch models mid-conversation
```

### Safe and Configurable

- Fine-grained capability control (resource access, tool registration, etc.)
- Role-based permissions (overseer, specialist, assistant)
- Tool confirmation for sensitive operations
- Command history and usage statistics


### First Agent Configuration

Agents are defined in YAML configuration files. The environment (tools and resources) can be configured either inline or in a separate file:
 (see [LLMling documentation](https://github.com/phil65/llmling) for YAML details)

#### Option 1: Separate Environment File

```yaml
# agents.yml - Agent configuration
agents:
  system_checker:
    model: openai:gpt-4o-mini
    role: assistant
    environment: env_system.yml  # Reference to environment file
    system_prompts:
      - "You help users check their system status."

# env_system.yml - Environment configuration (LLMling format)
tools:
  get_system_info:
    import_path: platform.platform
    description: "Get system platform information"
  get_memory:
    import_path: psutil.virtual_memory
    description: "Get memory usage information"
```

#### Option 2: Inline Environment

```yaml
# agents.yml - Complete configuration
agents:
  system_checker:
    model: openai:gpt-4o-mini
    role: assistant
    environment:  # Inline environment configuration
      type: inline
      config:
        tools:
          get_system_info:
            import_path: platform.platform
            description: "Get system platform information"
          get_memory:
            import_path: psutil.virtual_memory
            description: "Get memory usage information"
    system_prompts:
      - "You help users check their system status."
```

Both approaches are equivalent - choose what works best for your use case:
- **Separate files**: Better for reusing environments across agents or when configurations are large
- **Inline configuration**: Simpler for small configurations or self-contained agents


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

### Conversation History and Analytics

LLMling Agent provides built-in conversation tracking and analysis:

```bash
# View recent conversations
llmling-agent history show
llmling-agent history show --period 24h  # Last 24 hours
llmling-agent history show --query "database"  # Search content

# View usage statistics
llmling-agent history stats  # Basic stats
llmling-agent history stats --group-by model  # Model usage
llmling-agent history stats --group-by day    # Daily breakdown
```
