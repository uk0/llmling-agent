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

### Installation

```bash
pip install llmling-agent
```

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
    model: openai:gpt-3.5-turbo
    role: assistant
    environment: env_system.yml  # Reference to environment file
    system_prompts:
      - "You help users check their system status."
    user_prompts:
      - "What is the current system status?"
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

resources:
  system_template:
    type: text
    content: |
      System Status Report:
      Platform: {platform}
      Memory: {memory}
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

Key features shown here:
- Easy configuration through YAML files
- Separation of agent logic and environment setup
- Integration with LLMling's resource system
- Command-line interface for quick interactions
- Programmatic API for integration
- Built-in tool system for system interactions
- Support for both chat and single-query modes
