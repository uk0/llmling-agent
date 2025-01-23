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

### [Read the documentation! Unlike other frameworks, it doesnt mainly consist of "tools" ;)](https://phil65.github.io/llmling-agent/)

# 🚀 Getting Started

LLMling Agent is a framework for creating and managing LLM-powered agents. It integrates with LLMling's resource system and provides structured interactions with language models.

## ✨ Unique Features
- 🔄 Modern python written from ground up with Python 3.12
- ⚡ True async framework. Easy set-up of complex async agent flows. Faster initializations of "heavy" agents (and first experimentations with async "UI", supervision of Agents in real-time)
- 📝 Easy consistent APIs
- 🛡️ Complete agent defintion via YAML files including extensive JSON schema to help with creating configurations.
- 🔒 Leveraging the complete pydantic-based type-safe stack and bringing it to the multi-agent world
- 🎮 Complete integrated command sytem to control agents from prompt-based interfaces
- 🔌 Agent MCP server support, initialized when entering the async context.
- 👁️ Multi-modal support for both LiteLLM and Pydantic-AI (currently Images only)
- 💾 Storage providers to allow writing to local files, databases, etc. with many customizable backends. Log to SQL databases and pretty-print to a file according to your own wishes.
- 🧩 Support for creating prompts for many common python type(s / instances). Your agent understands common datatypes.
- 🎯 Integration of Meta-Model system based on [LLMling-models](https://github.com/phil65/llmling-models), also configurable via YAML.
- 🔐 Structured responses. With pydantic-AI at its core, the Agents allow injecting dependencies as well as defining a return type while keeping type safety.
- 📋 Response type definition via YAML. Structured response Agents can be defined in the agent config.
- 🛡️ Capabilites system allowing runtime modifications and "special" commands (on-the-fly agent generation, history lookups)
- 📊 Complete database logging of Agent interactions including easy recovery based on query parameters.
- ⚙️ pytest-inspired way to create agents from YAML in a type-safe manner. "Auto-populated signatures."
- 🛜 Comletely UPath backed. Any file operations under our control is routed through fsspec to allow referencing remote sourcces.
- 📕 Integrated prompt management system.
- 🔧 Tasks, tools, and what else you can expect from an Agent framework.
- 👥 Easy human-in-the-loop interactions on multiple levels (complete "providers" or model-based, see llmling-models)
- 💻 A CLI application with extensive slash command support to build agent flows interactively. Set up message connections via commands.

## 🔜 Coming Soon
- 🎯 Built-in event system for reactive agent behaviors (file changes, webhooks, timed events)
- 🖥️ Real-time-monitoring via Textual app in truly async manner. Talk to your agents while they are working and monitor the progress!




### Why LLMling-agent? 🤔

Why another framework you may ask? The framework stands out through three core principles:


#### 🛡️ Type Safety and Structure
Unlike other frameworks that rely on free-form text exchanges, LLMling-agent enforces type safety throughout the entire agent interaction chain. From input validation to structured outputs, every data flow is typed and validated, making it significantly more reliable for production systems.

#### ⚙️ Rich Configuration System
While other frameworks require extensive Python code for setup, LLMling-agent introduces a comprehensive YAML configuration system. This allows defining complex agent behaviors, capabilities, and interactions declaratively. The configuration supports inheritance, composition, and strong validation, making it easier to manage large-scale agent deployments.

#### 🤝 Human-AI Collaboration
Instead of choosing between fully autonomous or human-controlled operations, LLMling-agent offers flexible human-in-the-loop integration. From full human control to selective oversight of critical actions, the framework makes it natural to build systems that combine AI capabilities with human expertise.


### Comparison with Other Frameworks

**AutoGen** focuses on autonomous multi-agent conversations, making it great for research and exploration but less suited for production systems that need strict controls and validation.

**CrewAI** emphasizes sequential task execution with role-based agents, providing good structure but less flexibility in agent interactions and limited type safety.

**LLMling-agent** takes the best from both:
- AutoGen's flexible agent communication patterns
- CrewAI's structured task execution
- And adds:
  - End-to-end type safety leveraging the whole pydantic-stack
  - Rich YAML configuration which goes way beyond what CrewAI offers
  - Human oversight capabilities in many different forms


## Quick Start

The fastest way to start chatting with an AI:
```bash
# Start an ephemeral chat session (requires uv)
uvx llmling-agent quickstart openai:gpt-4o-mini
```

This creates a temporary agent ready for chat - no configuration needed!
The according API keys need to be set as environment variables.


## 🚀 Quick Examples

Three ways to create a simple agent setup that turns topics into Wikipedia explorations:


### Python Version
```python
from llmling_agent import AgentPool

async def main():
    async with AgentPool() as pool:
        # Create browser assistant
        browser = await pool.add_agent(
            "browser",
            system_prompt="Open Wikipedia pages matching the topics you receive.",
            model="openai:gpt-4o-mini",
            tools=["webbrowser.open"],
        )
        # Create main agent and connect
        agent = await pool.add_agent("assistant")
        connection = agent >> browser  # this sets up a permanent connection.
        await agent.run("What's your favorite holiday destination?")
```

### YAML Version
```yaml
# agents.yml
agents:
  browser:
    model: openai:gpt-4o-mini
    system_prompts:
      - "Open Wikipedia pages matching the topics you receive."
    environment:
      tools:
        open_url:
          import_path: webbrowser.open

  assistant:
    model: openai:gpt-4o-mini
    connections:  # this forwards any output to the 2nd agent
      - type: agent
        name: browser
```

```bash
llmling-agent run --config agents.yml "whats your favourite holiday destination?"
> What's your favorite holiday destination?
```


### CLI Version (Interactive using slash command system)
```bash
# Start session
llmling-agent quickstart --model openai:gpt-4o-mini
# Create browser assistant
/create-agent browser --system-prompt "Open Wikipedia pages matching the topics you receive." --tools webbrowser.open
# Connect and try it
/connect browser
> What's your favorite holiday destination?
```


### First Agent Configuration

Agents can be defined in YAML configuration files.
The YAML setup is extensive, you can define every detail of your agent:
Connections, routing, tools, MCP servers, resources, system prompts, and much much more.
Check out the Manual for the whole configuration!


```yaml
# agents.yml - Complete configuration
agents:
  analyzer:  # Agent name (key in agents dict)
    # Basic configuration
    type: "pydantic_ai"  # "pydantic_ai" | "human" | "litellm" | custom provider config
    name: "analyzer"  # Optional override for agent name
    inherits: "base_agent"  # Optional parent config to inherit from
    description: "Code analysis specialist"
    model: "openai:gpt-4"  # or structured model definition
    debug: false

    # Provider behavior
    retries: 1
    end_strategy: "early"  # "early" | "complete" | "confirm"

    # Structured output
    result_type:
      type: "inline"  # or "import" for Python types
      fields:
        success:
          type: "bool"
          description: "Whether analysis succeeded"
    result_tool_name: "final_result"  # Name for result validation tool
    result_tool_description: "Create final response"  # Optional description
    result_retries: 3  # Validation retry count

    # Agent behavior
    system_prompts: ["You are a code analyzer..."]
    user_prompts: ["Example query..."]  # Default queries
    model_settings: {}  # Additional model parameters

    # State management
    session:                 # Initial session loading
      name: my_session       # Optional session identifier
      since: 1h             # Only messages from last hour
    avatar: "path/to/avatar.png"  # Optional UI avatar

    # Capabilities
    capabilities:
      can_delegate_tasks: true
      can_load_resources: true
      history_access: "own"  # "none" | "own" | "all"
      # ... other capability settings

    # Environment & Resources
    environment:
      type: "file"  # or "inline"
      uri: "environments/analyzer.yml"

    # Knowledge configuration
    knowledge:
      paths: ["docs/**/*.md"]
      resources:
        - type: "repository"
          url: "https://github.com/user/repo"
      prompts:
        - type: "file"
          path: "prompts/analysis.txt"

    # MCP integration
    mcp_servers:
      - type: "stdio"
        command: "python"
        args: ["-m", "mcp_server"]
      - "python -m other_server"  # shorthand syntax

    # Agent relationships
    workers:
      - name: "formatter"
        reset_history_on_run: true
        pass_message_history: false
        share_context: false
      - "linter"  # shorthand syntax

    # Message routing
    connections:
      - type: "agent"
        name: "reporter"
        connection_type: "run"  # "run" | "context" | "forward"
        wait_for_completion: true

    # Event handling
    triggers:
      - type: "file"
        name: "code_change"
        paths: ["src/**/*.py"]
        extensions: [".py"]
        recursive: true

  # Additional agents...
```

### Running Your First Agent

1. Save configuration file:
   - `agents.yml` - Agent configuration


2. Start chatting with your agent:
```bash
llmling-agent chat --config agents.yml system_checker
```

3. Or run it programmatically (in general, pool usage is recommended though, see the following section):

```python
from llmling_agent import Agent

async with Agent.open_agent("agents.yml", "system_checker") as agent:
    result = await agent.run("How much memory is available?")
    print(result.data)
```

### Agent Pool: Multi-Agent Coordination

The `AgentPool` allows multiple agents to work together on tasks. Here's a practical example of parallel file downloading:

```python
# agents.yml
agents:
  file_getter_1:
    name: "File Downloader 1" # Agent name (can be anything unique)
    description: "Downloads files from URLs"
    model: openai:gpt-4o-mini  # Language model to use, takes pydantic-ai model names
    environment: # Environment configuration (can also be external YAML file)
      type: inline
      tools:
        download_file:  # Simple httpx-based download utility
        import_path: llmling_agent_tools.download_file
        description: "Download file from URL to local path"
    system_prompts:
      - |
        You are a download specialist. Just use the download_file tool
        and report its results. No explanations needed.

  file_getter_2:  # Same configuration as file_getter_1
    ... # ... (identical config to file_getter_1, omitting for brevity)

  overseer:
    name: "Download Coordinator"
    description: "Coordinates parallel downloads"
    model: openai:gpt-4o-mini
    system_prompts:
      - |
        You coordinate downloads by delegating to file_getter_1 and file_getter_2.
        Just delegate tasks and report results concisely. No explanations needed.

```

```python
from llmling_agent.delegation import AgentPool

async def main():
    async with AgentPool("agents.yml") as pool:
        # Run downloads in parallel (sequential mode also available)
        team = pool.create_team(["file_getter_1", "file_getter_2"])
        responses = await team.run_parallel("Download https://example.com/file.zip")

        # Or let a coordinator orchestrate
        coordinator = pool.get_agent("coordinator")
        result = await overseer.run(
            "Download https://example.com/file.zip using both getters..."
        )


#### Features

- **Team Tasks**: Run tasks across multiple agents either sequentially or in parallel
- **Agent Cloning**: Create variations of agents with different configurations
- **Resource Sharing**: Agents in a pool can share resources and tools
- **Overseer Pattern**: Use overseer agents to coordinate specialist agents
- **Built-in Collaboration**: Tools for delegation, brainstorming, and debates

#### Configuration

The pool is configured through an agents manifest (YAML):

```yaml
agents:
  agent_1:
    model: openai:gpt-4
    # ... agent-specific config

  agent_2:
    model: openai:gpt-4
    # ... agent-specific config

  overseer:
    model: openai:gpt-4
    # ... overseer config
```

Each agent can have its own:
- Model configuration
- Role and capabilities
- Tools and resources
- System prompts and behavior settings

### Message Forwarding

LLMling Agent supports message forwarding between agents, allowing creation of agent chains and networks.
When an agent processes a message, it can forward it to other agents for further processing:

```python
# Create two agents
async with Agent.open_agent("agents.yml", "analyzer") as agent_a, \
          Agent.open_agent("agents.yml", "reviewer") as agent_b:

    # Let agent_a pass its results to agent_b
    connection = agent_a.pass_results_to(agent_b)

    # Start the chain - agent_b will process agent_a's output
    await agent_a.run("Analyze this code")
    await agent_b.complete_tasks()  # Optinally wait for agent_b to finish
```

Each agent in the chain can:
1. Process the incoming message
2. Access the source agent as a dependency
3. Forward its own response to other agents

This enables simple creation of agent chains.
The connection objects are highly configurable and observable, they even allow routing!




This diagram shows the main components of the LLMling Agent framework:

```mermaid
classDiagram
    %% Core relationships
    AgentsManifest --* AgentConfig : contains
    AgentsManifest --> AgentPool : creates
    AgentPool --* Agent : manages
    FileEnvironment --> Config : loads
    InlineEnvironment --* Config : contains
    Config --> RuntimeConfig : initialized as
    Agent --> RuntimeConfig : uses
    AgentConfig --> FileEnvironment : uses
    AgentConfig --> InlineEnvironment : uses
    Agent --* ToolManager : uses
    Agent --* ConversationManager : uses

    class Config ["[LLMling Core] Config"] {
        Base configuration format defining tools, resources, and settings
        +
        +tools: dict
        +resources: dict
        +prompts: dict
        +global_settings: GlobalSettings
        +from_file()
    }

    class RuntimeConfig ["[LLMling Core] RuntimeConfig"] {
        Runtime state of a config with instantiated components
        +
        +config: Config
        +tools: dict[str, LLMCallableTool]
        +resources: dict[str, Resource]
        +prompts: dict[str, BasePrompt]
        +register_tool()
        +load_resource()
    }

    class AgentsManifest {
        Complete agent configuration manifest defining all available agents
        +
        +responses: dict[str, ResponseDefinition]
        +agents: dict[str, AgentConfig]
        +open_agent()
    }

    class AgentConfig {
        Configuration for a single agent including model, environment and capabilities
        +
        +name: str
        +model: str | Model
        +environment: AgentEnvironment
        +capabilities: Capabilities
        +system_prompts: list[str]
        +get_config(): Config
    }

    class FileEnvironment {
        Environment loaded from external YAML file
        +
        +type: "file"
        +uri: str
    }

    class InlineEnvironment {
        Direct environment configuration without external files
        +
        +type: "inline"
        +tools: ...
        +resources: ...
        +prompts: ...
    }

    class AgentPool {
        Manager for multiple initialized agents
        +
        +manifest: AgentsManifest
        +agents: dict[str, Agent]
        +open()
    }

    class Agent {
        Main agent class handling LLM interactions and tool usage
        +
        +runtime: RuntimeConfig
        +tools: ToolManager
        +conversation: ConversationManager
        +run()
        +run_stream()
        +open()
    }

    class ToolManager {
        Manages tool registration, enabling/disabling and access
        +
        +register_tool()
        +enable_tool()
        +disable_tool()
        +get_tools()
        +list_tools()
    }

    class ConversationManager {
        Manages conversation state and system prompts
        +
        +get_history()
        +clear()
        +add_context_from_path()
        +add_context_from_resource()
    }
```
