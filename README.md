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
- ⚡ True async framework. Easy set-up of complex async agent flows. Faster initializations of "heavy" agents (and first experimentations with async UI supervision f Agents in real-time)
- 📝 Easy consistent APIs
- 🛡️ Complete agent defintion via YAML files including extensive JSON schema to help with creating configurations.
- 🔒 Leveraging the complete pydantic-based type-safe stack and bringing it to the multi-agent world
- 🎮 Complete integrated command sytem to control agents from prompt-based interfaces
- 🔌 Agent MCP server support, initialized when entering the async context.
- 👁️ Multi-modal support for both LiteLLM and Pydantic-AI (currently Images and PDFs if model support is given)
- 💾 Storage providers to allow writing to local files, databases, etc. with many customizable backends. Log to SQL databases and pretty-print to a file according to your own wishes.
- 🧩 Support for creating "description prompts" for many common python type(s / instances). Your agent understands common datatypes.
- 🔗 Unique powerful connection-based messaging approach for object-oriented routing and observation.
- 🎯 Integration of Meta-Model system based on [LLMling-models](https://github.com/phil65/llmling-models), also configurable via YAML.
- 🔐 Deep integration of structured responses into workflows and (generic) typing system.
- 📋 Response type definition via YAML. Structured response Agents can be defined in the agent config.
- 🛡️ Capabilites system allowing runtime modifications and "special" commands (on-the-fly agent generation, history lookups)
- 📊 Complete database logging of Agent interactions including easy recovery based on query parameters.
- ⚙️ pytest-inspired way to create agents from YAML in a type-safe manner. "Auto-populated signatures."
- 🛜 Comletely UPath backed. Any file operations under our control is routed through fsspec to allow referencing remote sourcces.
- 📕 Integrated prompt management system.
- 🔧 Tasks, tools, and what else you can expect from an Agent framework.
- 🏎️ No fixed dependencies on all the super-heavy LLM libraries. Way faster startup than most other frameworks, and all IO in our control is async.
- 👥 Easy human-in-the-loop interactions on multiple levels (complete "providers" or model-based, see llmling-models)
- 💻 A CLI application with extensive slash command support to build agent flows interactively. Set up message connections via commands.
- ℹ️ The most easy way available to generate static websites in combination with [MkNodes](https://github.com/phil/mknodes) and  [the corresponding MkDocs plugin](https://github.com/phil65/mkdocs_mknodes)

## 🔜 Coming Soon
- 🎯 Built-in event system for reactive agent behaviors (file changes, webhooks, timed events)
- 🖥️ Real-time-monitoring via Textual app in truly async manner. Talk to your agents while they are working and monitor the progress!




### Why LLMling-agent? 🤔

Why another framework you may ask? The framework stands out through three core principles:


#### 🛡️ Type Safety and Structure
Unlike other frameworks that rely on free-form text exchanges, LLMling-agent enforces type safety throughout the entire agent interaction chain.
From input validation to structured outputs, every data flow is typed and validated, making it significantly more reliable for production systems.

#### 💬 Object-oriented async messaging and routing system
A powerful approach to messaging using Connection ("Talk") objects which allow all kind of new patterns for async agent communication

#### ⚙️ Rich Configuration System
While other frameworks require extensive Python code for setup, LLMling-agent introduces a comprehensive YAML configuration system.
This allows defining complex agent behaviors, capabilities, and interactions declaratively.
The configuration supports inheritance, composition, and strong validation, making it easier to manage large-scale agent deployments.

#### 🤝 Human-AI Collaboration
Instead of choosing between fully autonomous or human-controlled operations, LLMling-agent offers flexible human-in-the-loop integration.
From full human control to selective oversight of critical actions, or hooking in remotely via Network,
the framework makes it natural to build systems that combine AI capabilities with human supervision and interaction.


## Quick Start

The fastest way to start chatting with an AI:
```bash
# Start an ephemeral chat session (requires uv)
uvx llmling-agent quickstart openai:gpt-4o-mini
```

This creates a temporary agent ready for chat - no configuration needed!
The according API keys need to be set as environment variables.

Use `help` to see what commands are at your disposal.

## Provider support

| Provider Type | Streaming Support | Multi-Modal Support | Structured Response Support |
|--------------|------------------|---------------------|---------------------------|
| PydanticAI | Yes | (Model dependent) | Yes |
| LiteLLM | Yes | (Model dependent) | Yes |
| Human-in-the-loop | Yes (but more a gimmick) | No | Yes |
| Callable-based | (Depends on callback) | (Depends on callback) | Yes |

(Multi-modal support (Images & PDF) in PydanticAI and LiteLLM depends on the underlying model's capabilities)


## 🚀 Quick Examples

Three ways to create a simple agent flow:


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
        agent = await pool.add_agent("assistant", model="openai:gpt-4o-mini")
        connection = agent >> browser  # this sets up a permanent connection.
        await agent.run("Tell us a random major city! Just one word!")
        print(connection.stats.total_cost)  # Check cost of this connection
```

This flow will:

- Ask the 1st agent to tell a major city
- Will make the 2nd agent open a related webpage using that info


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
llmling-agent run assistant --config agents.yml "whats your favourite holiday destination?"
> What's your favorite holiday destination?
```


### CLI Version (Interactive using slash command system)
```bash
# Start session
llmling-agent quickstart --model openai:gpt-4o-mini
# Create browser assistant
/create-agent browser --system-prompt "Open Wikipedia pages matching the topics you receive." --tools webbrowser.open
# Connect the agents
/connect browser
# Speak to the main agent, which will auto-forward.
> What's your favorite holiday destination?
```

### YAML configuration

While you can define agents with 3 lines of YAML (or competely programmatic or via CLI),
you can also create agents as well as their connections, agent tasks, storage providers and much more via YAML.
This is the extended version

```yaml
# agents.yml
agents:
  analyzer:
    provider:  # Provider configuration
      type: "pydantic_ai"  # Provider type discriminator
      name: "PydanticAI Provider"  # Optional provider name
      end_strategy: "early"  # "early" | "complete" | "confirm"
      model:  # Model configuration
        type: "fallback"  # Lot of special "meta-models" included out of the box!
        models:  # Try models in sequence
          - "openai:gpt-4"
          - "openai:gpt-3.5-turbo"
          - "anthropic:claude-2"
      result_retries: 3  # Max retries for result validation
      defer_model_check: false  # Whether to defer model evaluation
      validation_enabled: true  # Whether to validate outputs
      allow_text_fallback: true  # Accept plain text when validation fails

    name: "Code Analyzer"  # Display name
    inherits: "base_agent"  # Optional parent config to inherit from
    description: "Code analysis specialist"
    debug: false
    retries: 1  # Number of retries for failed operations

    # Structured output
    result_type:
      type: "inline"  # or "import" for Python types
      fields:
        severity:
          type: "str"
          description: "Issue severity"
        issues:
          type: "list[str]"
          description: "Found issues"

    # Core behavior
    system_prompts:
      - "You analyze code for potential issues and improvements."

    # Session & History
    session:
      name: "analysis_session"
      since: "1h"  # Only load messages from last hour
      roles: ["user", "assistant"]  # Only specific message types

    # Capabilities (role-based permissions)
    capabilities:
      can_delegate_tasks: true
      can_load_resources: true
      can_register_tools: true
      history_access: "own"  # "none" | "own" | "all"
      stats_access: "all"

    # Environment configuration
    environment:
      type: "inline"  # or "file" for external config
      tools:
        analyze_complexity:
          import_path: "radon.complexity"
          description: "Calculate code complexity"
        run_linter:
          import_path: "pylint.lint"
          description: "Run code linting"
      resources:
        coding_standards:
          type: "text"
          content: "PEP8 guidelines..."

    # Knowledge sources
    knowledge:
      paths: ["docs/**/*.md"]  # Glob patterns for files
      resources:
        - type: "repository"
          url: "https://github.com/user/repo"
      prompts:
        - type: "file"
          path: "prompts/analysis.txt"

    # MCP Server integration
    mcp_servers:
      - type: "stdio"
        command: "python"
        args: ["-m", "mcp_server"]
        environment:
          DEBUG: "1"
      - "python -m other_server"  # shorthand syntax

    # Worker agents (specialists)
    workers:
      - name: "formatter"
        reset_history_on_run: true
        pass_message_history: false
        share_context: false
      - "linter"  # shorthand syntax

    # Message forwarding
    connections:
      - type: "agent"
        name: "reporter"
        connection_type: "run"  # "run" | "context" | "forward"
        priority: 1
        queued: true
        queue_strategy: "latest"
        transform: "my_module.transform_func"
        wait_for_completion: true
        filter_condition:  # When to forward messages
          type: "word_match"
          words: ["error", "warning"]
          case_sensitive: false
        stop_condition:  # When to disconnect
          type: "message_count"
          max_messages: 100
          count_mode: "total"  # or "per_agent"
        exit_condition:  # When to exit application
          type: "cost_limit"
          max_cost: 10.0
    # Event triggers
    triggers:
      - type: "file"
        name: "code_change"
        paths: ["src/**/*.py"]
        extensions: [".py"]
        debounce: 1000  # ms

# Response type definitions
responses:
  AnalysisResult:
    type: "inline"
    description: "Code analysis result format"
    fields:
      severity: {type: "str"}
      issues: {type: "list[str]"}

  ComplexResult:
    type: "import"
    import_path: "myapp.types.ComplexResult"

# Storage configuration
storage:
  providers:
    - type: "sql"
      url: "sqlite:///history.db"
      pool_size: 5
    - type: "text_file"
      path: "logs/chat.log"
      format: "chronological"
  log_messages: true
  log_conversations: true
  log_tool_calls: true
  log_commands: true

# Pre-defined jobs
jobs:
  analyze_code:
    name: "Code Analysis"
    description: "Analyze code quality"
    prompt: "Analyze this code: {code}"
    required_return_type: "AnalysisResult"
    knowledge:
      paths: ["src/**/*.py"]
    tools: ["analyze_complexity", "run_linter"]
```

You can use an Agents manifest in multiple ways:

- Use it for CLI sessions

```bash
llmling-agent chat --config agents.yml system_checker
```

- Run it using the CLI

```bash
llmling-agent run --config agents.yml my_agent "Some prompt"
```

- Use the defined Agent programmatically

```python
from llmling_agent import AgentPool

async with AgentPool("agents.yml") as pool:
    agent = pool.get_agent("my_agent")
    result = await agent.run("User prompt!")
    print(result.data)
```

- Start *watch mode* and only react to triggers

```bash
llmling-agent watch --config agents.yml
```


### Agent Pool: Multi-Agent Coordination

The `AgentPool` allows multiple agents to work together on tasks. Here's a practical example of parallel file downloading:

```python
# agents.yml
agents:
  file_getter:
    model: openai:gpt-4o-mini
    environment:
      tools:
        download_file:
          import_path: llmling_agent_tools.download_file  # a simple httpx based async callable
    system_prompts:
      - |
        You are a download specialist. Just use the download_file tool
        and report its results. No explanations needed.

  overseer:
    capabilities:
      can_delegate_tasks: true  # these capabilities are available as tools for the agent
      can_list_agents: true
    model: openai:gpt-4o-mini
    system_prompts:
      - |
        You coordinate downloads using available agents.
        1. Check out the available agents and assign each of them the download task
        2. Report the results.

```

```python
from llmling_agent.delegation import AgentPool

async def main():
    async with AgentPool("agents.yml") as pool:
        # first we create two agents based on the file_getter template
        file_getter_1 = pool.get_agent("file_getter")
        file_getter_2 = pool.get_agent("file_getter")
        # then we form a team and execute the task
        team = file_getter_1 & file_getter_2
        responses = await team.run_parallel("Download https://example.com/file.zip")

        # Or let a coordinator orchestrate using his capabilities.
        coordinator = pool.get_agent("coordinator")
        result = await overseer.run(
            "Download https://example.com/file.zip by delegating to all workers available!"
        )
```

### Advanced Connection Features

Connections between agents are highly configurable and support various patterns:

```python
# Basic connection in shorthand form.
connection = agent_a >> agent_b  # Forward all messages

# Extended setup: Queued connection (manual processing)
connection = agent_a.pass_results_to(
    agent_b,
    queued=True,
    queue_strategy="latest",  # or "concat", "buffer"
)
# messages can queue up now
await connection.trigger(optional_additional_prompt)  # Process queued messages sequentially

# Filtered connection (example: filter by keyword):
connection = agent_a.pass_results_to(
    agent_b,
    filter_condition=lambda message, target_agent, stats: "keyword" in message,
)

# Conditional disconnection (example: disconnect after cost limit):
connection = agent_a.pass_results_to(
    agent_b,
    filter_condition=lambda _, _, stats: stats.total_cost > 1.0,

)

# Message transformations
async def transform_message(message: str) -> str:
    return f"Transformed: {message}"

connection = agent_a.pass_results_to(agent_b, transform=transform_message)

# Connection statistics
print(f"Messages processed: {connection.stats.message_count}")
print(f"Total tokens: {connection.stats.token_count}")
print(f"Total cost: ${connection.stats.total_cost:.2f}")
```


### Human-in-the-Loop Integration

LLMling-Agent offers multiple levels of human integration:

```python
# Provider-level human integration
from llmling_agent import Agent

async with Agent(provider="human") as agent:
    result = await agent.run("We can ask ourselves and be part of Workflows!")
```

```yaml
# Or via YAML configuration
agents:
  human_agent:
    provider: "human"  # Complete human control
    timeout: 300  # Optional timeout in seconds
    show_context: true  # Show conversation context
```

You can also use LLMling-models for more sophisticated human integration:
- Remote human operators via network
- Hybrid human-AI workflows
- Input streaming support
- Custom UI integration

### Capability System

Fine-grained control over agent permissions:

```python
agent.capabilities.can_load_resources = True
agent.capabilities.history_access = "own"  # "none" | "own" | "all"
```

```yaml
agents:
  restricted_agent:
    capabilities:
      can_delegate_tasks: false
      can_register_tools: false
      history_access: "none"
```

### Event-Driven Automation

React to file changes, webhooks, and more:

```python
# File watching
agent.events.add_file_watch(paths=["src/**/*.py"], debounce=1000)

# Webhook endpoint
agent.events.add_webhook("/hooks/github",port=8000)

# Also included: time-based and email
```

### Multi-Modal Support

Handle images and PDFs alongside text (depends on provider / model support)

```python
import PIL.Image
from llmling_agent import Agent

async with Agent.open() as agent:
    result = await agent.run("What's in this image?", PIL.Image.open("image.jpg"))
    result = await agent.run("What's in this image?", pathlib.Path("image.jpg"))
    result = await agent.run("What's in this PDF?", pathlib.Path("document.pdf"))
```

### Command System

Extensive slash commands available in all interfaces:

```bash
/list-tools              # Show available tools
/enable-tool tool_name   # Enable specific tool
/connect other_agent     # Forward results
/model gpt-4            # Switch models
/history search "query"  # Search conversation
/stats                   # Show usage statistics
```

### Storage & Analytics

All interaction is tracked using (multiple) configurable storage providers.
Information can get fetched programmatically or via CLI.

```python
# Query conversation history
messages = await agent.conversation.filter_messages(
    SessionQuery(
        since="1h",
        contains="error",
        roles={"user", "assistant"},
    )
)

# Get usage statistics
stats = await agent.context.storage.get_conversation_stats(
    group_by="model",
    period="24h",
)
```


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

## 📚 MkDocs Integration

In combination with [MkNodes](https://github.com/phil65/mknodes) and the [MkDocs plugin](https://github.com/phil65/mkdocs_mknodes),
you can easily generate static documentation for websites with a few lines of code.

```python

@nav.route.page("Feature XYZ", icon="oui:documentation", hide="toc")
def gen_docs(page: mk.MkPage):
    """Generate docs using agents."""
    agent = Agent[None](model="openai:gpt-4o-mini")
    page += mk.MkAdmonition("MkNodes includes all kinds of Markdown objects to generate docs!")
    source_code = load_source_code_from_folder(...)
    page += mk.MkCode() # if you want to display source code
    result = agent.run_sync("Describle Feature XYZ in MkDocs compatible markdown including examples.", content)
    page += result.content
```



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


### [Read the documentation for further info!](https://phil65.github.io/llmling-agent/)
