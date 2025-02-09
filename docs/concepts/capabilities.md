# Agent Capabilities

## What are Capabilities?

Capabilities in LLMling define what "special" operations an agent is allowed to perform.
While tools provide specific functions an agent can use (like web searches or calculations),
capabilities control an agent's access to privileged operations that can modify the system itself or access sensitive information.

Think of capabilities as "administrative privileges" that determine what an agent is allowed to do beyond regular tool usage.
When capabilities are enabled, corresponding tools become available to the agent,
providing a secure and explicit way to control agent permissions.

## Defining Capabilities

Capabilities can be defined in YAML configuration:

```yaml
agents:
  my_agent:
    capabilities:
      # Agent Discovery & Delegation
      can_list_agents: false        # Whether agent can discover other agents
      can_delegate_tasks: false     # Whether agent can assign tasks to other agents
      can_observe_agents: false     # Whether agent can monitor other agents' activities
      can_ask_agents: false         # Whether agent can ask other agents directly

      # History & Statistics Access
      history_access: none          # Access to conversation history (none|own|all)
      stats_access: none           # Access to usage statistics (none|own|all)

      # Resource Management
      can_load_resources: false    # Whether agent can load resource content
      can_list_resources: false    # Whether agent can discover available resources

      # Tool Management
      can_register_tools: false    # Whether agent can register importable functions
      can_register_code: false     # Whether agent can create new tools from code
      can_install_packages: false  # Whether agent can install Python packages
      can_chain_tools: false       # Whether agent can chain multiple tool calls

      # Code Execution
      can_execute_code: false      # Whether agent can execute Python code (WARNING: No sandbox)
      can_execute_commands: false  # Whether agent can execute CLI commands

      # Agent / Team Creation
      can_create_workers: false    # Whether agent can create worker agents (as tools)
      can_create_delegates: false  # Whether agent can spawn temporary delegate agents
      can_add_agents: false       # Whether agent can add new agents to the pool
      can_add_teams: false       # Whether agent can add new teams to the pool
      can_connect_nodes: false       # Whether agent can connect two nodes
```

Or in Python:

```python
from llmling_agent.config import Capabilities

capabilities = Capabilities(
    can_list_agents=True,
    can_delegate_tasks=True,
    history_access="own"
)

agent = Agent(
    name="my_agent",
    capabilities=capabilities,
    model="gpt-4"
)
```

## Available Capabilities

### Agent / Team Interaction

Control how agents can discover and interact with each other:
```python
can_list_agents: bool = False
"""Whether the agent can discover other available agents."""

can_list_teams: bool = False
"""Whether the agent can discover teams of the pool."""

can_delegate_tasks: bool = False
"""Whether the agent can delegate tasks to other agents."""

can_observe_agents: bool = False
"""Whether the agent can monitor other agents' activities."""

can_ask_agents: bool = False
"""Whether the agent can ask other agents of the pool."""
```

### History & Statistics Access

Control access to conversation history and usage data:
```python
history_access: AccessLevel = "none"
"""Level of access to conversation history:
- "none": No access
- "own": Only own history
- "all": All agents' history
"""

stats_access: AccessLevel = "none"
"""Level of access to usage statistics:
- "none": No access
- "own": Only own stats
- "all": All agents' stats
"""
```

### Resource Management

Control access to resources and tools:
```python
can_load_resources: bool = False
"""Whether the agent can load and access resource content."""

can_list_resources: bool = False
"""Whether the agent can discover available resources."""

can_register_tools: bool = False
"""Whether the agent can register importable functions as tools."""

can_register_code: bool = False
"""Whether the agent can create new tools from provided code."""

can_install_packages: bool = False
"""Whether the agent can install Python packages for tools."""

can_chain_tools: bool = False
"""Whether the agent can chain multiple tool calls into one."""
```

### Code Execution

Control ability to execute code (use with caution):
```python
can_execute_code: bool = False
"""Whether the agent can execute Python code (WARNING: No sandbox)."""

can_execute_commands: bool = False
"""Whether the agent can execute CLI commands (use at own risk)."""
```

### Agent Creation

Control ability to create and manage other agents:
```python
can_create_workers: bool = False
"""Whether the agent can create worker agents (as tools)."""

can_create_delegates: bool = False
"""Whether the agent can spawn temporary delegate agents."""

can_add_agents: bool = False
"""Whether the agent can add other agents to the pool."""

can_add_agents: bool = False
"""Whether the agent can add new teams to the pool."""
```

## Common Patterns

Here are some common capability configurations for different agent roles:

### Basic Agent

```yaml
agents:
  restricted_agent:
    capabilities:
      # Minimal capabilities - can only use predefined tools
      can_load_resources: true    # Can load resources
```

### Power User Agent

```yaml
agents:
  power_user:
    capabilities:
      can_load_resources: true
      can_list_resources: true
      can_register_tools: true
      history_access: own         # Can access own history
      stats_access: own          # Can access own stats
```

### Team Lead Agent

```yaml
agents:
  team_lead:
    capabilities:
      # Can manage other agents but no code execution
      can_list_agents: true
      can_delegate_tasks: true
      can_observe_agents: true
      history_access: all
      stats_access: all
      can_create_workers: true
      can_create_delegates: true
```

### Admin Agent

```yaml
agents:
  admin:
    capabilities:
      # Full access to everything
      can_list_agents: true
      can_delegate_tasks: true
      can_observe_agents: true
      history_access: all
      stats_access: all
      can_load_resources: true
      can_list_resources: true
      can_register_tools: true
      can_register_code: true
      can_install_packages: true
      can_chain_tools: true
      can_execute_code: true
      can_execute_commands: true
      can_create_workers: true
      can_create_delegates: true
```
