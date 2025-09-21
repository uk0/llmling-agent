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
      can_manage_processes: false  # Whether agent can start and manage background processes

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

### History Access

Control access to conversation history and usage data:
```python
history_access: AccessLevel = "none"
"""Level of access to conversation history:
- "none": No access
- "own": Only own history
- "all": All agents' history
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

can_manage_processes: bool = False
"""Whether the agent can start and manage background processes."""
```

### Process Management

Control ability to manage background processes:
```python
can_manage_processes: bool = False
"""Whether the agent can start and manage background processes.

When enabled, provides access to:
- start_process: Start commands in background and get process ID
- get_process_output: Check current output from running processes
- wait_for_process: Block until process completes
- kill_process: Terminate running processes
- release_process: Clean up process resources
- list_processes: Show all active processes

This capability allows agents to run long-running commands, monitor their
progress, and manage multiple concurrent processes. Use with caution as
processes consume system resources.
"""
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
      can_load_resources: true
      can_list_resources: true
      can_register_tools: true
      can_register_code: true
      can_install_packages: true
      can_chain_tools: true
      can_execute_code: true
      can_execute_commands: true
      can_manage_processes: true
      can_create_workers: true
      can_create_delegates: true
```

### Process Manager Agent

```yaml
agents:
  process_manager:
    capabilities:
      # Can manage background processes and monitor them
      can_manage_processes: true
      can_execute_commands: true    # Often used together
      can_read_files: true         # To check process outputs/logs
      can_list_directories: true   # To navigate filesystem
      history_access: own          # Track process history
```
