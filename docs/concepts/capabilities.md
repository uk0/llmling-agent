# Agent Capabilities

## What are Capabilities?

Capabilities in LLMling define what "special" operations an agent is allowed to perform.
They act as access privileges that control an agent's ability to:

- Interact with other agents
- Access resources and history
- Execute code or commands
- Register and use tools

When capabilities are enabled, corresponding tools become available to the agent.
This provides a secure and explicit way to control agent permissions.

## Defining Capabilities

Capabilities are defined in the agent configuration:

```yaml
agents:
  supervisor:
    # Full access agent
    capabilities:
      can_list_agents: true
      can_delegate_tasks: true
      can_create_workers: true
      history_access: "all"

  worker:
    # Limited access agent
    capabilities:
      can_load_resources: true
      history_access: "own"
      can_execute_code: false
```

In Python:
```python
from llmling_agent.config import Capabilities

capabilities = Capabilities(
    can_list_agents=True,
    can_delegate_tasks=True,
    history_access="own"
)
```

## Available Capabilities

### Agent Interaction
Control how agents can discover and interact with each other:
```python
can_list_agents: bool = False
"""Whether the agent can discover other available agents."""

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

```
