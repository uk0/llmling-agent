# Agent Capabilities

Configure what operations an agent is allowed to perform via the `capabilities` field.

```yaml
agents:
  my_agent:
    capabilities:
      # Agent Discovery & Delegation
      can_list_agents: false        # Whether agent can discover other agents
      can_delegate_tasks: false     # Whether agent can assign tasks to other agents
      can_observe_agents: false     # Whether agent can monitor other agents' activities

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

      # Agent Creation
      can_create_workers: false    # Whether agent can create worker agents (as tools)
      can_create_delegates: false  # Whether agent can spawn temporary delegate agents
```


## Understanding Capabilities vs Tools

While tools provide specific functions an agent can use (like web searches or calculations), capabilities control an agent's access to privileged operations that can modify the system itself or access sensitive information. Think of capabilities as "administrative privileges" that determine what an agent is allowed to do beyond regular tool usage.

For example:

- A tool might provide specific statistics, but the `stats_access` capability gives access to all usage data of the agents itself
- While tools are standalone functions, capabilities like `can_register_tools` allow the agent to modify the system's available toolset itself
- Powerful functions like code execution are also included here to underline their potential impact (nothing stopping the user to add custom code execution tools of course)
- Through capabilities like `can_create_delegates`, agents can modify the agent pool itself

## Configuration

Configure what operations an agent is allowed to perform via the `capabilities` field:

## Examples

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

## Access Levels

For `history_access` and `stats_access`:
- `none`: No access (default)
- `own`: Access only to own data
- `all`: Access to all agents' data
