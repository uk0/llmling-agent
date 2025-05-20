# Agent Capabilities

Capabilities control what "special" operations an agent is allowed to perform.
You can think of them as a special kind of tools with "extra superpowers", having access to the internals.
They are defined in the agent's configuration:

```yaml
agents:
  my_agent:
    capabilities:
      # Agent Discovery & Delegation
      can_list_agents: true        # Discover other available agents
      can_delegate_tasks: true     # Delegate tasks to other agents
      can_observe_agents: false    # Monitor other agents' activities
      can_list_teams: false        # Discover available teams
      can_ask_agents: true         # Can ask other agents in the pool

      # History & Statistics Access
      history_access: "own"        # "none" | "own" | "all"
      stats_access: "none"         # "none" | "own" | "all"

      # Resource Management
      can_load_resources: true     # Load and access resource content
      can_list_resources: true     # Discover available resources
      can_read_files: false        # Read local and remote files
      can_list_directories: false  # List directories and their contents

      # Tool Management
      can_register_tools: false    # Register importable functions as tools
      can_register_code: false     # Create new tools from provided code
      can_install_packages: false  # Install Python packages for tools
      can_chain_tools: true       # Chain multiple tool calls into one

      # Code Execution (Use with caution)
      can_execute_code: false     # Execute Python code (no sandbox)
      can_execute_commands: false # Execute CLI commands

      # Agent Creation
      can_create_workers: true    # Create worker agents as tools
      can_create_delegates: true  # Spawn temporary delegate agents
      can_add_agents: true        # Create new persistent agents in the pool
      can_add_teams: false        # Create new teams in the pool
      can_connect_nodes: false    # Connect nodes in the system
```

## Capability Details

### Agent Discovery & Delegation
- `can_list_agents`: Allows discovering other agents in the pool
- `can_delegate_tasks`: Enables sending tasks to other agents
- `can_observe_agents`: Permits monitoring of other agents' activities
- `can_list_teams`: Allows discovering available teams
- `can_ask_agents`: Can ask other agents in the pool questions

### History & Statistics Access
- `history_access`: Controls access to conversation history
  - `"none"`: No access to history
  - `"own"`: Can access own conversation history
  - `"all"`: Can access all agents' history

- `stats_access`: Controls access to usage statistics
  - `"none"`: No access to statistics
  - `"own"`: Can view own usage statistics
  - `"all"`: Can view all agents' statistics

### Resource Management
- `can_load_resources`: Allows loading content from resources
- `can_list_resources`: Permits discovery of available resources
- `can_read_files`: Enables reading files from local or remote sources
- `can_list_directories`: Allows listing directories and their contents

### Tool Management
- `can_register_tools`: Allows registering Python functions as tools
- `can_register_code`: Permits creating new tools from provided code
- `can_install_packages`: Enables installing Python packages
- `can_chain_tools`: Allows chaining multiple tool calls together

### Execution Capabilities
!!! warning "Security Warning"
    These capabilities provide direct code execution and should be used with caution:
    - `can_execute_code`: Enables Python code execution (no sandbox)
    - `can_execute_commands`: Allows CLI command execution

### Agent Creation
- `can_create_workers`: Allows creating worker agents as tools
- `can_create_delegates`: Enables spawning temporary delegate agents
- `can_add_agents`: Allows creating new persistent agents in the pool
- `can_add_teams`: Enables creating new teams in the agent pool
- `can_connect_nodes`: Allows connecting messaging nodes in the system

## Default Configuration
By default, all capabilities are set to `false` or `"none"` for security. Enable only the capabilities that your agent needs.

## Example Use Cases

### Read-Only Agent
```yaml
agents:
  reader:
    capabilities:
      can_load_resources: true
      can_read_files: true
      can_list_resources: true
      history_access: "own"
      # All other capabilities: false
```

### Task Coordinator
```yaml
agents:
  coordinator:
    capabilities:
      can_list_agents: true
      can_list_teams: true
      can_delegate_tasks: true
      can_observe_agents: true
      history_access: "all"
      stats_access: "all"
```

### Tool Power User
```yaml
agents:
  power_user:
    capabilities:
      can_register_tools: true
      can_chain_tools: true
      can_load_resources: true
      can_list_resources: true
      can_read_files: true
```