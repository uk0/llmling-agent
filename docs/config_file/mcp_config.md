# MCP Server Configuration

MCP (Model Control Protocol) servers allow agents to use external tools through a standardized protocol. They can be configured at both agent and manifest level.

Support for resources and prompts will also come in the future.


## Basic Configuration (agent / team level)

If you want to limit availabbility to specific entitites, MCP servers can also be assigned
to agents or teams

```yaml
agents:
  assistant:
    mcp_servers:
      - "python -m mcp_server"           # Simple command
      - "node mcp_server.js --debug"     # With arguments
teams:
  my_team:
    # other settings..
    mcp_servers:
      - "python -m mcp_server"

```

## Detailed Configuration
Full server configuration with all options:
```yaml
agents:
  assistant:
    mcp_servers:
      # Stdio-based server
      - type: "stdio"
        command: "python"
        args: ["-m", "mcp_server", "--debug"]
        enabled: true
        environment:
          PYTHONPATH: "src"
          DEBUG: "1"

      # SSE-based server
      - type: "sse"
        url: "http://localhost:8000"
        enabled: true
```

## Manifest Level Configuration
Shared servers available to all agents:
```yaml
# Root level configuration
mcp_servers:
  - type: "stdio"
    command: "python"
    args: ["-m", "shared_mcp_server"]

  - type: "sse"
    url: "http://shared-server:8000"

agents:
  assistant:
    # Agent can still have its own servers
    mcp_servers:
      - "python -m agent_specific_server"
```

## Server Types

### Stdio Server
Uses standard input/output for communication:
```yaml
mcp_servers:
  - type: "stdio"
    command: "python"
    args: ["-m", "mcp_server"]
    environment:
      PYTHONPATH: "src"
```

### SSE Server
Uses Server-Sent Events over HTTP:
```yaml
mcp_servers:
  - type: "sse"
    url: "http://localhost:8000"
```

## Shorthand Syntax
Simple commands can use string shorthand:
```yaml
mcp_servers:
  - "python -m mcp_server"     # Converted to stdio config
  - "node server.js --debug"   # With arguments
```
