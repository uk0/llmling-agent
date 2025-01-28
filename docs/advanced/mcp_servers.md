# MCP Server Integration

## Overview
LLMling Agent supports integration with MCP (Model Control Protocol) servers to extend agent capabilities through standardized interfaces. Currently, we support tool integration with both stdio and SSE-based MCP servers.

## Configuration

MCP servers can be configured in two ways:

### String Configuration
Simple command-line style configuration:
```yaml
agents:
  my_agent:
    mcp_servers:
      - "uvx python-mcp-server --arg1 value1"
      - "node js-mcp-server.js"
```

### Full Configuration
Detailed configuration with environment variables and options:
```yaml
agents:
  my_agent:
    mcp_servers:
      - type: stdio
        command: "pipx"
        args: ["run", "python-mcp-server", "--debug"]
        environment:
          MY_VAR: "value"
      - type: sse  # Not yet implemented
        url: "http://localhost:8000"
```

## Architecture

### Components

#### MCPClient
Handles connection and communication with MCP servers:

- Manages server lifecycle (connect/disconnect)
- Lists available tools
- Forwards tool calls to server
- Handles response parsing

```python
client = MCPClient()
await client.connect(command="pipx", args=["run", "server"])
tools = client.get_tools()  # Get OpenAI function format
result = await client.call_tool("tool_name", arguments={})
```

#### ToolManager Integration
Tools from MCP servers are registered with the agent's ToolManager:

- Tools are available like any other agent tool
- Proper lifecycle management
- Multiple MCP servers per agent

### Lifecycle

1. Agent initialization
2. MCP server connections established in `__aenter__`
3. Tools registered with agent
4. Tool calls forwarded to appropriate server
5. Cleanup on `__aexit__`

## Current Limitations

- Only tool functionality implemented
- No SSE server support yet
- No streaming responses
- No bidirectional communication
- No resource management
- No state management

## Future Extensions

### Planned Features
- Resource management (read/write/subscribe)
- Prompt management
- Completion requests
- Progress notifications
- Bidirectional event system
- Server-side caching
- Tool state persistence

### Integration Ideas
- Replace RuntimeConfig with MCP backends
- Unified backend interface
- Plugin system through MCP
- Cross-agent communication
- Shared tool state

## Usage Example

```python
# Configure agent with MCP server
async with AgentPool("pool.yml") as pool:
    # MCP tools from YAML defined mcp servers are automatically available
    agent = self.get_agent("my_agent_from_yaml")
    result = await agent.run("Use MCP tool to process data")

    # Multiple servers
    async with Agent(
        name="agent_name",
        model="...",
        mcp_servers=[
            "uvx server1",
            "uvx server2 --debug"
        ]
    )) as agent:
        # Tools from both servers available
        result = await agent.run("Use tools from multiple servers")
```

## Development

### Adding New Features
1. Define protocol extension in MCP spec
2. Update client implementation
3. Add new methods to MCPClient
4. Integrate with agent system
5. Update documentation

### Testing
- Test server connections
- Test tool registration
- Test error handling
- Test cleanup
- Test multiple servers

### Best Practices
- Always use async context managers
- Handle server errors gracefully
- Clean up resources properly
- Type hints for everything
- Document public interfaces

## References
- MCP Specification: [link]
- Python MCP Implementation: [link]
- Example Servers: [link]
