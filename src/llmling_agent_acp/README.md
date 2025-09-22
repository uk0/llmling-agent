# llmling-agent ACP Integration

ACP (Agent Client Protocol) integration for llmling-agent, enabling seamless interoperability with desktop applications through JSON-RPC 2.0 communication over stdio streams.

## Overview

This package provides a bridge between llmling-agent's powerful agent system and the Agent Client Protocol, allowing you to:

- Expose llmling agents as ACP-compatible services
- Enable bidirectional JSON-RPC 2.0 communication over stdio
- Support session management and conversation history
- Integrate with desktop applications requiring agent capabilities
- Handle file system operations with permission management
- Support terminal integration for command execution
- Stream responses with content blocks (text, image, audio, resources)
- Seamless MCP (Model Context Protocol) server integration

## Installation

```bash
# Install with ACP support
pip install llmling-agent[acp]

# Or install from source
cd llmling-agent
pip install -e .[acp]
```

## Quick Start

### Basic Agent Server

```python
import asyncio
from llmling_agent import Agent
from llmling_agent_acp import ACPServer

async def main():
    # Create ACP server
    server = ACPServer()

    # Register an agent
    @server.agent(name="chat_agent", file_access=True)
    def create_chat_agent():
        return Agent(
            name="assistant",
            model="openrouter:openai/gpt-4o-mini",
            system_prompt="You are a helpful AI assistant with file access capabilities."
        )

    # Run the server (communicates over stdio)
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())
```

### Loading from Config File

Use existing llmling-agent configuration files:

```python
import asyncio
from llmling_agent_acp import ACPServer

async def main():
    # Load from existing agent config
    server = await ACPServer.from_config("agents.yml")
    
    # Run the server
    await server.run()

asyncio.run(main())
```

#### Example Config File

```yaml
# Standard llmling-agent configuration
agents:
  chat_agent:
    name: "ChatAssistant"
    model: "gpt-4o-mini"
    system_prompt: "You are helpful"
    description: "A friendly AI assistant"

  file_processor:
    name: "FileProcessor" 
    model: "gpt-4o-mini"
    system_prompt: "You process and analyze files"
    description: "File processing specialist"
    tools: ["file_tools"]

  code_assistant:
    name: "CodeAssistant"
    model: "claude-3-5-sonnet"
    system_prompt: "You help with coding tasks"
    description: "AI-powered coding assistant"
    tools: ["code_tools", "github_tools"]

teams:
  development:
    mode: "sequential"
    members: ["file_processor", "code_assistant"]
    description: "File analysis followed by code assistance"
```

ACP-specific settings are controlled via CLI parameters:

```bash
# Basic ACP server
llmling-agent acp agents.yml

# With file system access
llmling-agent acp agents.yml --file-access

# With full capabilities
llmling-agent acp agents.yml --file-access --terminal-access

# With debugging
llmling-agent acp agents.yml --file-access --show-messages --log-level DEBUG
```

### CLI Usage

```bash
# Run agents from config as ACP server
llmling-agent acp agents.yml

# With file system permissions
llmling-agent acp agents.yml --file-access

# Test with manual JSON-RPC (example)
echo '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":1},"id":1}' | llmling-agent acp agents.yml
```

## ACP Protocol Features

### JSON-RPC 2.0 Communication

ACP uses JSON-RPC 2.0 over newline-delimited JSON streams:

```json
// Initialize request
{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":1},"id":1}

// Initialize response  
{"jsonrpc":"2.0","result":{"protocolVersion":1,"agentCapabilities":{...}},"id":1}

// Create session
{"jsonrpc":"2.0","method":"session/new","params":{"cwd":"/tmp"},"id":2}

// Session response
{"jsonrpc":"2.0","result":{"sessionId":"sess_abc123"},"id":2}

// Send prompt
{"jsonrpc":"2.0","method":"session/prompt","params":{
  "sessionId":"sess_abc123",
  "prompt":[{"type":"text","text":"Hello!"}]
},"id":3}

// Streaming responses via session updates
{"jsonrpc":"2.0","method":"session/update","params":{
  "sessionId":"sess_abc123",
  "update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"Hi there!"}}
}}
```

### Content Blocks

ACP supports rich content blocks:

```python
# Text content
TextContent(type="text", text="Hello world")

# Image content  
ImageContent(type="image", data="base64...", mimeType="image/png")

# Audio content
AudioContent(type="audio", data="base64...", mimeType="audio/wav")

# Resource links
ResourceLink(type="resource_link", uri="file:///path/to/file", name="document.pdf")

# Embedded resources
EmbeddedResource(type="resource", resource=TextResourceContents(uri="...", text="..."))
```

### File System Integration

Agents can request file operations through the client:

```python
from llmling_agent_acp import FileSystemBridge

# Read file through ACP client
content = await fs_bridge.read_file("/path/to/file.txt", session_id)

# Write file through ACP client  
await fs_bridge.write_file("/path/to/output.txt", "content", session_id)

# Request permission for sensitive operations
allowed = await fs_bridge.request_permission("write", {"path": "/etc/config"}, session_id)
```

### Session Management

Sessions maintain conversation state and context:

```python
# Sessions are managed automatically
session_id = await session_manager.create_session(
    agent=my_agent,
    cwd="/working/directory", 
    client=acp_client
)

# Process prompts within session context
async for notification in session.process_prompt(content_blocks):
    # Stream responses to client
    await client.sessionUpdate(notification)
```

## Advanced Usage

### Custom Client Implementation

```python
from llmling_agent_acp import ACPClientInterface

class CustomACPClient:
    async def requestPermission(self, params):
        # Custom permission handling
        return RequestPermissionResponse(outcome=...)
    
    async def readTextFile(self, params):
        # Custom file reading logic
        return ReadTextFileResponse(content=...)
    
    async def writeTextFile(self, params):
        # Custom file writing logic
        pass
    
    async def sessionUpdate(self, params):
        # Handle session updates (streaming responses)
        print(f"Update: {params.update.sessionUpdate}")

# Use custom client
server = ACPServer(client=CustomACPClient())
```

### Tool Integration

```python
from llmling_agent_acp import ACPToolBridge

# Bridge handles tool execution with ACP protocol
bridge = ACPToolBridge(client)

# Execute tool and stream progress
async for notification in bridge.execute_tool(tool, params, session_id):
    await client.sessionUpdate(notification)

# Request permission for tool execution
allowed = await bridge.request_tool_permission(tool, params, session_id)
```

### MCP Server Integration

ACP provides seamless integration with MCP (Model Context Protocol) servers, allowing agents to access external tools and data sources. MCP servers are automatically connected when creating sessions.

#### How MCP Integration Works

1. **Session Creation**: Client provides MCP server configurations in `session/new`
2. **Automatic Connection**: ACP server connects to all specified MCP servers
3. **Tool Integration**: MCP tools become available to the agent automatically
4. **Transparent Usage**: Agent can use MCP tools just like built-in tools

#### MCP Server Configuration

MCP servers are specified in the `session/new` request:

```json
{
  "jsonrpc": "2.0",
  "method": "session/new",
  "params": {
    "cwd": "/home/user/project",
    "mcpServers": [
      {
        "name": "filesystem",
        "command": "mcp-server-filesystem",
        "args": ["--stdio"],
        "env": []
      },
      {
        "name": "web_search",
        "command": "node",
        "args": ["/path/to/search-server.js"],
        "env": [
          {
            "name": "API_KEY",
            "value": "your-api-key"
          }
        ]
      }
    ]
  }
}
```

#### Supported MCP Transports

Currently supports:
- **stdio**: Standard input/output (all agents must support)
- **SSE**: Server-Sent Events (optional)
- **HTTP**: HTTP transport (optional)

#### Example: Agent with MCP Tools

```python
# Agent automatically gets access to MCP tools
async def demo_mcp_integration():
    # MCP servers are connected during session creation
    # Tools become available immediately
    
    response = await agent.run(
        "Search for Python tutorials and save the results to a file"
    )
    
    # Agent can now use:
    # - web_search tool (from MCP server)
    # - file_write tool (from MCP server)
    # - Built-in reasoning capabilities
```

#### MCP Tool Execution Flow

1. **Agent decides to use tool** → Regular llmling-agent tool selection
2. **Tool execution** → Routed to appropriate MCP server
3. **Results** → Returned to agent for processing
4. **Streaming** → Tool progress streamed to ACP client

#### Benefits

- **Zero Configuration**: MCP servers work out of the box
- **Tool Ecosystem**: Access to the growing MCP tool ecosystem
- **Transparent Integration**: No changes needed to agent logic
- **Automatic Discovery**: All MCP tools become available instantly

### Desktop Application Integration

ACP is designed for desktop applications:

1. **Start ACP server as subprocess**:
```python
import subprocess
import json

# Start llmling-agent ACP server
process = subprocess.Popen(
    ["llmling-agent", "acp", "config.yml", "--file-access"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True
)
```

2. **Send JSON-RPC requests**:
```python
# Initialize
request = {"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":1},"id":1}
process.stdin.write(json.dumps(request) + "\n")
process.stdin.flush()

# Read response
response = json.loads(process.stdout.readline())
```

3. **Handle streaming responses**:
```python
# Session updates are sent as notifications
while True:
    line = process.stdout.readline()
    if not line:
        break
    
    message = json.loads(line)
    if message.get("method") == "session/update":
        # Handle streaming response
        update = message["params"]["update"]
        print(f"Agent: {update['content']['text']}")
```

## Examples

### File Processing Agent

```python
@server.agent(name="file_processor", file_access=True)
def create_file_agent():
    async def analyze_file(path: str) -> str:
        """Analyze a file and return insights."""
        # File access is handled through ACP client automatically
        return f"Analysis of {path}: ..."
    
    return Agent(
        name="file_analyzer",
        model="gpt-4",
        system_prompt="You analyze files and provide insights.",
        tools=[analyze_file]
    )
```

### Terminal Integration

```python
@server.agent(name="terminal_helper", terminal_access=True) 
def create_terminal_agent():
    return Agent(
        name="terminal_assistant", 
        model="gpt-4",
        system_prompt="You help with terminal commands and system administration."
    )
```

### Multi-Agent Collaboration

```python
# Sequential processing pipeline
@server.agent(name="analyzer")
def create_analyzer():
    return Agent(name="analyzer", model="gpt-4", system_prompt="Analyze content.")

@server.agent(name="summarizer")  
def create_summarizer():
    return Agent(name="summarizer", model="gpt-4", system_prompt="Create summaries.")

# Agents work together through session management
```

## Protocol Compatibility

Supports the full ACP specification:

- **Agent Methods**: `initialize`, `session/new`, `session/load`, `session/prompt`, `session/cancel`, `authenticate`
- **Client Methods**: `fs/read_text_file`, `fs/write_text_file`, `session/request_permission`, `session/update`
- **Content Types**: Text, Image, Audio, Resource Links, Embedded Resources  
- **Session Management**: Creation, loading, history, cancellation
- **Permission System**: User confirmation for sensitive operations
- **Tool Execution**: Progress tracking, error handling, streaming updates
- **Terminal Operations**: Command execution, output streaming, process management

## Error Handling

ACP provides robust error handling:

```python
# JSON-RPC errors are automatically formatted
try:
    result = await agent.run(prompt)
except Exception as e:
    # Automatically converted to JSON-RPC error response
    error = RequestError.internal_error({"details": str(e)})
    response = {"jsonrpc":"2.0","error":error.to_error_obj(),"id":request_id}
```

## Security

- **Permission System**: All file operations require user approval
- **Sandboxed Execution**: Agents run in controlled environment  
- **Path Validation**: File paths are validated before access
- **Session Isolation**: Each session maintains separate state
- **Input Validation**: All inputs are validated using Pydantic models

## Performance

- **Streaming Responses**: Real-time response delivery
- **Async Operations**: Non-blocking I/O throughout
- **Session Caching**: Efficient conversation state management
- **Connection Pooling**: Optimized stdio communication
- **Tool Execution**: Parallel tool execution with progress tracking

## Comparison with Other Protocols

| Feature | ACP (This) | A2A (BeeAI) | Original ACP |
|---------|------------|-------------|--------------|
| Transport | stdio/JSON-RPC | HTTP/REST | HTTP/REST |
| Use Case | Desktop apps | Web services | Web services |
| File Access | Native support | Limited | Via tools |
| Bidirectional | Yes | Limited | No |
| Streaming | JSON-RPC notifications | Server-sent events | Polling |
| Sessions | Built-in | Manual | Manual |
| Permissions | User prompts | Server policy | Server policy |
| MCP Integration | Native, automatic | Not supported | Not supported |

## License

Same as llmling-agent main project.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for ACP integration
4. Submit a pull request

## Support

- [GitHub Issues](https://github.com/phil65/llmling-agent/issues)
- [Documentation](https://llmling-agent.readthedocs.io/)
- [Agent Client Protocol Specification](https://agentclientprotocol.com/)