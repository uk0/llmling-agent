# Key Concepts

## Agents and Their Capabilities

An agent in LLMling Agent is a configurable entity that combines:
- A language model (like GPT-4)
- A set of capabilities (what it can do)
- Tools it can use
- System prompts that define its behavior

Agents have different capability levels defined by roles:

```yaml
# Basic roles and their capabilities
roles:
  overseer:
    # Full access to agent management and history
    can_list_agents: true
    can_delegate_tasks: true
    can_observe_agents: true
    history_access: "all"
    stats_access: "all"

  specialist:
    # Access to own history and statistics
    history_access: "own"
    stats_access: "own"

  assistant:
    # Basic access to own history only
    history_access: "own"
    stats_access: "none"
```

## Configuration Files

LLMling Agent uses two types of configuration files:

1. **Agent Configuration** (agents.yml)
   - Defines agents and their behaviors
   - Specifies models and system prompts
   - References environment configurations
   - Configures response types and roles

2. **Environment Configuration** (env_*.yml)
   - Defines available tools and resources
   - Managed by the LLMling core library
   - Can be shared between multiple agents
   - Configures technical capabilities

Example agent configuration:
```yaml
agents:
  file_analyzer:
    model: openai:gpt-4
    role: specialist
    environment: env_files.yml
    result_type: FileAnalysis  # Reference to a response type
    system_prompts:
      - "You analyze file contents and metadata."

responses:
  FileAnalysis:
    description: "File analysis result"
    type: inline
    fields:
      content_summary:
        type: str
        description: "Summary of file contents"
      size_bytes:
        type: int
        description: "File size in bytes"
```

## Tools and Resources

Tools are functions that agents can call to interact with the system or perform tasks:

1. **Built-in Tools**
   - File operations
   - System information
   - History access (based on capabilities)

2. **Custom Tools**
   - Defined in environment configurations
   - Can be Python functions or external commands
   - Have schemas for type safety
   - Can require confirmation before execution

Example tool usage in code:
```python
@agent.tool
async def analyze_file(ctx: RunContext[AgentContext], path: str) -> str:
    """Analyze a file's contents."""
    if not ctx.deps.capabilities.can_read_files:
        raise PermissionError("No permission to read files")
    # ... implementation
```

## Chat Sessions

Chat sessions provide interactive conversations with agents:

1. **Types of Interaction**
   - CLI-based interactive chat
   - Web interface (using Gradio)
   - Programmatic conversation API

2. **Features**
   - Message history management
   - Tool state tracking
   - Model overrides
   - Token usage monitoring
   - Conversation export

Example chat session:
```python
from llmling_agent.chat_session import ChatSessionManager

# Create a session
manager = ChatSessionManager()
session = await manager.create_session(agent)

# Interactive conversation
response = await session.send_message("Analyze this file")
print(response.content)

# Stream responses
async for chunk in session.send_message("Long analysis...", stream=True):
    print(chunk.content, end="", flush=True)
```

For more detailed information about specific concepts, see:
- [Agent Configuration](https://phil65.github.io/llmling-agent/user-guide/agent-configuration.html)
- [Running Agents](https://phil65.github.io/llmling-agent/user-guide/running-agents.html)
- [Tool Development](https://phil65.github.io/llmling-agent/user-guide/agent-development.html)
