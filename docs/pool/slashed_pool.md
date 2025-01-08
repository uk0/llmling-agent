# SlashedPool

The `SlashedPool` is a high-level interface for managing and interacting with multiple agents. It provides a unified way to communicate with either individual agents or broadcast to all agents, while supporting both command-based and message-based interactions.

## Purpose

SlashedPool serves as the main interface for UI applications that need to:
- Interact with multiple agents
- Route messages to specific agents
- Execute commands on agents
- Get responses either synchronously or as they arrive

## Core Features

### Agent Routing

Messages and commands can be routed to specific agents in two ways:

```python
# Using @ syntax
await pool.run("@agent1 analyze this data")

# Using agent parameter
await pool.run("analyze this data", agent="agent1")
```

### Broadcasting

Messages can be sent to all agents:

```python
# Sends to all agents and waits for all responses
responses = await pool.run("status report")
print(responses.combined_content)  # All responses formatted together
```

### Command Execution

Commands can be routed just like messages:

```python
# Agent-specific command
await pool.execute_command("@agent1 /enable-tool mytool")

# Pool-level command
await pool.execute_command("/list-agents")
```

## Main Methods

### run()

The primary method for sending messages and getting responses:

```python
# Single agent - returns ChatMessage
response = await pool.run("analyze this", agent="agent1")
print(response.content)

# Multiple agents - returns MultiAgentResponse
responses = await pool.run("status report")
for name, msg in responses.responses.items():
    print(f"{name}: {msg.content}")
```

### run_iter()

For getting responses as they complete, useful for showing progress:

```python
# Shows responses as they arrive
total = len(pool.agents)
completed = 0
async for msg in pool.run_iter("analyze data"):
    completed += 1
    print(f"Progress: {completed}/{total}")
    print(f"From {msg.metadata['sender']}: {msg.content}")
```

### execute_command()

For executing slash commands with agent routing:

```python
# Execute command on specific agent
result = await pool.execute_command("@agent1 /status")

# Execute pool-level command
result = await pool.execute_command("/list-agents")
```

## Signal System

SlashedPool provides signals for UI updates:

```python
# Message signals
pool.message_output.connect(handle_message)
pool.streamed_output.connect(handle_stream)

# Agent management signals
pool.agent_added.connect(handle_new_agent)
pool.agent_removed.connect(handle_agent_removed)

# Command signals
pool.agent_command_executed.connect(handle_command)
```

## Usage in Applications

SlashedPool is designed to be the primary interface for UI applications:

```python
# Create pool with agents
pool = SlashedPool(agent_pool)

# GUI Application
class AgentUI:
    def __init__(self, pool: SlashedPool):
        self.pool = pool
        self.pool.message_output.connect(self.update_chat)
        self.pool.agent_command_executed.connect(self.update_status)

    async def send_message(self, content: str):
        if content.startswith("/"):
            await self.pool.execute_command(content)
        else:
            await self.pool.run(content)

    async def show_progress(self, content: str):
        progress = 0
        async for msg in self.pool.run_iter(content):
            progress += 1
            self.update_progress_bar(progress / len(self.pool.agents))
```

## Benefits

1. **Unified Interface**: One consistent way to interact with agents
2. **Type Safety**: Full type support for responses
3. **Flexible Routing**: Easy routing to specific agents or broadcasting
4. **Progress Tracking**: Get responses as they arrive
5. **UI Integration**: Signal system for UI updates
6. **Command System**: Integrated slash command support

## Integration with Other Components

SlashedPool works with:
- `Agent`: Base agent functionality
- `SlashedAgent`: Command-enabled agents
- `AgentPool`: Core agent management
- UI frameworks through its signal system

SlashedPool acts as the bridge between your UI application and the agent ecosystem, providing a clean, type-safe interface for all agent interactions.
