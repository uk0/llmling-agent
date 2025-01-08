# SlashedAgent - CLI Integration Interface

The `SlashedAgent` class provides a unified interface for building prompt-based applications (like CLI) that interact with LLMling agents.
It wraps a regular agent with slash command support and standardized output handling, making it ideal for terminal-based applications.
The run and run_stream methods have the same signature as the agent itself, but integrate it with [Slashed](https://github.com/phil65/slashed) for slash command support.

## Key Features

### Slash Command Support
- Process commands starting with `/` before regular prompts
- Built-in command history
- Command completion support
- Extensible command system

### Unified Output Handling
The agent provides two main output signals for UIs to connect to:

- `message_output`: Regular agent messages and events
- `streamed_output`: Real-time streaming content

These signals emit `AgentOutput` events with the following structure:
```python
@dataclass
class AgentOutput:
    type: OutputType     # Type of the output event
    content: Any         # The main content
    timestamp: datetime  # When it was generated
    metadata: dict       # Additional context
```

Output types include:
- `message_received`: New user messages
- `message_sent`: Agent responses
- `tool_called`: Tool usage
- `command_executed`: Slash command execution
- `status`: Status updates
- `error`: Error messages

### Basic Usage

```python
from llmling_agent import Agent, SlashedAgent

# Create base agent
agent = Agent(runtime=runtime, model="gpt-4")

# Wrap with slash command support
slashed = SlashedAgent(
    agent,
    command_history_path="~/.myapp/history",
)

# Connect to output signals
@slashed.message_output.connect
async def handle_message(output: AgentOutput):
    match output.type:
        case "message_sent":
            print(f"Assistant: {output.content}")
        case "tool_called":
            print(f"Using tool: {output.metadata['tool_name']}")

# Handle streaming output
@slashed.streamed_output.connect
async def handle_stream(output: AgentOutput):
    print(output.content, end="", flush=True)

# Run with command support
await slashed.run("/help")  # Execute help command
await slashed.run("Hello!")  # Regular message
```

### Building CLIs

The SlashedAgent is designed to work with any terminal interface. Here's a simple example using prompt_toolkit:

```python
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter

async def main():
    slashed = SlashedAgent(agent)
    session = PromptSession()

    while True:
        try:
            # Get input with command completion
            text = await session.prompt_async("> ")

            # Run with streaming for real-time output
            async with slashed.run_stream(text) as stream:
                async for chunk in stream:
                    print(chunk, end="", flush=True)

        except KeyboardInterrupt:
            continue
        except EOFError:
            break
```

### Custom Commands

You can add your own commands using Slashed's command system:

```python
from slashed import SlashedCommand, CommandContext

class CustomCommand(SlashedCommand):
    """Custom command implementation."""

    name = "mycmd"
    category = "custom"

    async def execute_command(
        self,
        ctx: CommandContext[AgentPoolView],
        arg1: str,
        optional: int = 42
    ):
        """Execute the command."""
        await ctx.output.print(f"Running {arg1} with {optional}")

# Register with agent
slashed.commands.register_command(CustomCommand())
```

### Properties

The SlashedAgent provides access to common agent functionality through properties:

- `tools`: Tool management interface
- `conversation`: Conversation history management
- `provider`: Access to the underlying provider
- `pool`: Agent pool access
- `model_name`: Current model name
- `context`: Agent context
