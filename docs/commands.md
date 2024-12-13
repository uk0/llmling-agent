## Command System

LLMling Agent implements a command system that allows users to execute special operations directly in the chat interface.
Unlike regular chat messages which are processed by the LLM,
commands (prefixed with /) provide direct access to system functionality like managing prompts, viewing help, or controlling the session.

### Command Syntax

Commands use a consistent syntax across all interfaces:

```bash
/command arg1 arg2 --kwarg1 value1 --kwarg2 value2
```

- Commands always start with a forward slash (/)
- Arguments can be passed as positional args or keyword args (with --)
- Arguments containing spaces should be quoted: `--message "hello world"`
- All argument values are passed as strings

Examples:
```bash
/hello                     # Simple command without args
/hello Alice              # Command with one positional arg
/hello --greeting Hi      # Command with one keyword arg
/hello Alice --greeting "Good morning"  # Both types of args
```

### Built-in Commands

#### General
- `/help [command]` - Show all available commands or details for a specific command
- `/hello [name] [--greeting msg]` - Simple test command

#### Prompts
LLMling Agent integrates with the prompt system from RuntimeConfig, allowing you to list and execute prompts directly in chat:

- `/list-prompts` - Show all available prompts
- `/prompt <name> [arg1=value1] [arg2=value2]` - Execute a specific prompt

Example prompt usage:
```bash
# List available prompts
/list-prompts

# Execute a prompt named "analyze" with arguments
/prompt analyze --text "Hello world" --language en

# Execute a prompt with multiple arguments
/prompt generate --style formal --length short --topic "AI ethics"
```

### Custom Commands

Interfaces (like CLI) can register their own commands. For example, the CLI adds:

- `/exit` - Exit the chat session

The web interface may provide additional commands specific to the web UI.

### Command Output

Command responses are displayed differently depending on the interface:
- CLI: Shows output in the terminal
- Web UI: Displays output in the chat window as system messages
