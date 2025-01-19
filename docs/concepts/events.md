# Events

## Overview

Events in LLMling allow agents to react to external triggers such as:

- File system changes
- Webhook calls (coming soon)
- User interface actions
- System signals

Events can trigger agent execution automatically, making agents responsive to their environment.

## Event Types

### File System Events
Monitors file changes and triggers agent execution:
```yaml
triggers:
  watch_code:
    type: file
    name: code_watcher
    paths: ["src/**/*.py"]  # Glob patterns supported
    extensions: [".py"]     # Optional filter
    ignore_paths: ["**/__pycache__"]
    recursive: true
    debounce: 1600  # ms
```

When files change, the agent receives events with:
- Path of changed file
- Type of change (added/modified/deleted)
- Timestamp of change

### Webhook Events (Coming Soon)
Listen for HTTP requests:
```yaml
triggers:
  github:
    type: webhook
    name: github_hook
    port: 8000
    path: "/github"
    secret: "your-secret"  # For validation
```

## Event Manager

Each agent has an event manager (`agent.events`) that handles:

- Event source lifecycle
- Event processing
- Callback management

```python
# Access event manager
manager = agent.events

# Add custom callback
async def on_event(event: EventData):
    print(f"Event received: {event}")

await manager.add_callback(on_event)

# Remove callback
await manager.remove_callback(on_event)
```

## YAML Configuration

Events can be configured in the agent manifest:

```yaml
agents:
  code_reviewer:
    description: "Reviews code changes"
    model: openai:gpt-4

    # Event configuration
    triggers:
      # File watcher
      - type: file
        name: watch_source
        paths: ["src"]
        extensions: [".py", ".js"]
        ignore_paths: ["**/tests"]

      # Multiple watchers
      - type: file
        name: watch_docs
        paths: ["docs"]
        extensions: [".md"]

      # Webhook (coming soon)
      - type: webhook
        name: pr_hook
        port: 8000
        path: "/github"
```

## CLI Watch Mode

LLMling provides a CLI command to run agents in watch mode:

```bash
# Start watching with configuration
llmling-agent watch --config agents.yaml

# Watch specific agent
llmling-agent watch --config agents.yaml --agent code_reviewer

# Additional options
llmling-agent watch \
  --config agents.yaml \
  --agent code_reviewer \
  --debug \
  --log-level DEBUG
```

Watch mode:

1. Loads agent configuration
2. Sets up configured event sources
3. Runs until interrupted (Ctrl+C)
4. Logs events and agent responses

## Event Processing

When an event occurs:

1. Event source creates `EventData`:
```python
@dataclass(frozen=True)
class FileEvent(EventData):
    """File system event."""
    path: str
    type: "added" | "modified" | "deleted"
```

2. Event manager processes event:
```python
async def emit_event(self, event: EventData):
    """Emit event to all callbacks."""
    # Run custom callbacks
    for callback in self._callbacks:
        await callback(event)

    # Run default handler (agent.run)
    if self.auto_run:
        prompt = event.to_prompt()
        await self.agent.run(prompt)
```

3. Agent receives converted prompt:
```python
# File event example
"File modified: src/main.py
Please review this change."
```

## Example: Code Review Bot

```yaml
agents:
  reviewer:
    description: "Automated code reviewer"
    model: openai:gpt-4

    # Watch for code changes
    triggers:
      - type: file
        name: watch_code
        paths: ["src"]
        extensions: [".py"]
        debounce: 1600

    # Tools for reviewing
    tools:
      - import_path: tools.git.get_diff
      - import_path: tools.analyze_code

    # System prompt
    system_prompts:
      - "You are a code review bot that provides quick feedback on code changes."
```

Run in watch mode:
```bash
llmling-agent watch --config agents.yaml --agent reviewer
```

Now the agent automatically:

1. Detects file changes
2. Gets git diff
3. Analyzes changes
4. Provides review comments

The event system makes agents proactive, responding to changes in their environment without manual intervention.
