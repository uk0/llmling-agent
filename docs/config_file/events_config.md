# Event Configuration

Events (or "triggers") define automated activations of agents based on various sources. They allow agents to respond to:
- File system changes
- Webhook calls
- Manual triggers
- (Future: Message queues, time schedules, etc.)

## File Watch Events
Monitor file system changes and trigger agent actions:

```yaml
agents:
  file_monitor:
    triggers:
      - type: "file"
        name: "python_watcher"
        enabled: true
        paths: ["src/**/*.py"]        # Paths to watch
        extensions: [".py"]           # Optional file type filter
        ignore_paths: ["**/__pycache__"]  # Optional ignore patterns
        recursive: true               # Watch subdirectories
        debounce: 1600               # Minimum ms between triggers
```

## Webhook Events
Listen for HTTP requests:

```yaml
agents:
  api_handler:
    triggers:
      - type: "webhook"
        name: "github_webhook"
        enabled: true
        port: 8000
        path: "/github"
        secret: "${WEBHOOK_SECRET}"   # Optional validation secret
```

## Manual Triggers
Define reusable agent actions that can be triggered on demand:

```yaml
agents:
  code_reviewer:
    triggers:
      - type: "manual"
        name: "review_pr"
        enabled: true
        prompt: "Review the changes in the PR and provide feedback"
        description: "Trigger a code review for a pull request"
```

## Common Properties
All trigger types share these properties:

```yaml
triggers:
  - name: "my_trigger"               # Unique identifier
    enabled: true                    # Whether trigger is active
    knowledge:                       # Optional knowledge to load
      paths: ["context/*.md"]        # Files to load as context
      resources:                     # LLMling resources
        - type: "cli"
          command: "git status"
      prompts:                       # Context prompts
        - "Consider this background information..."
```

## Multiple Triggers
Agents can have multiple triggers of different types:

```yaml
agents:
  project_assistant:
    triggers:
      # Watch for code changes
      - type: "file"
        name: "code_watcher"
        paths: ["src/**/*.py"]
        extensions: [".py"]

      # Listen for GitHub webhooks
      - type: "webhook"
        name: "github_events"
        port: 8000
        path: "/github"

      # Manual review trigger
      - type: "manual"
        name: "review_code"
        prompt: "Review latest changes"
```

## Advanced Features
- **Knowledge Loading**: Each trigger can load specific context
- **Conditional Activation**: (Future) Trigger only if conditions met
- **Event Filtering**: Configure which events to respond to
- **Action Chaining**: Triggers can activate multiple agents
```
