# Storage Configuration

The storage configuration defines how agent interactions, messages, and tool usage are logged. It's defined at the root level of the manifest.

## Basic Structure

```yaml
storage:
  # List of storage providers
  providers:
    - type: "sql"  # SQL database (default)
      url: "sqlite:///history.db"
      pool_size: 5
      log_messages: true
      log_conversations: true
      log_tool_calls: true
      log_commands: true
      log_context: true

  # Global logging flags (apply to all providers)
  log_messages: true      # Log all messages
  log_conversations: true # Log conversation metadata
  log_tool_calls: true   # Log tool executions
  log_commands: true     # Log command executions
```

## Available Providers

### SQL Storage (Default)
Uses SQLModel for database storage. Supports SQLite, PostgreSQL, etc.

```yaml
storage:
  providers:
    - type: "sql"
      url: "sqlite:///history.db"  # Database URL
      pool_size: 5  # Connection pool size
      # Logging flags (override global)
      log_messages: true
      log_conversations: true
      log_tool_calls: true
      log_commands: true
      log_context: true
```

### Text Log
Writes logs to text files with configurable format.

```yaml
storage:
  providers:
    - type: "text_file"
      path: "logs/chat.log"  # Log file path
      format: "chronological"  # "chronological" | "conversations"
      template: "chronological"  # predefined or custom template path
      encoding: "utf-8"  # File encoding
      # Logging flags
      log_messages: true
      log_conversations: true
      log_tool_calls: true
      log_commands: true
      log_context: true
```

### File Storage
Stores data in structured files (JSON, YAML, etc.).

```yaml
storage:
  providers:
    - type: "file"
      path: "data/history.json"  # Storage file path
      format: "auto"  # "auto" | "json" | "yaml" | etc.
      encoding: "utf-8"  # File encoding
      # Logging flags
      log_messages: true
      log_conversations: true
      log_tool_calls: true
      log_commands: true
      log_context: true
```

### Memory Storage
In-memory storage for testing.

```yaml
storage:
  providers:
    - type: "memory"  # No additional configuration needed
      # Logging flags
      log_messages: true
      log_conversations: true
      log_tool_calls: true
      log_commands: true
      log_context: true
```

## Multiple Providers

You can use multiple providers simultaneously:

```yaml
storage:
  providers:
    - type: "sql"  # Primary storage in database
      url: "sqlite:///history.db"
    - type: "text_file"  # Additional logging to file
      path: "logs/chat.log"
    - type: "memory"  # Temporary storage for testing

  # Global settings apply to all providers
  log_messages: true
  log_conversations: true
```

## Provider Selection

When loading history, providers are selected in this order:
1. Explicitly specified preferred provider
2. Default provider if configured
3. First capable provider in the list
4. Raises error if no capable provider found

## Notes
- Individual provider settings override global settings
- The SQL provider is recommended for production use
- Memory provider is useful for testing
- Text logs are good for debugging
- File storage is suitable for simple deployments
