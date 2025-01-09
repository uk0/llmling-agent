# Agent Logging

LLMling-agent uses SQLModel with SQLite to maintain a history of agent interactions.
The database is automatically created and managed.

## What Gets Logged

- **Conversations**: Basic metadata like agent name, start time
- **Messages**: Complete message history including:
  - Content
  - Role (user/assistant/system)
  - Model used
  - Token usage and costs
  - Timestamp
  - Message forwarding chains
- **Commands**: Command history with session context
- **Tool Usage**: Tool calls with arguments and results

## Storage Location

The SQLite database is stored using platformdirs:
```python
# Default location:
# Linux: ~/.local/share/llmling/history.db
# Windows: %LOCALAPPDATA%\llmling\history.db
# macOS: ~/Library/Application Support/llmling/history.db
```

## Enabling/Disabling Logging

Logging can be controlled per agent:

Via YAML:
```yaml
agents:
  assistant:
    model: openai:gpt-4o-mini
    enable_db_logging: false  # Disable logging for this agent
```

Via code:
```python
# In constructor
agent = Agent(..., enable_db_logging=False)

# When using open():
async with Agent.open(..., enable_db_logging=False) as agent:
    ...
```

## Conversation Recovery

Sessions can be recovered in multiple ways:

Simple recovery by session name:
```yaml
agents:
  assistant:
    session: "my_session_name"  # Simple session identifier

  analyst:
    model: openai:gpt-4o-mini
    # No session = new conversation each time
```

Advanced query-based recovery:
```yaml
agents:
  assistant:
    model: openai:gpt-4o-mini
    session:
      name: my_session_name    # Optional session identifier
      agents:                  # Filter by specific agents
        - assistant
        - analyst
      since: 1h               # Only messages from last hour
      contains: "analysis"    # Filter by content
      roles:                  # Only specific message types
        - user
        - assistant
      include_forwarded: true # Include forwarded messages
```

Via code:
```python
# Store session ID for later
session = agent.conversation.id

# Recover conversation in new session
async with Agent.open(..., session=session) as agent:
    # Conversation history is automatically loaded
    ...
```

The session configuration supports flexible filtering to recover exactly the conversation context you need.
