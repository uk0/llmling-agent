# Agent Configuration

This guide explains how to configure agents in LLMling Agent through YAML files.

## Configuration File Structure

An agent configuration file consists of three main sections:
- `agents`: Defines one or more agents
- `responses`: Defines structured response types (optional)
- `roles`: Defines custom capability roles (optional)

Basic structure:
```yaml
responses:
  # Define structured response types
  ResponseName:
    description: "Response description"
    type: inline
    fields:
      field_name:
        type: str
        description: "Field description"

roles:
  # Custom roles (extends built-in roles)
  analyst:
    can_list_agents: false
    history_access: "own"
    stats_access: "own"

agents:
  # Agent definitions
  agent_name:
    model: openai:gpt-4
    role: analyst
    environment: env_file.yml
    result_type: ResponseName
    system_prompts: []
    user_prompts: []
```

## Response Types

Response types define structured outputs for agents. They use Pydantic models under the hood:

```yaml
responses:
  FileAnalysis:
    description: "Analysis of a file's contents"
    type: inline
    fields:
      summary:
        type: str
        description: "Content summary"
      word_count:
        type: int
        description: "Number of words"
      sentiment:
        type: float
        description: "Sentiment score (-1 to 1)"
        constraints:
          ge: -1
          le: 1

  SystemStatus:
    description: "System health check result"
    type: inline
    fields:
      status:
        type: str
        description: "Overall status"
        constraints:
          enum: ["healthy", "warning", "error"]
      messages:
        type: list[str]
        description: "Status messages"
```

## System Prompts

System prompts define an agent's behavior and capabilities. You can provide multiple prompts that build on each other:

```yaml
agents:
  code_reviewer:
    model: openai:gpt-4
    system_prompts:
      # Base behavior
      - "You are a code review assistant."

      # Additional context
      - |
        Review guidelines:
        - Focus on security and performance
        - Suggest improvements
        - Be constructive

      # Tool usage instructions
      - |
        Available tools:
        - analyze_code: Static code analysis
        - test_coverage: Get test coverage
        Use these tools to support your review.

    # Default prompts to test the agent
    user_prompts:
      - "Review this Python file: main.py"
      - "What's the test coverage?"
```

## Environmental Settings

Agent configurations reference LLMling environment files that define available tools and resources:

```yaml
agents:
  file_analyzer:
    environment: env_files.yml  # Reference to environment file
    model_settings:  # Model-specific settings
      temperature: 0.7
      max_tokens: 2000
    # ... other settings
```

The referenced environment file (managed by LLMling):
```yaml
# env_files.yml
tools:
  read_file:
    import_path: builtins.open
    description: "Read file contents"

  get_stats:
    import_path: os.stat
    description: "Get file statistics"

resources:
  templates:
    type: text
    content: |
      Analysis Template:
      Size: {size}
      Modified: {mtime}
      Content Summary:
      {summary}
```

## Tool Management

Tools can be:
1. Enabled/disabled per agent
2. Restricted by role capabilities
3. Protected by confirmation handlers

```yaml
agents:
  secure_agent:
    # ... other settings ...

    # Tool confirmation settings
    confirm_tools:
      - delete_file  # Require confirmation for specific tools
      - modify_data

    # Or enable confirmation for all tools
    confirm_tools: true

    # Tool-specific settings (if supported by the tool)
    tool_settings:
      read_file:
        max_size: 1000000
      analyze_code:
        ignore_patterns: ["*.pyc", "__pycache__"]
```

## Role-Based Capabilities

Roles define what agents can do. Use built-in roles or define custom ones:

```yaml
# Built-in roles:
# - overseer: Full access
# - specialist: Own history/stats
# - assistant: Basic access

# Custom roles
roles:
  developer:
    # Extend built-in capabilities
    can_list_agents: true
    can_delegate_tasks: false
    can_observe_agents: false
    history_access: "own"  # "none", "own", or "all"
    stats_access: "own"

  security_auditor:
    can_list_agents: true
    can_observe_agents: true
    history_access: "all"
    stats_access: "all"

agents:
  code_agent:
    role: developer  # Use custom role
    # ... other settings ...

  audit_agent:
    role: security_auditor
    # ... other settings ...
```

For complete examples and templates, see:
- [Basic Examples](https://phil65.github.io/llmling-agent/examples/basic.html)
- [Advanced Configuration](https://phil65.github.io/llmling-agent/examples/advanced.html)
- [Environment Setup](https://phil65.github.io/llmling/configuration.html)
