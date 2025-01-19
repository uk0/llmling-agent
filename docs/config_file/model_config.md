# Models & Providers


In addition to the regular models, LLMling-Agent supports some special kinds of models
provided by [LLMling-models](https://github.com/phil65/LLMling-models).


## Human-Interaction Models
Models that facilitate human interaction and input:

### Input Model
Basic console-based human input for testing and debugging:
```yaml
agents:
  reviewer:
    model:
      type: "input"
      prompt_template: "👤 Please respond to: {prompt}"
      show_system: true
      input_prompt: "Your response: "
```
!!! note
    This mechanism is similar to HumanProviders, but implemented at a different level.
    A HumanProvider has more "access" and is the more powerful way to take over an agent.

### Remote Input Model
Connect to a remote human operator via REST or WebSocket:
```yaml
agents:
  remote_reviewer:
    model:
      type: "remote-input"
      url: "ws://operator:8000/v1/chat/stream"
      protocol: "websocket"  # or "rest"
      api_key: "your-api-key"
```

### User Select Model
Let users interactively choose which model to use:
```yaml
agents:
  interactive:
    model:
      type: "user-select"
      models: ["openai:gpt-4", "openai:gpt-3.5-turbo"]
      prompt_template: "🤖 Choose model for: {prompt}"
      input_prompt: "Enter model number (0-{max}): "
```

## Multi-Models

### Fallback Model
Try models in sequence until one succeeds:
```yaml
agents:
  resilient:
    model:
      type: "fallback"
      models:
        - "openai:gpt-4"         # Try first
        - "openai:gpt-3.5-turbo" # Fallback
        - "anthropic:claude-2"    # Last resort
```

### Cost-Optimized Model
Select models based on budget constraints:
```yaml
agents:
  budget_aware:
    model:
      type: "cost-optimized"
      models: ["openai:gpt-4", "openai:gpt-3.5-turbo"]
      max_input_cost: 0.1  # USD per request
      strategy: "best_within_budget"  # or "cheapest_possible"
```

### Token-Optimized Model
Select models based on context window requirements:
```yaml
agents:
  context_aware:
    model:
      type: "token-optimized"
      models:
        - "openai:gpt-4-32k"     # 32k context
        - "openai:gpt-4"         # 8k context
      strategy: "efficient"  # or "maximum_context"
```

### Delegation Model
Use a model to choose the most appropriate model:
```yaml
agents:
  smart_router:
    model:
      type: "delegation"
      selector_model: "openai:gpt-4-turbo"
      models: ["openai:gpt-4", "openai:gpt-3.5-turbo"]
      selection_prompt: "Pick gpt-4 for complex tasks, gpt-3.5-turbo for simple queries."
```

## Wrapper Models

### AISuite Adapter
Use models from the AISuite library (limited functionality, no tool calls and structured output):
```yaml
agents:
  aisuite_agent:
    model:
      type: "aisuite"
      model: "anthropic:claude-3-opus"
      config:
        anthropic:
          api_key: "your-api-key"
```

### LLM Adapter
Use models from the LLM library (limited functionality, no tool calls and structured output):
```yaml
agents:
  aisuite_agent:
    model:
      type: "llm"
      model: "claude-3-opus"
```

### Import Model
Import and use custom model implementations:
```yaml
agents:
  custom:
    model:
      type: "import"
      model: "myapp.models:CustomModel"
```
