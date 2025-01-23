# Provider Configuration

Providers determine how an agent processes messages and generates responses. The provider configuration is set in the agent's `type` field.

## AI Provider (PydanticAI)
The default provider, using pydantic-ai for language model integration.

```yaml
agents:
  my-agent:
    provider:
      type: "pydantic_ai"  # provider discriminator
      name: "gpt4-agent"  # optional provider instance name
      end_strategy: "early"  # "early" | "complete" | "confirm"
      result_retries: 3  # max retries for result validation
      defer_model_check: false  # whether to defer model check until first run
      model_settings:  # additional settings passed to pydantic-ai
        temperature: 0.7
        max_tokens: 1000
      validation_enabled: true  # whether to validate outputs against schemas
      allow_text_fallback: true  # accept plain text when structure fails
```

## Human Provider
Provider that routes messages to human operators for manual responses.

```yaml
agents:
  human-agent:
    provider:
      type: "human"
      name: "human-reviewer"  # optional instance name
      timeout: 300  # seconds to wait for response (null = wait forever)
      show_context: true  # whether to show conversation context
```

## LiteLLM Provider
Provider using LiteLLM for unified model access.

```yaml
agents:
  my-agent:
    provider:
      type: "litellm"
      name: "litellm-agent"  # optional instance name
      retries: 3  # max retries for failed calls
      model: "openai:gpt-4"  # optional model override
      model_settings:  # additional settings passed to LiteLLM
        temperature: 0.5
        api_key: "${OPENAI_API_KEY}"  # environment variable
```

## Callback Provider
Provider that uses Python functions for responses.

```yaml
agents:
  callback-agent:
    provider:
      type: "callback"
      name: "custom-processor"  # optional name
      callback: "myapp.processors.analyze_text"  # import path to function
```

## Shorthand Syntax
For common providers, you can use string shortcuts instead of full configuration:

```yaml
provider: "pydantic_ai"  # Use default AI provider (pydantic-ai
# or
provider: "human"  # Use default human provider
# or
provider: "litellm"  # Use default LiteLLM provider
# or
provider: "path.to.callable"  # Create a CallableProvider
```

## Configuration Notes
- The `type` field serves as discriminator for provider configurations
- Provider settings affect only message processing, not agent infrastructure
- Each provider type has its own validation requirements
- Settings can be overridden at runtime through the Python API
- Environment variables can be used in configuration values
