# System Prompts Configuration

System prompts define an agent's role, behavior, and capabilities. LLMling provides multiple ways to configure and access prompts.

## Prompt Providers

LLMling supports multiple prompt sources through a provider system:

### Builtin Prompts

Define prompts directly in your configuration:

```yaml
prompts:
  system_prompts:
    code_reviewer:
      content: "You are a code reviewer specialized in {{ language }}."
      type: role

    error_handler:
      content: |
        You help developers fix {{ error_type }} errors.
        Current context: {{ context }}
      type: methodology
```

### External Providers

Connect to external prompt management systems:

```yaml
prompts:
  providers:
    - type: langfuse
      secret_key: "your-secret"
      public_key: "your-public-key"
      host: "https://cloud.langfuse.com"

    - type: openlit
      url: "https://api.openlit.ai"
      api_key: "your-api-key"

    - type: promptlayer
      api_key: "your-api-key"
```

## Using Prompts in Agents

### Direct System Prompts

Simple string prompts for basic behaviors:

```yaml
agents:
  helper:
    system_prompts:
      - "You are a helpful assistant."
      - "You always provide concise answers."
```

### Library Prompts

Reference prompts from any configured provider:

```yaml
agents:
  expert:
    library_system_prompts:
      - "code_reviewer?language=python"  # Builtin prompt with variables
      - "langfuse:expert_prompt@v2"      # Versioned Langfuse prompt
      - "openlit:code_analysis"          # OpenLIT prompt
```

## Prompt Reference Syntax

Access prompts using a unified reference syntax:

```
[provider:]identifier[@version][?var1=val1,var2=val2]
```

Examples:

- `error_handler` - Builtin prompt
- `langfuse:intro@v2` - Versioned Langfuse prompt
- `code_review?language=python&style=detailed` - Prompt with variables

## Provider Features

Different providers support different features:

- **Builtin**:

- Jinja2 templating
- Variable validation
- No versioning

- **Langfuse**:

- Version control
- Team collaboration
- Usage tracking

- **OpenLIT**:

- Version control
- Variables
- Metadata support

## Best Practices

1. **Use Builtin for Simple Cases**: Keep basic prompts in your config
2. **External for Team Work**: Use Langfuse/OpenLIT for collaborative prompt management
3. **Version Critical Prompts**: Use versioned providers for production prompts
4. **Clear Variables**: Document required variables in prompt descriptions
5. **Consistent Templating**: Use Jinja2 syntax for variables (`{{ var }}`)
