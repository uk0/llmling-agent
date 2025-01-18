# System Prompts Configuration

System prompts define an agent's role, behavior, and capabilities. LLMling provides flexible ways to configure these prompts.

## Basic Configuration

The simplest way is to provide string prompts:

```yaml
agents:
  helper:
    system_prompts:
      - "You are a helpful assistant."
      - "You always provide concise answers."
```

## Advanced Prompt Types

### File-based Prompts
Load prompts from external files:

```yaml
agents:
  expert:
    system_prompts:
      - type: "file"
        description: "Expert knowledge base"
        path: "prompts/expert_role.md"
        format: "markdown"  # or "text", "jinja2"
        watch: true  # Reload when file changes
```

### Dynamic Prompts
Generate prompts using Python functions:

```yaml
agents:
  contextual:
    system_prompts:
      - type: "function"
        description: "Dynamic context loader"
        import_path: "my_app.prompts.get_context"
        template: "Current context: {result}"
```

## Mixed Configuration

You can combine different prompt types:

```yaml
agents:
  assistant:
    system_prompts:
      # Simple string prompt
      - "You are an AI assistant named {name}."

      # File-based role definition
      - type: "file"
        path: "prompts/assistant_role.md"

      # Dynamic context
      - type: "function"
        import_path: "my_app.prompts.get_capabilities"
        template: "Your capabilities: {result}"
```

## Best Practices

1. **Keep Base Prompts Simple**: Use string prompts for basic behaviors
2. **Use Files for Complex Prompts**: Move longer prompts to files
3. **Dynamic When Needed**: Use dynamic prompts only when context changes
4. **Clear Structure**: Organize prompts from general to specific
