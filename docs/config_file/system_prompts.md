# System Prompts Configuration

System prompts define an agent's role, behavior, and capabilities. LLMling provides multiple ways to configure system prompts using a flexible discriminated union approach.

## Basic System Prompts

The simplest way to define system prompts is using strings:

```yaml
agents:
  helper:
    system_prompts:
      - "You are a helpful assistant."
      - "You always provide concise answers."
```

## Prompt Configuration Types

System prompts support four different configuration types through a discriminated union:

### 1. Static Prompts

Direct text prompts for simple cases:

```yaml
agents:
  assistant:
    system_prompts:
      # String shortcut (auto-converted to static)
      - "You are a helpful assistant."

      # Explicit static prompt
      - type: static
        content: "You provide detailed explanations."
```

### 2. File-based Prompts

Load prompts from Jinja2 template files:

```yaml
agents:
  expert:
    system_prompts:
      - type: file
        path: "prompts/code_reviewer.j2"
        variables:
          language: "python"
          experience_level: "senior"

      - type: file
        path: "templates/role_prompt.j2"
        variables:
          role: "data analyst"
          domain: "healthcare"
```

Template file example (`prompts/code_reviewer.j2`):
```jinja2
You are a {{ experience_level }} {{ language }} code reviewer.

Your expertise includes:
- Code quality assessment
- Performance optimization
- Security best practices

Always provide constructive feedback and suggest improvements.
```

### 3. Library Reference Prompts

Reference prompts from the configured prompt library:

```yaml
prompts:
  system_prompts:
    expert_analyst:
      content: |
        You are an expert data analyst.
        Focus on finding patterns and insights.
      category: role

agents:
  analyst:
    system_prompts:
      - type: library
        reference: "expert_analyst"

      - type: library
        reference: "step_by_step"
```

### 4. Function-generated Prompts

Generate prompts dynamically using callable functions:

```yaml
agents:
  dynamic_agent:
    system_prompts:
      - type: function
        function: "my_module:generate_role_prompt"
        arguments:
          role: "technical_writer"
          specialization: "API documentation"

      - type: function
        function: "prompts.dynamic:weather_context"
        arguments:
          location: "San Francisco"
```

Function example:
```python
def generate_role_prompt(role: str, specialization: str) -> str:
    return f"""
    You are a {role} specializing in {specialization}.

    Key responsibilities:
    - Create clear, comprehensive documentation
    - Follow industry best practices
    - Ensure accessibility and usability
    """

def weather_context(location: str) -> str:
    # Could fetch real weather data
    return f"Current weather context for {location}: Sunny, 72°F"
```

## Mixed Prompt Types

You can combine different prompt types in a single agent:

```yaml
agents:
  advanced_assistant:
    system_prompts:

      - "You are an advanced AI assistant."  # String shortcut

      - type: static  # Static prompt
        content: "You have access to real-time information."

      - type: file  # File-based template
        path: "prompts/capabilities.j2"
        variables:
          version: "2.0"
          features: ["analysis", "generation", "reasoning"]

      - type: library  # Library reference
        reference: "professional_tone"

      - type: function # Function-generated
        function: "context:get_current_capabilities"
        arguments:
          include_experimental: false
```

## Prompt Library Configuration

Define reusable prompts in your configuration:

```yaml
prompts:
  system_prompts:
    # Role definitions
    technical_expert:
      category: role
      content: |
        You are a technical expert specializing in:
        - Software architecture
        - System design
        - Performance optimization

    code_reviewer:
      category: role
      content: |
        You are an experienced code reviewer.
        Focus on:
        - Code quality and maintainability
        - Security best practices
        - Performance considerations

    # Methodology definitions
    step_by_step:
      category: methodology
      content: |
        Follow this systematic approach:
        1. Understand the requirements
        2. Break down into smaller tasks
        3. Execute methodically
        4. Verify results

    # Communication styles
    professional:
      category: tone
      content: |
        Maintain professional communication:
        - Use formal language
        - Be concise but thorough
        - Provide evidence for claims

agents:
  senior_dev:
    system_prompts:
      - "You specialize in Python and TypeScript."
      - type: library
        reference: "technical_expert"
      - type: library
        reference: "code_reviewer"
      - type: library
        reference: "step_by_step"
      - type: library
        reference: "professional"
```

## File Organization

Keep your configuration organized by separating prompts:

```yaml
# prompts.yml
prompts:
  system_prompts:
    expert_roles:
      category: role
      content: |
        You are a domain expert...

# agents.yml
INHERIT: prompts.yml

agents:
  my_agent:
    system_prompts:
      - type: library
        reference: "expert_roles"
```

## Path Resolution

File-based prompts resolve paths relative to the configuration file:

```
project/
├── config.yml
├── prompts/
│   ├── roles/
│   │   └── expert.j2
│   └── styles/
│       └── formal.j2
└── agents/
    └── specialist.yml
```

```yaml
# config.yml
agents:
  expert:
    system_prompts:
      - type: file
        path: "prompts/roles/expert.j2"  # Relative to config.yml

# agents/specialist.yml
system_prompts:
  - type: file
    path: "../prompts/styles/formal.j2"  # Relative to specialist.yml
```

## Best Practices

1. **Use String Shortcuts**: For simple, static prompts, use string shortcuts
2. **File Templates**: Use file-based prompts for complex, reusable templates
3. **Library References**: Create a prompt library for team consistency
4. **Function Generation**: Use functions for dynamic, context-aware prompts
5. **Mixed Approaches**: Combine different types as needed
6. **Clear Organization**: Separate prompts into logical files and directories
7. **Version Control**: Include prompt templates in version control
8. **Documentation**: Document template variables and function signatures

## Function Limitations

!!! info "Async Function Support"
    Currently, function-generated prompts only support synchronous functions.
    Async function support is planned for future releases when the agent
    initialization process becomes asynchronous.
