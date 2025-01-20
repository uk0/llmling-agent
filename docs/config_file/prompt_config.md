# Prompt Library Configuration

## Overview
LLMling's prompt library allows defining reusable prompts that can be shared across agents.
Prompts are defined in the `prompts` section of your configuration and can be referenced by name.

## Basic Structure

```yaml
prompts:
  # Main system prompts for defining agent behavior
  system_prompts:
    expert_analyst:
      content: |
        You are an expert data analyst.
        Focus on finding patterns and insights.
      type: role

    step_by_step:
      content: |
        Break tasks into sequential steps.
        Explain each step thoroughly.
      type: methodology

# Using prompts in agents
agents:
  analyst:
    system_prompts:           # Direct prompts
      - "You help with analysis."
    library_system_prompts:   # Reference library prompts
      - expert_analyst
      - step_by_step
```

## Prompt Categories

### System Prompts
Define core agent behaviors and methodologies:

```yaml
prompts:
  system_prompts:
    # Role definition
    technical_writer:
      content: |
        You are an expert technical writer.
        Focus on clarity and precision.
        Use proper terminology consistently.
      type: role

    # Methodology definition
    step_by_step:
      content: |
        Follow these steps for each task:
        1. Understand requirements
        2. Plan approach
        3. Execute systematically
        4. Verify results
      type: methodology

    # Tone/style definition
    professional:
      content: |
        Maintain formal, business-appropriate language.
        Be concise but thorough.
      type: tone
```

## Prompt Types

The `type` field categorizes prompts by their purpose:

### Role Types
Define WHO the agent is:
```yaml
system_prompts:
  expert_dev:
    type: role
    content: |
      You are a senior software developer...

  data_scientist:
    type: role
    content: |
      You specialize in data analysis...
```

### Methodology Types
Define HOW the agent works:
```yaml
system_prompts:
  analytical:
    type: methodology
    content: |
      Approach problems systematically:
      1. Gather data
      2. Analyze patterns
      3. Form conclusions

  iterative:
    type: methodology
    content: |
      Work in small iterations...
```

### Tone Types
Define communication STYLE:
```yaml
system_prompts:
  formal:
    type: tone
    content: |
      Use professional language...

  friendly:
    type: tone
    content: |
      Be approachable and helpful...
```

### Format Types
Define output STRUCTURE:
```yaml
system_prompts:
  markdown:
    type: format
    content: |
      Format responses using Markdown:
      - Use headers for sections
      - Use lists for items
      - Use code blocks for code
```

## Using Library Prompts

Reference prompts in agent configuration:

```yaml
agents:
  technical_assistant:
    model: gpt-4
    # Direct prompts
    system_prompts:
      - "You are a technical assistant."
      - "Focus on helping with code."

    # Library prompts
    library_system_prompts:
      - technical_writer    # Role
      - step_by_step       # Methodology
      - professional       # Tone
      - markdown          # Format

  data_analyst:
    library_system_prompts:
      - expert_analyst
      - analytical
      - formal
```

## Complete Example

```yaml
prompts:
  system_prompts:
    # Roles
    technical_expert:
      type: role
      content: |
        You are a technical expert specializing in:
        - Software development

    code_reviewer:
      type: role
      content: |
        You are an experienced code reviewer.
        Focus on:
        - Code quality

    # Methodologies
    systematic:
      type: methodology
      content: |
        Follow this systematic approach:
        1. Understand requirements fully

    # Tones
    professional:
      type: tone
      content: |
        Maintain professional communication:
        - Use formal language
        ...

    # Formats
    structured:
      type: format
      content: |
        Structure responses with:
        1. Clear headings
        2. Bulleted lists
        3. Code examples
        4. Summary points

agents:
  senior_dev:
    model: gpt-4
    description: "Senior developer specialized in code review and optimization"
    system_prompts:
      - "Specialize in Python and TypeScript."
    library_system_prompts:
      - technical_expert
      - code_reviewer
      - systematic
      - professional
      - structured

  teacher:
    model: gpt-4
    description: "Programming teacher and mentor"
    library_system_prompts:
      - technical_expert
      - iterative
      - educational
```


!!! tip "Organizing Prompts"
    It's recommended to keep prompt libraries in separate files and use YAML inheritance
    to include them. This keeps your agent configurations clean and promotes reuse:
    ```yaml
    # prompts.yml
    prompts:
      system_prompts:
        my_prompt:
          content: ...
          type: role

    # agents.yml
    INHERIT: prompts.yml
    agents:
      my_agent:
        library_system_prompts:
          - my_prompt
    ```

The integration of this functionality will get improved soon!
