# System Prompts

System prompts are a crucial part of agent configuration, defining the agent's role, behavior, and capabilities. LLMling provides flexible ways to manage and format system prompts.

## Basic Usage

The simplest way is to provide string prompts:

```python
agent = Agent(
    name="helper",
    system_prompt="You are a helpful assistant.",
)
```

Multiple prompts are concatenated with proper spacing:

```python
agent = Agent(
    system_prompt=[
        "You are a helpful assistant.",
        "You always provide concise answers.",
    ]
)
```

## Dynamic System Prompts

System prompts can be dynamic, either evaluating once on first run or for each interaction:

```python
from llmling_agent import Agent

async def get_weather_context():
    weather = await fetch_weather()
    return f"Current weather: {weather}"

agent = Agent(
    name="weather_advisor",
    system_prompt=[
        "You are a weather advisor.",
        get_weather_context  # Updates each run
    ]
)
agent.sys_prompts.dynamic = True  # Already default
```

## Structured Prompts

You can use Pydantic models to create structured prompts:

```python
from pydantic import BaseModel

class PiratePersonality(BaseModel):
    behavior: str = "You are a fearsome pirate captain."
    goals: str = "Find treasure and command your crew."
    style: str = "Speak in pirate dialect, use nautical terms."

agent = Agent(
    system_prompt=PiratePersonality()
)
```

!!! info
    You can basically use any structured context or "dataclass-ish" objects
    from stdlib as well as dataclass-equivalents of many libraries as prompts.
    This also applies to the Agent.run() methods.

## Tool Integration

System prompts can include information about available tools:

```python
agent = Agent(
    name="log_analyzer",
    system_prompt=["Analyze system logs for issues"],
    tools=[read_logs, analyze_logs, report_issue]
)

# Make tools part of agent's core identity
agent.sys_prompts.inject_tools = "all"  # Include all enabled tools
agent.sys_prompts.tool_usage_style = "suggestive"  # or "strict"
```

This will automatically include tool descriptions in the system prompt:
```
You are log_analyzer. Analyze system logs for issues.

You have access to these tools:

- read_logs: Read system log files
- analyze_logs: Analyze logs for patterns
- report_issue: Create issue report

Use them when appropriate to complete your tasks.
```

For strict enforcement:
```python
agent.sys_prompts.inject_tools = "required"
agent.sys_prompts.tool_usage_style = "strict"
```

This changes the tone:
```
You MUST use these tools to complete your tasks:

- read_logs: Read system log files
- analyze_logs: Analyze logs for patterns
- report_issue: Create issue report

Do not attempt to perform tasks without using appropriate tools.
```


### 3. Temporary System prompts

For temporary system prompt changes, The system prompt manager provides an async context manager:

```python
# Temporarily use a different system prompt
with agent.sys_prompts.temporary_prompt(prompt):
    # temporary prompt additional to agent's prompts
    ...

# Temporarily add sys_prompts and disable all others
with agent.sys_prompts.temporary_prompt(prompt, exclusive=True):
    # Only prompt is used here
    ...

# Original prompts are restored after context exit
```

## Caching

Dynamic prompts can be cached to avoid unnecessary re-evaluation:

```python
# Cache after first evaluation
agent.sys_prompts.dynamic = False

# Always re-evaluate (default)
agent.sys_prompts.dynamic = True
```

## Agent Info Injection

System prompts can automatically include agent identity:

```python
agent = Agent(
    name="analyst",
    description="Expert in data analysis",
    system_prompt=["Analyze data thoroughly"]
)

# Control agent info injection
agent.sys_prompts.inject_agent_info = True  # Default
agent.sys_prompts.inject_agent_info = False  # Disable
```

## Custom Templates

While rarely needed, you can customize the complete template used to generate the system prompt.

```python
agent.sys_prompts.template = custom_template
```

In the global spacename, you will have

- the `agent`
- the `prompts` in form of `AnyPromptType`.
- `to_prompt` helper (AnyPromptType -> str)
- the `dynamic` setting (bool)
- the `inject_tools` setting (bool)
- the `tool_usage_style` setting (Literal["suggestive", "strict"])

For further information, check out agent/sys_prompts.py in the codebase.

!!! info "About Prompt Engineering"
    By default, LLMling-Agent does not engage in prompt engineering or manipulation. The features described
    above (tool injection, strict mode, etc.) are strictly opt-in. We believe in transparency and
    explicit control - any modifications to system prompts are clearly visible and configurable.
    We do include agent name and description though because we consider this essential
    for proper coordination and context. You can disable this behavior by setting `inject_agent_info=False`.

## Prompt Library

LLMling includes a library of pre-defined system prompts that can be used across agents. These prompts are organized by type:

### Prompt Types

- `role`: Defines WHO the agent is (identity, expertise, personality)
- `methodology`: Defines HOW the agent approaches tasks (process, methods)
- `tone`: Defines the STYLE of communication (language, attitude)
- `format`: Defines OUTPUT STRUCTURE (not content)

### Using Library Prompts

You can reference prompts from the library:

```python
# Add a single library prompt by reference
agent.sys_prompts.add("step_by_step")

# In YAML configuration:
agents:
  my_agent:
    model: gpt-4
    system_prompts:    # direct prompts
      - "You are a helpful assistant"
    library_system_prompts:      # reference library prompts
      - step_by_step
      - professional
```

### Defining Library Prompts

```yaml
prompts:
  system_prompts:
    expert_analyst:
      content: |
        You are an expert data analyst.
        Focus on finding patterns and insights in data.
        Always provide evidence for your conclusions.
      type: role

    step_by_step:
      content: |
        Break tasks into clear, sequential steps.
        For each step:
        1. Explain what to do
        2. Note important considerations
        3. Define completion criteria
      type: methodology
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

### Available Prompts

By default, INHERIT is set to builtin prompt library with a few silly prompts to get started.
These can all be referenced by name without any further configuration.

Roles:

- `technical_writer`: Expert in clear, precise documentation
- `code_reviewer`: Expert code analysis and feedback
- `rubber_duck`: Debugging assistant with personality

Methodologies:

- `step_by_step`: Break tasks into clear sequences
- `minimalist`: Concise, essential responses
- `detailed`: Comprehensive coverage of topics

Tones:

- `professional`: Formal business communication
- `pirate`: Maritime flavor (fun!)
- `shakespearean`: Classical, poetic style

Formats:

- `markdown`: Structured Markdown formatting
