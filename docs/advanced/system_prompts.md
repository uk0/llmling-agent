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
from llmling import Agent
from toprompt import DynamicPrompt

async def get_weather_context():
    weather = await fetch_weather()
    return f"Current weather: {weather}"

agent = Agent(
    name="weather_advisor",
    system_prompt=[
        "You are a weather advisor.",
        DynamicPrompt(get_weather_context)  # Updates each run
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

While rarely needed, you can customize the complete template:

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
