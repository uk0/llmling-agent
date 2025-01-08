# Agent Injection and Execution Patterns

LLMling-agent provides a clean, pytest-inspired way to work with agents. Instead of complex decorators
or string-based configurations, agents are automatically injected as function parameters.

## Basic Agent Injection

Agents defined in your YAML configuration are automatically available as function parameters:

```python
async def analyze_code(
    analyzer: Agent[Any],
    reviewer: Agent[Any],
    code: str,
) -> CodeAnalysis:
    """Analyze and review code using two agents."""
    analysis = await analyzer.run(f"Analyze this code:\n{code}")
    review = await reviewer.run(f"Review this analysis:\n{analysis.content}")
    return CodeAnalysis(analysis=analysis.content, review=review.content)
```

The function receives fully configured agents matching the parameter names from your YAML:

```yaml
agents:
  analyzer:
    model: gpt-4
    system_prompts:
      - "You are an expert code analyzer..."
  reviewer:
    model: gpt-4
    system_prompts:
      - "You are an expert code reviewer..."
```

## Function Execution Patterns

### Basic Function

The `agent_function` decorator marks functions for automatic execution and dependency handling:

```python
@agent_function
async def simple_task(
    analyst: Agent[Any],
    data: str,
) -> str:
    return await analyst.run(f"Analyze: {data}")
```

### Sequential Dependencies

Functions can depend on the results of other functions:

```python
@agent_function(order=1)  # Runs first
async def gather_data(
    researcher: Agent[Any],
    topic: str,
) -> str:
    return await researcher.run(f"Research: {topic}")

@agent_function(depends_on="gather_data")  # Uses gather_data's result
async def analyze_data(
    analyst: Agent[Any],
    gather_data: str,  # Result from previous function
) -> str:
    return await analyst.run(f"Analyze this:\n{gather_data}")
```

### Parallel Execution

Functions without dependencies can run in parallel:

```python
@agent_function
async def expert1_review(
    expert1: Agent[Any],
    document: str,
) -> str:
    return await expert1.run(f"Review: {document}")

@agent_function
async def expert2_review(
    expert2: Agent[Any],
    document: str,
) -> str:
    return await expert2.run(f"Review: {document}")

# Execute both reviews in parallel:
results = await execute_functions(
    [expert1_review, expert2_review],
    pool=pool,
    inputs={"document": "..."},
    parallel=True,
)
```

### Worker Pattern

Register agents as tools for other agents:

```python
@agent_function
async def improve_code(
    manager: Agent[Any],
    formatter: Agent[Any],
    type_checker: Agent[Any],
    code: str,
) -> str:
    # Register specialists as tools for manager
    manager.register_worker(formatter)
    manager.register_worker(type_checker)
    return await manager.run(f"Improve this code:\n{code}")
```

### Controlled Execution

Use AgentGroup for interactive or rule-based agent interaction:

```python
@agent_function
async def collaborative_task(
    coordinator: Agent[Any],
    specialist1: Agent[Any],
    specialist2: Agent[Any],
    task: str,
) -> str:
    team = AgentGroup([coordinator, specialist1, specialist2])
    results = await team.run_controlled(
        prompt=f"Solve: {task}",
        decision_callback=my_router,
    )
    return results[-1].content
```

### Continuous Monitoring

Set up agents for continuous operation:

```python
@agent_function
async def monitor_system(
    watcher: Agent[Any],
    alerter: Agent[Any],
):
    await watcher.run_continuous(
        "Check system status",
        interval=300,  # every 5 minutes
        max_count=None,  # run indefinitely
    )
    watcher.pass_results_to(alerter)
```

## Error Handling

The system provides clear error messages for common issues:

```python
# Missing agent in pool
@agent_function
async def missing_agent(
    nonexistent: Agent[Any],
) -> str:
    ...  # Raises: AgentInjectionError: No agent named 'nonexistent' found in pool

# Duplicate agent parameter
@agent_function
async def duplicate_agent(
    analyst: Agent[Any],
) -> str:
    ...
# This raises: AgentInjectionError: Cannot inject agent 'analyst': Parameter already provided
result = await duplicate_agent(analyst=some_agent)
```

## Tips and Best Practices

1. **Type Hints**: Always use `Agent[Any]` or appropriate generic type for proper typing.

2. **Default Values**: Use `Agent[Any] | None = None` when agent is optional:
```python
async def optional_review(
    reviewer: Agent[Any] | None = None,
    text: str,
) -> str:
    if reviewer:
        return await reviewer.run(f"Review: {text}")
    return text
```

3. **Pool Access**: You can also inject the pool directly:
```python
async def dynamic_team(
    pool: AgentPool,
    task: str,
):
    team = [pool.get_agent(name) for name in ["agent1", "agent2"]]
    group = AgentGroup(team)
    return await group.run_parallel(task)
```

4. **Context Sharing**: Use shared dependencies for coordinated agents:
```python
async def shared_analysis(
    analyzer1: Agent[Context],
    analyzer2: Agent[Context],
    context: Context,
):
    group = AgentGroup[Context](
        [analyzer1, analyzer2],
        shared_deps=context,
    )
    return await group.run_parallel("Analyze using shared context")
```

## Connection to Manifest

The connection between your YAML manifest and the injection system is made through the AgentPool:

```python
from llmling_agent import AgentPool, AgentsManifest

# 1. Define your agents in YAML
manifest = """
agents:
  researcher:
    model: gpt-4
    system_prompts: ["You are an expert researcher..."]
  writer:
    model: gpt-4
    system_prompts: ["You are an expert writer..."]
"""

# 2. Create pool from manifest
async def main():
    async with AgentPool.open(manifest) as pool:
        # 3. Connect injection system via decorator
        @with_agents(pool)
        async def research_and_write(
            researcher: Agent[Any],  # Will get "researcher" from pool
            writer: Agent[Any],      # Will get "writer" from pool
            topic: str,
        ) -> str:
            research = await researcher.run(f"Research: {topic}")
            return await writer.run(f"Write about:\n{research.content}")

        # 4. Use the function - agents are automatically injected
        result = await research_and_write(topic="quantum computing")
```

The key is that the `with_agents` decorator needs a pool, which is your connection to the manifest. This design:
1. Keeps configuration in YAML (easy to edit/version)
2. Provides clean dependency injection in code
3. Allows flexible pool management strategies
4. Maintains type safety throughout

For more examples and detailed API documentation, see the [API Reference](api_reference.md).
