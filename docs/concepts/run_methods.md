# Run Interface

The run interface provides a consistent way to interact with all message handlers (Agents, Teams, TeamRuns) in llmling-agent.
It serves both as a messaging protocol and a public API.

## Core Methods

### run()

Executes a prompt and returns a single ChatMessage:

```python

msg = await agent.run("analyze this")
msg = await team.run("analyze this")  # parallel execution
msg = await team_run.run("analyze this")  # sequential chain
```

### run_sync()

Synchronous convenience wrapper for `run()`:

```python
# Useful in sync contexts or notebooks
msg = agent.run_sync("analyze this")
msg = team.run_sync("analyze this")
msg = team_run.run_sync("analyze this")
```

### run_in_background()

Start execution in background and monitor progress:

```python
# Start execution
stats = await agent.run_in_background(
    "analyze this",
    max_count: int | None = None,  # Max number of runs
    interval: float = 1.0,  # Seconds between runs
    block: bool = False,  # Whether to block until completion
)

# Monitor execution
while agent.is_running:
    print(f"Messages processed: {stats.message_count}")
    await asyncio.sleep(1)

# Cancel if needed
await agent.cancel()
```

### run_iter()

Asynchronously yields ChatMessages:

```python
async for msg in agent.run_iter(
    "analyze this",
    store_history=True,
    model="gpt-4",
):
    print(msg.content)
```

### run_job()

Executes a predefined job:

```python
result = await agent.run_job(
    job,
    store_history: bool = True,
    include_agent_tools: bool = True,  # Keep agent's tools alongside job tools
)
```

### run_stream()

Stream responses (supported by Agents and TeamRuns):

```python
async with agent.run_stream(
    "analyze",
    store_history=True,
    model="gpt-4",
) as stream:
    async for chunk in stream.stream():
        print(chunk)
```

Note: Parallel Teams don't support streaming as it wouldn't provide any benefit over run_iter().

## Advanced Usage

For advanced use cases requiring detailed execution information, stats tracking, or message flow intervention,
TeamRun provides an additional `execute_iter()` method:

```python
async for item in team_run.execute_iter("analyze"):
    match item:
        case Talk():
            print(f"Connection: {item.source.name} -> {item.targets[0].name}")
        case AgentResponse():
            print(f"Response from {item.agent_name}: {item.message.content}")
```

## Summary

The run interface provides:

- Consistent interaction patterns across all message handlers
- Synchronous and asynchronous execution options
- Background execution with monitoring
- Streaming and iteration capabilities
- Support for both simple and complex use cases
