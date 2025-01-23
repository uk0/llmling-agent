# Team Runs

## What is a TeamRun?

A TeamRun represents the orchestrated execution of a prompt by a group of agents. Unlike simple agent groups (Teams) which just define membership, TeamRuns manage how agents work together - particularly their execution order and interaction patterns.

The key concept is that a TeamRun determines:

- Which agents participate in the execution
- In what order they process the input
- How they interact with each other
- How the execution can be monitored

## Creating TeamRuns

### From Agent Groups
```python
# Create team
team = pool.create_team(["analyzer", "planner", "executor"])

# Create observable execution object with specific mode
execution = team.monitored("sequential")
# or
run = TeamRun(team, mode="sequential")
```

### Using Pipeline Operator
```python
# Create sequential pipeline
pipeline = analyzer | planner | executor

# With function transformers
pipeline = analyzer | clean_data | planner | execute_plan

# Mixing teams
pipeline = analysis_team | planning_team | execution_team
```

## Execution Modes

### Parallel
All agents process the input simultaneously:
```python
execution = team.monitored("parallel")
results = await execution.run("Analyze this data")
# All agents receive same input
# No guaranteed order
# Results contain all responses
```

### Sequential
Agents process in order, each receiving previous agent's output:
```python
execution = team.monitored("sequential")
results = await execution.run("Process this")
# analyzer -> planner -> executor
# Each agent receives previous output
# Results contain all intermediate steps
```

## Running a TeamRun

### Direct Execution
Simple run that waits for completion:
```python
# Run and wait for results
results = await run.run("Process this task")

# Access results
for response in results:
    print(f"{response.agent_name}: {response.message.content}")
```

### Monitored Runs
Get stats while run is executing in background:
```python
# Start run and get stats object
stats = run.run_in_background("Process this task")

# Monitor progress by polling stats
while run.is_running:
    # Access stats through TeamTalk interface
    print(f"Active connections: {len(stats)}")
    for talk in stats:
        print(f"\nConnection: {talk.source_name} -> {talk.target_names}")
        print(f"Messages: {len(talk.stats.messages)}")
        print(f"Tool calls: {len(talk.stats.tool_calls)}")
    if stats.errors:
        print("\nErrors:")
        for agent, error, time in stats.errors:
            print(f"  {agent}: {error} at {time}")
    await asyncio.sleep(0.5)

# Wait for completion when needed
results = await run.wait()
```

## TeamRun Statistics

TeamRuns provide detailed run statistics:
```python
stats = run.stats

# Activity monitoring
print(f"Active agents: {stats.active_agents}")
print(f"Duration: {stats.duration:.2f}s")

# Message tracking
print(f"Messages per agent: {stats.message_counts}")
print(f"Total tokens: {stats.total_tokens}")
print(f"Total cost: ${stats.total_cost:.4f}")

# Tool usage
print(f"Tools per agent: {stats.tool_counts}")

# Error tracking
if stats.has_errors:
    for agent, error, time in stats.error_log:
        print(f"{agent} failed at {time}: {error}")
```

## Monitoring Execution Flow

For sequential executions, you can monitor and control the execution flow using `run_iter()`.
This yields items in alternating order:
- Connection objects (`Talk`) before they're used
- Responses (`AgentResponse`) after each agent executes

This allows you to configure routing before messages flow through connections and monitor the results.

Here are common usage patterns:

### Basic Monitoring
Simple progress tracking of the execution chain:
```python
execution = team.monitored("sequential")
async for item in execution.run_iter("analyze this"):
    match item:
        case Talk():
            print(f"Next connection: {item.source.name} -> {item.target.name}")
        case AgentResponse():
            print(f"Got response from {item.agent_name}: {item.data}")
```

### Error Handling
Stop execution when an agent fails:
```python
execution = team.monitored("sequential")
async for item in execution.run_iter("analyze"):
    if isinstance(item, AgentResponse):
        if not item.success:
            print(f"Chain failed at {item.agent_name}: {item.error}")
            break
        print(f"✅ {item.agent_name}")
```

### Message Transformation
Modify messages before they're forwarded:
```python
execution = team.monitored("sequential")
async for item in execution.run_iter("analyze"):
    if isinstance(item, Talk):
        # Configure connection before it's used
        item.transform = lambda msg: f"Previous: {msg.content}"
```

### Progress Tracking
Integration with progress bars:
```python
from rich.progress import Progress
async def track_progress():
    with Progress() as progress:
        task = progress.add_task("Processing...", total=len(team.agents))
        async for item in execution.run_iter("analyze"):
            if isinstance(item, AgentResponse):
                progress.advance(task)
```

The alternating pattern of connection->response makes it clear when you can configure routing and when you receive results.

## Advanced Features

### Custom Monitoring
```python
# Monitor specific aspects
run.monitor(
    lambda stats: print(f"Tokens: {stats.total_tokens}"),
    interval=0.5  # Check every 500ms
)

# Multiple monitors
run.monitor(cost_tracker)
run.monitor(progress_ui)
run.monitor(error_logger)
```

### Run Control
```python
# Cancel run
await run.cancel()

# Clean up resources
await run.cleanup()

# Check status
if run.is_running:
    print("Still processing...")
```

TeamRuns provide a flexible way to orchestrate multiple agents while maintaining visibility into their execution.
The combination of different run modes and monitoring capabilities makes them suitable for both simple pipelines and complex multi-agent interactions.
