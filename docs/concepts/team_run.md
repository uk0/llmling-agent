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

### Controlled
Execution flow is determined by routing decisions:
```python
execution = team.monitored("controlled")
results = await execution.run(
    "Handle this task",
    decision_callback=interactive_controller
)
# Flow determined by decisions
# Can branch and loop
# Results track execution path
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
Start run in background with monitoring:
```python
# Start run
run.start_background(
    "Process this task",
    monitor_callback=on_update
)

# Monitor progress
async def on_update(stats: TeamRunStats):
    print(f"Active agents: {stats.active_agents}")
    print(f"Messages: {stats.message_counts}")
    print(f"Tokens used: {stats.total_tokens}")
    print(f"Cost so far: ${stats.total_cost:.4f}")

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
