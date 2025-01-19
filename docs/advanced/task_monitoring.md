# Team Execution and Monitoring

The Team class provides different ways to execute tasks across multiple agents. This guide explains both the basic execution methods and the advanced monitoring capabilities.

## Basic Execution Methods

The Team class offers three primary execution modes:

```python
# Simple parallel execution
result = await team.run_parallel("Analyze this text")

# Sequential execution
result = await team.run_sequential("Process in order")

# Controlled execution with routing decisions
result = await team.run_controlled("Handle with decisions")
```

## Monitored Execution

For more complex scenarios where you need to track execution progress, use the monitored interface:

```python
# Create monitored execution
execution = team.monitored("parallel")  # or "sequential"/"controlled"

# Start in background
execution.start_background("Task to execute")

# Monitor progress
execution.monitor(on_stats_update)  # provide callback for updates

# Wait for completion
result = await execution.wait()
```

### The Monitor Object

Similar to how Talk objects manage agent connections, the TeamRunMonitor tracks execution progress through agent signals. It:

- Captures message exchanges
- Records tool usage
- Tracks errors
- Measures timing
- Maintains execution state

This happens asynchronously without blocking the main execution.

### Stats Object

The TeamRunStats object provides a complete view of the execution state:

```python
stats = execution.stats

# Raw data
stats.received_messages  # All messages received by agents
stats.sent_messages     # All messages sent by agents
stats.tool_calls       # All tool calls made
stats.error_log       # Any errors that occurred

# Derived metrics
stats.active_agents   # Currently processing agents
stats.message_counts  # Messages per agent
stats.total_tokens    # Total tokens used
stats.total_cost     # Total cost incurred
stats.duration       # Time elapsed
```

### Control Methods

The execution object provides several control methods:

```python
# Start execution
execution.start_background(prompt)  # Non-blocking start
await execution.start(prompt)       # Direct start

# Monitor progress
execution.monitor(callback)         # Register update callback

# Check state
execution.is_running               # Whether still processing

# Control execution
await execution.wait()            # Wait for completion
await execution.cancel()          # Cancel execution
```

### Example: Progress Monitoring

```python
async def on_stats_update(stats: TeamRunStats):
    print(f"Active agents: {stats.active_agents}")
    print(f"Messages processed: {stats.message_counts}")
    print(f"Tools used: {stats.tool_counts}")

execution = team.monitored("parallel")
execution.start_background("Task")
execution.monitor(on_stats_update)
result = await execution.wait()
```

### Cleanup

All resources (signal connections, background tasks) are automatically cleaned up when:
- Execution completes
- cancel() is called
- An error occurs

### Integration with TaskManagerMixin

The execution system integrates with TaskManagerMixin to provide:
- Proper task management
- Resource cleanup
- Background task tracking
- Error handling

## Summary

The monitoring system provides a flexible way to:
- Track execution progress
- Collect execution metrics
- Monitor agent activity
- Analyze tool usage
- Handle errors
- Control execution flow

All while maintaining asynchronous operation and proper resource management.
