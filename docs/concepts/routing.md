# Message Routing System

## Overview

LLMling's message routing system provides a flexible and powerful way to control how messages flow between agents. Unlike simple point-to-point connections, the routing system allows for sophisticated decision-making based on message content, agent state, and historical statistics.


## Basic Concepts

At its core, routing is handled through filter functions that determine whether a message should be forwarded to a specific target. These functions can consider:

- Message content and metadata
- Target agent's capabilities and state
- Historical interaction statistics
- System-wide metrics

## Filter Functions

```python
# Simple message-based routing
agent >> other_agent.when(
    lambda msg: "urgent" in str(msg.content)
)

# Target-aware routing
agent >> other_agent.when(
    lambda msg, target: (
        "code" in str(msg.content) and
        not target.is_busy()
    )
)

# Full context routing (message, target, stats)
agent >> other_agent.when(
    lambda msg, target, stats: (
        "database" in str(msg.content) and
        target.capabilities.can_execute_code and
        stats.message_count < 10
    )
)
```

### Routing Decisions

Filter functions can make routing decisions based on:

1. **Message Content and Metadata**
   - Content analysis
   - Message role
   - Cost information
   - Custom metadata

2. **Target Agent State**
   - Current workload
   - Capabilities
   - Model type
   - Connection status

3. **Connection Statistics**
   - Message count
   - Token usage
   - Historical interaction
   - Timing information

### Advanced Routing Patterns

#### Load Balancing
```python
def least_busy(msg: ChatMessage, target: AnyAgent):
    return not target.is_busy()

source >> team.when(least_busy)
```


#### Cost-Aware Routing
```python
def within_budget(msg: ChatMessage, target: AnyAgent, stats):
    return stats.total_cost < 1.0  # $1 limit

source >> target.when(within_budget)
```

### Async Support

Filter functions can be either synchronous or asynchronous:

```python
# Sync filter
agent >> other.when(lambda msg: "urgent" in str(msg.content))

# Async filter
async def check_database(msg: ChatMessage, target: AnyAgent):
    result = await db.check_priority(msg.content)
    return result == "high"

agent >> other.when(check_database)
```

### Combining Conditions

Multiple conditions can be combined using regular Python logic:

```python
def complex_route(msg: ChatMessage, target: AnyAgent, stats: TalkStats):
    return (
        # Content-based
        any(topic in str(msg.content) for topic in ["urgent", "critical"]) and
        # Target-based
        target.capabilities.can_execute_code and
        not target.is_busy() and
        # Stats-based
        stats.message_count < 100 and
        stats.total_cost < 5.0
    )

source >> target.when(complex_route)
```

### Error Handling

Filter functions should handle their own exceptions. Any unhandled exceptions will prevent message routing:

```python
def safe_route(msg: ChatMessage, target: AnyAgent):
    try:
        return complex_check(msg.content)
    except Exception as e:
        logger.warning("Routing check failed: %s", e)
        return False

source >> target.when(safe_route)
```

This routing system provides:

- Flexible decision-making
- Access to relevant context
- Type safety through signature checking
- Async support
- Clean, functional interface
