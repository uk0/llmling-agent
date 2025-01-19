# Agent Connection System

## Overview

LLMling provides a robust, object-oriented approach to managing agent communications through dedicated connection objects. The system supports various connection patterns and offers fine-grained control over message flow and monitoring.

## Core Components

### Talk

The fundamental connection unit representing a one-to-many relationship between agents:

```python
class Talk:
    def __init__(
        self,
        source: AnyAgent[Any, Any],
        targets: list[AnyAgent[Any, Any]],
        connection_type: ConnectionType = "run",
        priority: int = 0,
        delay: timedelta | None = None,
    )
```

### Connection Types

Three different ways messages can be handled:

- `run`: Execute message as a new run in target agent
- `context`: Add message as context to target's conversation
- `forward`: Forward message directly to target's outbox

### Connection Management

Connections are managed by the `TalkManager`, which provides:

- Connection creation and cleanup
- Message routing
- Wait state management
- Statistics tracking

## Connection Patterns

### Agent-to-Agent

Simple connection between two agents:
```python
# Direct connection
agent_a.pass_results_to(agent_b)

# Named connection (using pool)
agent_a.pass_results_to("agent_b")
```

### Agent-to-Team

Connect an agent to multiple targets:
```python
# Create team
team = agent_b & agent_c & agent_d

# Connect agent to team
agent_a.pass_results_to(team)
```

### Team-to-Team

Connect groups of agents:
```python
team_a = agent_1 & agent_2
team_b = agent_3 & agent_4
team_a.pass_results_to(team_b)
```

## Message Flow Control

### Statistics and Monitoring

Each connection tracks:

- Message count
- Token usage
- Byte count
- Timing information

```python
@dataclass(frozen=True)
class TalkStats:
    message_count: int
    token_count: int
    byte_count: int
    last_message_time: datetime | None
    source_name: str | None
    target_names: set[str]
```

### Flow Control

Connections support:

- Priority-based message handling
- Delayed execution
- Message filtering
- State tracking (active/inactive)

```python
# Set up connection with control
talk = agent.pass_results_to(
    target,
    priority=1,
    delay=timedelta(seconds=5)
)

# Filter messages
talk.when(lambda msg: "important" in msg.content)
```

### Team Management

If a team is connected to other entities, a TeamTalk object is returned, containing multiple one-to-many connections.
The TeamTalk object provides a similar interface to the Talk object and forwards the method calls to all contained Talk objects.

`TeamTalk` provides aggregate operations for multiple connections:

- Collective statistics
- Group operations (pause/resume)
- Recursive target resolution

## Unique Features

1. **Object-Oriented Design**
   - Each connection is a first-class object
   - Strong typing and clear interfaces
   - Easy to extend and customize

2. **Flexible Routing**
   - Multiple connection types
   - Configurable message handling
   - Support for complex topologies

3. **Rich Monitoring**
   - Detailed statistics
   - Connection state tracking
   - Performance metrics

4. **Type Safety**
   - Generic type parameters for dependencies and results
   - Type-safe message passing
   - Proper typing for team operations

5. **Clean API**
   - Operator overloading for intuitive syntax (`agent >> other_agent`)
   - Chainable configuration
   - Consistent interface across patterns

## Example Usage

```python
# Create agents
analyzer = Agent(name="analyzer")
planner = Agent(name="planner")
executor = Agent(name="executor")

# Create team
planning_team = planner & executor

# Set up connection
talk =analyzer.pass_results_to(planning_team, connection_type="run")

talk.when(lambda msg: msg.metadata.get("priority") == "high")

# Monitor
print(f"Processed {talk.stats.message_count} messages")
