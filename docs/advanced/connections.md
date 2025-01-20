# Agent Connection System

## Overview
LLMling provides a robust, object-oriented approach to managing agent communications through dedicated connection objects. The system supports various connection patterns and offers fine-grained control over message flow and monitoring. Connections can be defined both programmatically and through YAML configuration.

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
        transform: Callable[[Any], Any | Awaitable[Any]] | None = None,
        filter_condition: AnyFilterFn | None = None,
        stop_condition: AnyFilterFn | None = None,
        exit_condition: AnyFilterFn | None = None,
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

## YAML Configuration

Connections can be defined in agent configuration:

```yaml
agents:
  analyzer:
    # ... other config ...
    connections:
      - type: agent
        name: planner
        filter_condition:
          type: word_match
          words: ["analyze", "examine"]
        stop_condition:
          type: message_count
          max_messages: 5
        transform: myapp.transforms.process_message
```

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

In this scenario each team member of team_a gets connected to all team members of team b.

## Message Flow Control

### Statistics and Monitoring

Each connection tracks:

- Message count
- Token usage
- Byte count
- Timing information

### Control Mechanisms

1. **Message Filtering**:
   ```python
   # Using lambda
   talk.when(lambda msg: "important" in msg.content)

   # Using YAML
   filter_condition:
     type: word_match
     words: ["important"]
   ```

2. **Connection Control**:
   - `stop_condition`: Disconnect this connection
   - `exit_condition`: Exit the entire process (raises SystemExit)
   - `transform`: Modify messages as they flow

3. **Flow Control**:
   - Priority-based handling
   - Delayed execution
   - Message queuing

```python
# Set up connection with control
talk = agent.pass_results_to(
    target,
    priority=1,
    delay=timedelta(seconds=5),
    stop_condition=lambda msg: msg.content == "STOP",
    exit_condition=lambda msg: msg.content == "EXIT"
)
```

### Team Management

TeamTalk provides aggregate operations for multiple connections:

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

6. **Configuration Options**
   - Programmatic setup
   - YAML configuration
   - Runtime modification

## Example Usage

```python
# Create agents
analyzer = Agent(name="analyzer")
planner = Agent(name="planner")
executor = Agent(name="executor")

# Create team
planning_team = planner & executor

# Set up connection with control
talk = analyzer.pass_results_to(
    planning_team,
    connection_type="run",
    transform=lambda msg: preprocess_message(msg),
    stop_condition=lambda msg: msg.metadata.get("complete", False),
    exit_condition=lambda msg: msg.metadata.get("error", False)
)

talk.when(lambda msg: msg.metadata.get("priority") == "high")

# Monitor
print(f"Processed {talk.stats.message_count} messages")
```
