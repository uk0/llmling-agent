# Forward Targets

Forward targets define how agents automatically forward their messages to other agents or destinations. They allow creating agent pipelines and communication patterns.

## Agent Forwarding
Forward messages to another agent:
```yaml
agents:
  analyzer:
    connections:
      - type: "agent"
        name: "summarizer"              # Target agent name
        connection_type: "run"          # How to handle messages
        wait_for_completion: true       # Wait for target to finish
```

## File Output
Save messages to files:
```yaml
agents:
  logger:
    connections:
      - type: "file"
        path: "logs/{date}/{agent}.txt"  # Supports variables
        wait_for_completion: true        # Wait for write to complete
```

## Connection Types
The `connection_type` field supports three modes:

### "run"
Runs the message as a new prompt:
```yaml
agents:
  analyzer:
    connections:
      - type: "agent"
        name: "summarizer"
        connection_type: "run"          # Process as new prompt
```

### "context"
Adds message to target's context:
```yaml
agents:
  researcher:
    connections:
      - type: "agent"
        name: "writer"
        connection_type: "context"      # Add as context
```

### "forward"
Simply forwards the message:
```yaml
agents:
  monitor:
    connections:
      - type: "agent"
        name: "logger"
        connection_type: "forward"      # Pass through
```

## Complex Example
Multiple forward targets with different configurations:
```yaml
agents:
  analyzer:
    connections:
      # Forward to another agent
      - type: "agent"
        name: "summarizer"
        connection_type: "run"
        wait_for_completion: true

      # Save to log file
      - type: "file"
        path: "logs/{date}/analysis_{time}.txt"
        wait_for_completion: false

      # Add to reviewer's context
      - type: "agent"
        name: "reviewer"
        connection_type: "context"
        wait_for_completion: false
```

## File Path Variables
Available variables for file paths:
- `{date}`: Current date (YYYY-MM-DD)
- `{time}`: Current time (HH-MM-SS)
- `{agent}`: Name of the source agent

Example:
```yaml
agents:
  logger:
    connections:
      - type: "file"
        path: "logs/{date}/{agent}_{time}.txt"
```
