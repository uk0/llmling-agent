# Agent Context

## Overview

AgentContext serves multiple purposes:

1. Holds agent-specific configuration and state
2. Provides access to shared resources (pool, storage)
3. Acts as a bridge between agents and the pool
4. Carries typed dependencies for agent operations
5. Enables tool validation and execution control

```python
class AgentContext[TDeps]:
    """Runtime context for agent execution.

    Generically typed with AgentContext[Type of Dependencies]
    """

    agent_name: str
    """Name of the current agent"""

    capabilities: Capabilities
    """Agent's capability configuration"""

    definition: AgentsManifest
    """Complete agent configuration"""

    data: TDeps | None
    """Typed dependencies for this agent"""

    pool: AgentPool | None
    """Reference to parent pool"""

    runtime: RuntimeConfig | None
    """Access to resources and tools of LLMLing Runtime."""
```

## Type-Safe Dependencies

AgentContext enables type-safe dependency injection:

```python
# Define dependencies
class AppConfig:
    api_key: str
    endpoint: str

my_agent(deps=AppConfig())
```


## Tool Context (Current Implementation)

Currently, tools receive a nested context structure. This more has historical reasons,
so be aware that this is subject to change.


```python
# Tool receives PydanticAI context containing our context
async def my_tool(ctx: RunContext[AgentContext[TDeps]], arg: str) -> str:
    # Access through PydanticAI context
    agent_ctx = ctx.deps

    # Access pool
    pool = agent_ctx.pool
    other_agent = pool.get_agent("helper")

    # Access capabilities
    if agent_ctx.capabilities.can_delegate_tasks:
        await other_agent.run(arg)
```

!!! note "Future Changes"
    The nested context structure is planned for revision to provide a more
    direct and cleaner interface for tool implementations.

## Pool Integration

AgentContext acts as a bridge to the agent pool:

```python
# Access pool functionality
if context.pool:
    # Get other agents
    helper = context.pool.get_agent("helper")
```

## Tool Validation

Context enables tool validation through capabilities:

```python
# Check capability
if context.capabilities.can_execute_code:
    # Execute code...
else:
    raise ToolError("Missing required capability")

# Validate tool requirements
async def handle_confirmation(
    ctx: AgentContext,
    tool: ToolInfo,
    args: dict[str, Any]
) -> str:
    """Handle tool execution confirmation."""
    if tool.requires_capability:
        if not ctx.capabilities.has_capability(tool.requires_capability):
            return "skip"  # Tool not allowed
    return "allow"
```

## Storage Access

Provides access to the storage system:

```python
# Direct storage access
await context.storage.log_message(...)
await context.storage.log_tool_call(...)

# Through pool's storage
if context.pool:
    store = context.pool.storage
    await store.log_conversation(...)
```
