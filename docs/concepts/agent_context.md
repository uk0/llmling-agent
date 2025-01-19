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

# Create context with typed deps
context = AgentContext[AppConfig].create_default(
    name="my-agent",
    deps=app_config
)

# Type-safe access in tools
async def call_api(ctx: RunContext[AgentContext[AppConfig]], query: str) -> str:
    api_key = ctx.deps.data.api_key  # Type-safe access
    endpoint = ctx.deps.data.endpoint
    return await make_request(endpoint, api_key, query)
```

## Tool Context (Current Implementation)

Currently, tools receive a nested context structure:

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

    # Create teams
    team = context.pool.create_team(["analyzer", "planner"])

    # Access storage
    await context.pool.storage.log_message(...)
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

## Creating Contexts

### Default Context
```python
# Create minimal context
context = AgentContext[None].create_default(
    name="my-agent"
)

# With capabilities
context = AgentContext[AppConfig].create_default(
    name="my-agent",
    capabilities=Capabilities(can_delegate_tasks=True),
    deps=app_config,
    pool=pool
)
```

### From Configuration
```python
# Create from agent config
context = AgentContext[AppConfig](
    agent_name="my-agent",
    capabilities=config.capabilities,
    definition=manifest,
    config=agent_config,
    data=app_config
)
```

## Best Practices

### Dependency Management
- Use typed dependencies for type safety
- Keep dependencies immutable
- Document required dependency structure

### Tool Implementation
- Always type context parameter
- Check required capabilities
- Handle missing pool/runtime gracefully

### Error Handling
```python
if not context.pool:
    raise RuntimeError("Pool required for this operation")

if not context.runtime:
    raise RuntimeError("Runtime required for this operation")
```

AgentContext provides a central point for agent configuration, capabilities, and pool integration, though its interface is planned for improvement in future versions.
