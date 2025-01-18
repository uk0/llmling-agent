from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timedelta
import io
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from llmling import ToolError
from pydantic_ai import RunContext  # noqa: TC002

from llmling_agent.models.context import AgentContext  # noqa: TC001


if TYPE_CHECKING:
    from llmling_agent.agent import AnyAgent


async def delegate_to(  # noqa: D417
    ctx: RunContext[AgentContext],
    agent_name: str,
    prompt: str,
) -> str:
    """Delegate a task. Allows to assign a task to task to a specific agent.

    If an action requires you to delegate a task, this tool can be used to assign and
    execute a task. Instructions can be passed via the prompt parameter.

    Args:
        agent_name: The agent to delegate the task to
        prompt: Instructions for the agent to delegate.

    Returns:
        The result of the task you delegated.
    """
    if not ctx.deps.pool:
        msg = "Agent needs to be in a pool to delegate tasks"
        raise ToolError(msg)
    specialist = ctx.deps.pool.get_agent(agent_name)
    result = await specialist.run(prompt)
    return str(result.data)


async def list_available_agents(  # noqa: D417
    ctx: RunContext[AgentContext],
    only_idle: bool = False,
) -> list[str]:
    """List all agents available in the current pool.

    Args:
        only_idle: If True, only returns agents that aren't currently busy.
                    Use this to find agents ready for immediate tasks.

    Returns:
        List of agent names that you can use with delegate_to
    """
    if not ctx.deps.pool:
        msg = "Agent needs to be in a pool to list agents"
        raise ToolError(msg)

    agents = list(ctx.deps.pool.list_agents())
    if only_idle:
        return [n for n in agents if not ctx.deps.pool.get_agent(n).is_busy()]
    return agents


async def create_worker_agent[TDeps](
    ctx: RunContext[AgentContext[TDeps]],
    name: str,
    system_prompt: str,
    model: str | None = None,
) -> str:
    """Create a new agent and register it as a tool.

    The new agent will be available as a tool for delegating specific tasks.
    It inherits the current model unless overridden.
    """
    from llmling_agent import Agent

    if not ctx.deps.pool:
        msg = "Agent needs to be in a pool to list agents"
        raise ToolError(msg)

    model = model or ctx.model.name()
    worker = Agent[TDeps](
        name=name,
        model=model,
        system_prompt=system_prompt,
        context=ctx.deps,
    )
    assert ctx.deps.agent
    tool_info = ctx.deps.agent.register_worker(worker)
    return f"Created worker agent and registered as tool: {tool_info.name}"


async def spawn_delegate[TDeps](
    ctx: RunContext[AgentContext[TDeps]],
    task: str,
    system_prompt: str,
    model: str | None = None,
    capabilities: dict[str, bool] | None = None,
    connect_back: bool = False,
) -> str:
    """Spawn a temporary agent for a specific task.

    Creates an ephemeral agent that will execute the task and clean up automatically
    Optionally connects back to receive results.
    """
    from llmling_agent import Agent

    if not ctx.deps.pool:
        msg = "No agent pool available"
        raise ToolError(msg)

    name = f"delegate_{uuid4().hex[:8]}"
    model = model or ctx.model.name()
    agent = Agent[TDeps](
        name=name,
        model=model,
        system_prompt=system_prompt,
        context=ctx.deps,
    )

    if connect_back:
        assert ctx.deps.agent
        ctx.deps.agent.pass_results_to(agent)

    await agent.run(task)
    return f"Spawned delegate {name} for task"


async def search_history(
    ctx: RunContext[AgentContext],
    query: str | None = None,
    hours: int = 24,
    limit: int = 5,
) -> str:
    """Search conversation history."""
    from llmling_agent_storage.formatters import format_output

    if ctx.deps.capabilities.history_access == "none":
        msg = "No permission to access history"
        raise ToolError(msg)

    provider = ctx.deps.storage.get_history_provider()
    results = await provider.get_filtered_conversations(
        query=query,
        period=f"{hours}h",
        limit=limit,
    )
    return format_output(results)


async def show_statistics(
    ctx: RunContext[AgentContext],
    group_by: Literal["agent", "model", "hour", "day"] = "model",
    hours: int = 24,
) -> str:
    """Show usage statistics for conversations."""
    from llmling_agent_storage.formatters import format_output
    from llmling_agent_storage.models import StatsFilters

    if ctx.deps.capabilities.stats_access == "none":
        msg = "No permission to view statistics"
        raise ToolError(msg)

    cutoff = datetime.now() - timedelta(hours=hours)
    filters = StatsFilters(cutoff=cutoff, group_by=group_by)

    provider = ctx.deps.storage.get_history_provider()
    stats = await provider.get_conversation_stats(filters)

    return format_output(
        {
            "period": f"{hours}h",
            "group_by": group_by,
            "entries": [
                {
                    "name": key,
                    "messages": data["messages"],
                    "total_tokens": data["total_tokens"],
                    "models": sorted(data["models"]),
                }
                for key, data in stats.items()
            ],
        },
        output_format="text",
    )


async def execute_python(ctx: RunContext[AgentContext], code: str) -> str:
    """Execute Python code directly."""
    if not ctx.deps.capabilities.can_execute_code:
        msg = "No permission to execute code"
        raise ToolError(msg)

    try:
        # Capture output
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            exec(code, {"__builtins__": __builtins__})
            return buf.getvalue() or "Code executed successfully"
    except Exception as e:  # noqa: BLE001
        return f"Error executing code: {e}"


async def execute_command(ctx: RunContext[AgentContext], command: str) -> str:
    """Execute a shell command."""
    if not ctx.deps.capabilities.can_execute_commands:
        msg = "No permission to execute commands"
        raise ToolError(msg)

    try:
        pipe = asyncio.subprocess.PIPE
        proc = await asyncio.create_subprocess_shell(command, stdout=pipe, stderr=pipe)
        stdout, stderr = await proc.communicate()
        return stdout.decode() or stderr.decode() or "Command completed"
    except Exception as e:  # noqa: BLE001
        return f"Error executing command: {e}"


async def add_agent(  # noqa: D417
    ctx: RunContext[AgentContext],
    name: str,
    system_prompt: str,
    model: str | None = None,
    tools: list[str] | None = None,
    session: str | None = None,
    result_type: str | None = None,
) -> str:
    """Add a new agent to the pool.

    Args:
        name: Name for the new agent
        system_prompt: System prompt defining agent's role/behavior
        model: Optional model override (uses default if not specified)
        tools: Names of tools to enable for this agent
        session: Session ID to recover conversation state from
        result_type: Name of response type from manifest (for structured output)

    Returns:
        Confirmation message about the created agent
    """
    assert ctx.deps.pool, "No agent pool available"
    agent: AnyAgent[Any, Any] = await ctx.deps.pool.add_agent(
        name=name,
        system_prompt=system_prompt,
        model=model,
        tools=tools,
        result_type=result_type,
        session=session,
    )
    return f"Created agent {agent.name} using model {agent.model_name}"


async def ask_agent(  # noqa: D417
    ctx: RunContext[AgentContext],
    agent_name: str,
    message: str,
    *,
    model: str | None = None,
    store_history: bool = True,
) -> str:
    """Send a message to a specific agent and get their response.

    Args:
        agent_name: Name of the agent to interact with
        message: Message to send to the agent
        model: Optional temporary model override
        store_history: Whether to store this exchange in history

    Returns:
        The agent's response
    """
    assert ctx.deps.pool, "No agent pool available"
    agent = ctx.deps.pool.get_agent(agent_name)
    result = await agent.run(
        message,
        model=model,
        store_history=store_history,
    )
    return str(result.content)
