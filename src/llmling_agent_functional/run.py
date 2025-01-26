"""Functional wrappers for Agent usage."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Unpack, overload

from typing_extensions import TypeVar

from llmling_agent import Agent


if TYPE_CHECKING:
    import os

    import PIL.Image
    from toprompt import AnyPromptType

    from llmling_agent.agent.agent import AgentKwargs
    from llmling_agent.models.messages import ChatMessage


TResult = TypeVar("TResult", default=str)


@overload
async def run_agent[TResult](
    prompt: AnyPromptType | PIL.Image.Image | os.PathLike[str],
    *,
    result_type: type[TResult],
    **kwargs: Unpack[AgentKwargs],
) -> ChatMessage[TResult]: ...


@overload
async def run_agent(
    prompt: AnyPromptType | PIL.Image.Image | os.PathLike[str],
    **kwargs: Unpack[AgentKwargs],
) -> ChatMessage[str]: ...


async def run_agent(
    prompt: AnyPromptType | PIL.Image.Image | os.PathLike[str],
    *,
    result_type: type[Any] | None = None,
    **kwargs: Unpack[AgentKwargs],
) -> ChatMessage[Any]:
    """Run prompt through agent and return result."""
    agent = Agent[None](**kwargs)
    async with agent:
        return await agent.run(prompt, result_type=result_type)


@overload
def run_agent_sync[TResult](
    prompt: AnyPromptType | PIL.Image.Image | os.PathLike[str],
    *,
    result_type: type[TResult],
    **kwargs: Unpack[AgentKwargs],
) -> ChatMessage[TResult]: ...


@overload
def run_agent_sync(
    prompt: AnyPromptType | PIL.Image.Image | os.PathLike[str],
    **kwargs: Unpack[AgentKwargs],
) -> ChatMessage[str]: ...


def run_agent_sync(
    prompt: AnyPromptType | PIL.Image.Image | os.PathLike[str],
    *,
    result_type: type[Any] | None = None,
    **kwargs: Unpack[AgentKwargs],
) -> ChatMessage[Any]:
    """Sync wrapper for run_agent."""
    return asyncio.run(
        run_agent(
            prompt,
            result_type=result_type,  # type: ignore
            **kwargs,
        )
    )
