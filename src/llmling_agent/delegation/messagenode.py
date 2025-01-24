from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    from llmling_agent import AnyAgent, Team
    from llmling_agent.models.messages import ChatMessage
    from llmling_agent.talk import Talk, TeamTalk


@runtime_checkable
class MessageNode[TDeps, TResult](Protocol):
    """Component that can be part of a TeamRun chain."""

    name: str

    async def run(self, *prompts: Any, **kwargs: Any) -> ChatMessage[TResult]: ...

    def pass_results_to(
        self,
        other: AnyAgent[Any, Any] | Team[Any] | MessageNode[Any, Any],
        **kwargs: Any,
    ) -> Talk[TResult] | TeamTalk: ...
