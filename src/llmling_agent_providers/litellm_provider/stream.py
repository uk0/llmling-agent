"""LiteLLM Provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from llmling_agent.log import get_logger
from llmling_agent_providers.litellm_provider.utils import Usage


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


logger = get_logger(__name__)


@dataclass
class LiteLLMStream[TResult]:
    """Wrapper to match StreamingResponseProtocol."""

    _stream: Any
    _response: Any | None = None
    model_name: str | None = None
    formatted_content: TResult | None = None
    is_complete: bool = False
    _accumulated_content: str = ""
    _final_usage: Usage | None = None

    async def stream(self) -> AsyncGenerator[TResult, None]:
        """Stream chunks as they arrive."""
        try:
            final_chunk = None
            async for chunk in self._stream:
                if content := chunk.choices[0].delta.content:
                    self._accumulated_content += content
                    # Cast to expected type (usually str)
                    yield self._accumulated_content  # type: ignore
                final_chunk = chunk

            self.is_complete = True
            self.formatted_content = self._accumulated_content  # type: ignore

            # Store usage from final chunk if available
            if final_chunk and hasattr(final_chunk, "usage"):
                self._final_usage = Usage(
                    total_tokens=final_chunk.usage.total_tokens,  # type: ignore
                    request_tokens=final_chunk.usage.prompt_tokens,  # type: ignore
                    response_tokens=final_chunk.usage.completion_tokens,  # type: ignore
                )

        except Exception as e:
            logger.exception("Error during streaming")
            self.is_complete = True
            msg = "Streaming failed"
            raise RuntimeError(msg) from e

    async def stream_text(self, delta: bool = False) -> AsyncGenerator[str, None]:
        """Stream chunks as they arrive."""
        try:
            final_chunk = None
            async for chunk in self._stream:
                if content := chunk.choices[0].delta.content:
                    self._accumulated_content += content
                    if delta:
                        yield str(content)
                    else:
                        yield self._accumulated_content
                final_chunk = chunk

            self.is_complete = True
            self.formatted_content = self._accumulated_content  # type: ignore

            # Store usage from final chunk if available
            if final_chunk and hasattr(final_chunk, "usage"):
                self._final_usage = Usage(
                    total_tokens=final_chunk.usage.total_tokens,  # type: ignore
                    request_tokens=final_chunk.usage.prompt_tokens,  # type: ignore
                    response_tokens=final_chunk.usage.completion_tokens,  # type: ignore
                )

        except Exception as e:
            logger.exception("Error during streaming")
            self.is_complete = True
            msg = "Streaming failed"
            raise RuntimeError(msg) from e

    def usage(self) -> Usage:
        """Get token usage statistics."""
        if not self._final_usage:
            return Usage(total_tokens=0, request_tokens=0, response_tokens=0)
        return self._final_usage
