"""OpenAI-compatible responses endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from fastapi import HTTPException

from llmling_agent_server.responses_models import (
    Response,
    ResponseMessage,
    ResponseOutputText,
    ResponseRequest,
    ResponseToolCall,
)


if TYPE_CHECKING:
    from llmling_agent.agent import AnyAgent


async def handle_request(request: ResponseRequest, agent: AnyAgent[Any, Any]):
    match request.input:
        case str():
            content = request.input
        case list():
            # Get last text content from structured input
            last = request.input[-1]["content"]
            text_parts = [p["text"] for p in last if p["type"] == "input_text"]
            content = "\n".join(text_parts)
        case _:
            raise HTTPException(400, "Invalid input format")

    message = await agent.run(content)
    text = ResponseOutputText(text=str(message.content))
    output = [ResponseMessage(id=f"msg_{uuid4().hex}", role="assistant", content=[text])]
    calls = [
        ResponseToolCall(type=f"{tc.tool_name}_call", id=tc.tool_call_id)
        for tc in message.tool_calls
    ]

    return Response(
        model=request.model,
        output=calls + output,
        instructions=request.instructions,
        max_output_tokens=request.max_output_tokens,
        temperature=request.temperature,
        tools=request.tools,
        tool_choice=request.tool_choice,
        usage=message.cost_info.token_usage if message.cost_info else None,  # pyright: ignore
        metadata=request.metadata,
    )
