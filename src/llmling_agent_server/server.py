"""OpenAI-compatible API server for LLMling agents."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from llmling_agent import AgentPool, AnyAgent

logger = get_logger(__name__)


class OpenAIModelInfo(BaseModel):
    """OpenAI model info format."""

    id: str
    object: str = "model"
    owned_by: str = "llmling"
    created: int
    description: str | None = None
    permissions: list[str] = []


class FunctionCall(BaseModel):
    """Function call information."""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """Tool call information."""

    id: str
    type: str = "function"
    function: FunctionCall


class OpenAIMessage(BaseModel):
    """OpenAI chat message format."""

    role: Literal["system", "user", "assistant", "tool", "function"]
    content: str | None  # Content can be null in function calls
    name: str | None = None
    function_call: FunctionCall | None = None
    tool_calls: list[ToolCall] | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request."""

    model: str
    messages: list[OpenAIMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | None = Field(default="auto")


class Choice(BaseModel):
    """Choice in a completion response."""

    index: int = 0
    message: OpenAIMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    """OpenAI chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: dict[str, int] | None = None


class ChatCompletionChunk(BaseModel):
    """Chunk of a streaming chat completion."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[dict[str, Any]]


async def stream_response(
    agent: AnyAgent[Any, Any], content: str, request: ChatCompletionRequest
) -> AsyncGenerator[str, None]:
    """Generate streaming response chunks."""
    response_id = f"chatcmpl-{int(time.time() * 1000)}"
    created = int(time.time())

    try:
        # First chunk with role
        choice = {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
        first_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [choice],
        }
        yield f"data: {json.dumps(first_chunk)}\n\n"
        async with agent.run_stream(content) as stream:
            async for chunk in stream.stream():
                # Skip empty chunks
                if not chunk:
                    continue
                choice = {
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": None,
                }
                chunk_data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [choice],
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
        final_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.exception("Error during streaming response")
        error_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": f"Error: {e!s}"},
                    "finish_reason": "error",
                }
            ],
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"


class OpenAIServer:
    """OpenAI-compatible API server backed by LLMling agents."""

    def __init__(self, pool: AgentPool):
        self.pool = pool
        self.app = FastAPI()
        self.setup_routes()

    def verify_api_key(
        self, authorization: Annotated[str | None, Header(alias="Authorization")] = None
    ):
        """Verify API key if configured."""
        if not authorization:
            raise HTTPException(401, "Missing API key")
        if not authorization.startswith("Bearer "):
            raise HTTPException(401, "Invalid authorization format")

    def setup_routes(self):
        """Configure API routes."""
        self.app.get("/v1/models")(self.list_models)
        self.app.post(
            "/v1/chat/completions",
            dependencies=[Depends(self.verify_api_key)],
            response_model=None,  # This is the key change
        )(self.create_chat_completion)

    async def list_models(self) -> dict[str, Any]:
        """List available agents as models."""
        models = []
        for name, agent in self.pool.agents.items():
            info = OpenAIModelInfo(id=name, created=0, description=agent.description)
            models.append(info)
        return {"object": "list", "data": models}

    async def create_chat_completion(self, request: ChatCompletionRequest) -> Response:
        """Handle chat completion requests."""
        # Get agent by model name
        try:
            agent = self.pool.agents[request.model]
        except KeyError:
            raise HTTPException(404, f"Model {request.model} not found") from None

        # Just take the last message content - let agent handle history
        content = request.messages[-1].content or ""

        # Check if streaming is requested
        if request.stream:
            return StreamingResponse(
                stream_response(agent, content, request),
                media_type="text/event-stream",
            )
        # Non-streaming response
        try:
            response = await agent.run(content)
            message = OpenAIMessage(role="assistant", content=str(response.content))
            completion_response = ChatCompletionResponse(
                id=response.message_id,
                created=int(response.timestamp.timestamp()),
                model=request.model,
                choices=[Choice(message=message)],
                usage=response.cost_info.token_usage if response.cost_info else None,  # pyright: ignore
            )
            return Response(
                content=completion_response.model_dump_json(),
                media_type="application/json",
            )
        except Exception as e:
            logger.exception("Error processing chat completion")
            raise HTTPException(500, f"Error: {e!s}") from e


if __name__ == "__main__":
    import asyncio
    import logging

    import httpx
    import uvicorn

    from llmling_agent import AgentPool

    async def test_client():
        """Test the API with a direct HTTP request."""
        logger = logging.getLogger(__name__)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/v1/chat/completions",
                headers={"Authorization": "Bearer dummy"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "Tell me a joke"}],
                    "stream": True,
                },
                timeout=30.0,  # Longer timeout for streaming
            )

            if response.is_success:
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix
                        if data == "[DONE]":
                            break
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"]
                        if "content" in delta:
                            print(delta["content"], end="", flush=True)
                print("\n")
            else:
                logger.error("Response error: %s", response.text)

    async def main():
        """Run server and test client."""
        pool = AgentPool[None]()
        await pool.add_agent("gpt-4o-mini", model="openai:gpt-4o-mini")
        async with pool:  # Ensure pool is properly initialized
            server = OpenAIServer(pool)
            config = uvicorn.Config(
                server.app, host="0.0.0.0", port=8000, log_level="info"
            )
            server_instance = uvicorn.Server(config)
            await server_instance.serve()

    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=fmt)
    asyncio.run(main())
