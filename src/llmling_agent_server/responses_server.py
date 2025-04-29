"""OpenAI-compatible responses endpoint."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import ConfigDict, Field
from schemez import Schema


if TYPE_CHECKING:
    from collections.abc import Sequence

    from llmling_agent import AgentPool


class InputText(Schema):
    """Text input part."""

    type: Literal["input_text"] = "input_text"
    text: str


class InputImage(Schema):
    """Image input part."""

    type: Literal["input_image"] = "input_image"
    image_url: str


class OutputText(Schema):
    """Text output part."""

    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[dict[str, Any]] = Field(default_factory=list)


class ToolCall(Schema):
    """Tool call in response."""

    type: str  # web_search_call etc
    id: str
    status: Literal["completed", "error"] = "completed"


class Message(Schema):
    """Message in response."""

    type: Literal["message"] = "message"
    id: str
    status: Literal["completed", "error"] = "completed"
    role: Literal["user", "assistant", "system"]
    content: list[OutputText]


class Usage(TypedDict):
    """Token usage information."""

    input_tokens: int
    input_tokens_details: dict[str, int]
    output_tokens: int
    output_tokens_details: dict[str, int]
    total_tokens: int


class ResponseRequest(Schema):
    """Request for /v1/responses endpoint."""

    model: str
    input: str | list[dict[str, Any]]
    instructions: str | None = None
    stream: bool = False
    temperature: float = 1.0
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: str = "auto"
    max_output_tokens: int | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


class Response(Schema):
    """Response from /v1/responses endpoint."""

    id: str = Field(default_factory=lambda: f"resp_{uuid4().hex}")
    object: Literal["response"] = "response"
    created_at: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    status: Literal["completed", "error"] = "completed"
    error: str | None = None
    model: str
    output: Sequence[Message | ToolCall]

    # Include all the request parameters
    instructions: str | None = None
    max_output_tokens: int | None = None
    temperature: float = 1.0
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: str = "auto"
    usage: Usage | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


class ResponsesServer:
    """OpenAI-compatible /v1/responses endpoint."""

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
        """Set up API routes."""
        self.app.post("/v1/responses", dependencies=[Depends(self.verify_api_key)])(
            self.create_response
        )

    async def create_response(self, request: ResponseRequest) -> Response:
        """Handle response creation requests."""
        try:
            agent = self.pool.agents[request.model]
            match request.input:
                case str():
                    content = request.input
                case list():
                    # Get last text content from structured input
                    last = request.input[-1]["content"]
                    text_parts = [p["text"] for p in last if p["type"] == "input_text"]
                    content = "\n".join(text_parts)
                case _:
                    raise HTTPException(400, "Invalid input format")  # noqa: TRY301

            message = await agent.run(content)
            text = OutputText(text=str(message.content))
            output = [Message(id=f"msg_{uuid4().hex}", role="assistant", content=[text])]
            calls = [
                ToolCall(type=f"{tc.tool_name}_call", id=tc.tool_call_id)
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

        except Exception as e:
            raise HTTPException(500, str(e)) from e


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
                "http://localhost:8000/v1/responses",
                headers={"Authorization": "Bearer dummy"},
                json={
                    "model": "gpt-4o",
                    "input": "Tell me a three sentence bedtime story about a unicorn.",
                },
            )
            logger.info("Response: %s", response.text)

            if not response.is_success:
                logger.error("Error: %s", response.text)

    async def main():
        """Run server and test client."""
        pool = AgentPool[None]()
        await pool.add_agent("gpt-4o", model="openai:gpt-4")
        async with pool:
            server = ResponsesServer(pool)
            config = uvicorn.Config(
                server.app, host="0.0.0.0", port=8000, log_level="info"
            )
            server_instance = uvicorn.Server(config)
            await server_instance.serve()

    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=fmt)
    asyncio.run(main())
