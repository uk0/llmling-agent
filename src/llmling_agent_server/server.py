"""OpenAI-compatible API server for LLMling agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from llmling_agent.delegation.pool import AgentPool

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


class OpenAIServer:
    """OpenAI-compatible API server backed by LLMling agents."""

    def __init__(self, pool: AgentPool):
        self.pool = pool
        self.app = FastAPI()  # (lifespan=self.lifespan)
        self.setup_routes()

    def verify_api_key(
        self, authorization: Annotated[str | None, Header(alias="Authorization")] = None
    ) -> None:
        """Verify API key if configured."""
        if not authorization:
            raise HTTPException(401, "Missing API key")
        if not authorization.startswith("Bearer "):
            raise HTTPException(401, "Invalid authorization format")

    # @asynccontextmanager
    # async def lifespan(self, app: FastAPI):
    #     """Initialize agent pool."""
    #     async with self.pool:
    #         yield

    def setup_routes(self):
        """Configure API routes."""
        self.app.get("/v1/models")(self.list_models)
        self.app.post(
            "/v1/chat/completions", dependencies=[Depends(self.verify_api_key)]
        )(self.create_chat_completion)

    async def list_models(self) -> dict[str, Any]:
        """List available agents as models."""
        models = []
        for name, agent in self.pool.agents.items():
            models.append(
                OpenAIModelInfo(
                    id=name,
                    created=0,
                    description=agent.description,
                )
            )
        return {"object": "list", "data": models}

    async def create_chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Handle chat completion requests."""
        # Get agent by model name
        agent = self.pool.agents[request.model]

        # Just take the last message content - let agent handle history
        content = request.messages[-1].content or ""

        # Run agent with the content
        response = await agent.run(content)

        # Convert response to OpenAI format
        return ChatCompletionResponse(
            id=response.message_id,
            created=int(response.timestamp.timestamp()),
            model=request.model,
            choices=[
                Choice(
                    message=OpenAIMessage(role="assistant", content=str(response.content))
                )
            ],
            usage=response.cost_info.token_usage if response.cost_info else None,
        )


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
                headers={"Authorization": "Bearer dummy"},  # This matches OpenAI's format
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "Tell me a joke"}],
                },
            )
            logger.debug("Raw response: %s", response.text)
            logger.debug("Status code: %s", response.status_code)
            if not response.is_success:
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
