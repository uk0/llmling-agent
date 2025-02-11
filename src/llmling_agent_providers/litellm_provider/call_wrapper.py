from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel


if TYPE_CHECKING:
    from litellm import CustomStreamWrapper
    from litellm.files.main import ModelResponse

    from llmling_agent.tools.base import ToolInfo
    from llmling_agent_providers.base import UsageLimits


class FakeAgent:
    def __init__(self, model: str, model_settings: dict[str, Any] | None = None):
        self.model = model
        self.model_settings = model_settings or {}
        self.extra_headers = None
        self.base_url: str | None = None
        if self.model.startswith("copilot:"):
            self.model = self.model.removeprefix("copilot:")
            token = os.getenv("GITHUB_COPILOT_API_KEY")
            self.extra_headers = {
                "Authorization": f"Bearer {token}",
                "editor-version": "Neovim/0.9.0",
                "Copilot-Integration-Id": "vscode-chat",
            }
            self.base_url = "https://api.githubcopilot.com"

    async def run(
        self,
        messages: list[dict[str, Any]],
        result_type: type[BaseModel] | None = None,
        usage_limits: UsageLimits | None = None,
        num_retries: int | None = None,
        tools: list[ToolInfo] | None = None,
    ) -> ModelResponse:
        from litellm import acompletion

        schemas = [t.schema for t in tools or []]
        if self.base_url:
            self.model_settings["base_url"] = self.base_url
        if self.extra_headers:
            self.model_settings["extra_headers"] = self.extra_headers
        return await acompletion(  # type: ignore
            stream=False,
            model=self.model,
            messages=messages,
            max_tokens=usage_limits.response_tokens_limit if usage_limits else None,
            response_format=result_type
            if result_type and issubclass(result_type, BaseModel)
            else None,
            num_retries=num_retries,
            tools=schemas or None,
            tool_choice="auto" if schemas else None,
            **self.model_settings,
        )

    async def run_stream(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        result_type: type[BaseModel] | None = None,
        usage_limits: UsageLimits | None = None,
        num_retries: int | None = None,
        tools: list[ToolInfo] | None = None,
    ) -> CustomStreamWrapper:
        from litellm import acompletion

        schemas = [t.schema for t in tools or []]
        if self.base_url:
            self.model_settings["base_url"] = self.base_url
        if self.extra_headers:
            self.model_settings["extra_headers"] = self.extra_headers
        return await acompletion(  # type: ignore
            stream=True,
            model=self.model,
            messages=messages,
            max_tokens=usage_limits.response_tokens_limit if usage_limits else None,
            response_format=result_type
            if result_type and issubclass(result_type, BaseModel)
            else None,
            num_retries=num_retries,
            tools=schemas or None,
            tool_choice="auto" if schemas else None,
            **self.model_settings,
        )


if __name__ == "__main__":

    async def main():
        agent = FakeAgent("copilot:gpt-4o-mini")
        return await agent.run([])

    import asyncio

    result = asyncio.run(main())
    print(result)
