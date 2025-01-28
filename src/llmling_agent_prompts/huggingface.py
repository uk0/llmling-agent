from __future__ import annotations

import os
from typing import TYPE_CHECKING

from huggingface_hub import HfApi
from upath import UPath
import yamling

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from llmling_agent.models.prompt_hubs import HuggingFaceConfig
    from llmling_agent.prompts.models import PromptTemplate


logger = get_logger(__name__)


class HuggingFaceProvider:
    """Provider for prompts stored in HuggingFace repositories."""

    name = "huggingface"
    supports_versions = True

    def __init__(self, config: HuggingFaceConfig):
        """Initialize HuggingFace provider.

        Args:
            config: Provider configuration
        """
        self._config = config
        self._api = HfApi(token=config.api_key or os.getenv("HF_TOKEN"))
        if config.base_url:
            self._api.endpoint = config.base_url

    async def get_prompt(
        self, prompt_id: str, version: str | None = None
    ) -> PromptTemplate:
        """Get prompt from HuggingFace Hub.

        Args:
            prompt_id: Format "user/repo/path.yml"
            version: Optional git revision/tag

        Returns:
            Loaded prompt template

        Raises:
            ValueError: If prompt_id format is invalid
            RuntimeError: If prompt loading fails
        """
        if "/" not in prompt_id:
            msg = "Invalid prompt_id format. Expected: user/repo/path"
            raise ValueError(msg)

        *repo_parts, path = prompt_id.split("/")
        repo_id = "/".join(repo_parts)

        try:
            path_obj = UPath(
                self._api.hf_hub_download(
                    repo_id=repo_id,
                    filename=path,
                    revision=version,
                    token=self._api.token,
                )
            )

            # Load YAML using yamling
            data = yamling.load_yaml_file(path_obj)

            # Import here to avoid circular import
            from llmling_agent.prompts.models import PromptTemplate

            return PromptTemplate.model_validate(data)

        except Exception as e:
            msg = f"Failed to load prompt {prompt_id}: {e}"
            raise RuntimeError(msg) from e

    async def list_prompts(self) -> list[str]:
        """List available prompts in configured workspace.

        Returns:
            List of prompt IDs in workspace
        """
        if not self._config.workspace:
            return []

        try:
            # Get all YAML files in workspace using correct API method
            files = self._api.list_repo_files(
                repo_id=self._config.workspace,
                token=self._api.token,
            )
            return [
                f"{self._config.workspace}/{f}"
                for f in files
                if f.endswith((".yml", ".yaml"))
            ]

        except Exception:
            logger.exception("Failed to list prompts")
            return []

    def __repr__(self) -> str:
        """Create readable representation."""
        workspace = self._config.workspace or "no workspace"
        return f"HuggingFaceProvider({workspace})"
