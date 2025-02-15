"""Jinja2 environment configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from llmling import ConfigModel
from pydantic import BaseModel, Field

from llmling_agent.models.tools import ToolConfig  # noqa: TC001


if TYPE_CHECKING:
    from jinjarope import Environment, Template

UndefinedBehaviour = Literal["default", "strict", "debug", "chainable"]


class Jinja2EnvironmentConfig(ConfigModel):
    """Configuration for Jinja2 environment.

    See: https://jinja.palletsprojects.com/en/3.1.x/api/#jinja2.Environment
    """

    block_start_string: str = "{%"
    """String denoting the beginning of a block (default: '{%')."""

    block_end_string: str = "%}"
    """String denoting the end of a block (default: '%}')."""

    variable_start_string: str = "{{"
    """String denoting the beginning of a variable (default: '{{')."""

    variable_end_string: str = "}}"
    """String denoting the end of a variable (default: '}}')."""

    comment_start_string: str = "{#"
    """String denoting the beginning of a comment (default: '{#')."""

    comment_end_string: str = "#}"
    """String denoting the end of a comment (default: '#}')."""

    line_statement_prefix: str | None = None
    """Prefix that begins a line-based statement (e.g., '#' for line statements)."""

    line_comment_prefix: str | None = None
    """Prefix that begins a line-based comment."""

    trim_blocks: bool = False
    """Remove first newline after a block (affects whitespace control)."""

    lstrip_blocks: bool = False
    """Remove leading spaces and tabs from the start of a line to a block."""

    newline_sequence: Literal["\n", "\r\n", "\r"] = "\n"
    """Sequence that starts a newline (default: '\n')."""

    keep_trailing_newline: bool = False
    """Preserve the trailing newline when rendering templates."""

    extensions: list[str] = Field(default_factory=list)
    """List of Jinja2 extensions to load (e.g., 'jinja2.ext.do')."""

    undefined: UndefinedBehaviour = "default"
    """Behavior when accessing undefined variables (default, strict, debug, chainable)."""

    filters: list[ToolConfig] = Field(default_factory=list)
    """Custom filters as list of tool configurations."""

    tests: list[ToolConfig] = Field(default_factory=list)
    """Custom tests as list of tool configurations."""

    globals: dict[str, BaseModel] = Field(default_factory=dict)
    """Global variables available to all templates."""

    def create_environment(self, *, enable_async: bool = False) -> Environment:
        """Create configured Jinja2 environment.

        Args:
            enable_async: Whether to enable async features
        """
        from jinjarope import Environment

        return Environment(enable_async=enable_async, **self.create_environment_kwargs())

    def create_environment_kwargs(self) -> dict[str, Any]:
        """Convert config to Jinja2 environment kwargs.

        Creates a dictionary of kwargs for jinja2.Environment with proper
        conversion of special values.

        Returns:
            Dict of kwargs for jinja2.Environment constructor

        Raises:
            ValueError: If filter or test imports fail
        """
        from jinja2 import (
            ChainableUndefined,
            DebugUndefined,
            StrictUndefined,
            Undefined,
        )

        # Basic config
        kwargs = self.model_dump(exclude={"undefined", "filters", "tests"})

        # Handle undefined setting
        kwargs["undefined"] = {
            "default": Undefined,
            "strict": StrictUndefined,
            "debug": DebugUndefined,
            "chainable": ChainableUndefined,
        }[self.undefined]

        try:
            # Convert filters - use tool name as filter name
            filters = {}
            for tool_config in self.filters:
                tool = tool_config.get_tool()
                filters[tool.name] = tool.callable.callable
            kwargs["filters"] = filters

            # Convert tests - same pattern
            tests = {}
            for tool_config in self.tests:
                tool = tool_config.get_tool()
                tests[tool.name] = tool.callable.callable
            kwargs["tests"] = tests

        except Exception as exc:
            msg = f"Failed to import Jinja2 filters/tests: {exc}"
            raise ValueError(msg) from exc

        return kwargs


class Jinja2Template(ConfigModel):
    """Template with environment configuration."""

    template: str
    """The template string to render."""

    environment: Jinja2EnvironmentConfig = Field(default_factory=Jinja2EnvironmentConfig)
    """Environment configuration for this template."""

    def get_template(self, *, enable_async: bool = False) -> Template:
        """Get compiled Jinja2 template.

        Args:
            enable_async: Whether to enable async features

        Returns:
            Compiled template ready for rendering
        """
        env = self.environment.create_environment(enable_async=enable_async)
        return env.from_string(self.template)

    def render(self, **kwargs: Any) -> str:
        """Render template with provided variables.

        Args:
            **kwargs: Template variables

        Returns:
            Rendered template string
        """
        template = self.get_template()
        return template.render(**kwargs)

    async def render_async(self, **kwargs: Any) -> str:
        """Render template asynchronously.

        Args:
            **kwargs: Template variables

        Returns:
            Rendered template string
        """
        template = self.get_template(enable_async=True)
        return await template.render_async(**kwargs)
