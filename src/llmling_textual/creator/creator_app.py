from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import ValidationError
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Header, Input, Static
from upath import UPath
from upathtools import read_path
from yaml import YAMLError

from llmling_agent import Agent, AgentsManifest
from llmling_agent.utils.count_tokens import count_tokens
from llmling_agent_cli import agent_store


EXAMPLE = """
# Example agent with team
agents:
  analyzer:
    name: "Analyzer"
    model: "gpt-4"
    capabilities:
      can_load_resources: true

teams:
  analysis_team:
    mode: "sequential"
    members: ["analyzer"]
    connections:
      - target: "output_handler"
        type: "forward"
"""

SYS_PROMPT = """
You are an expert at creating LLMling-agent configurations.
Generate complete, valid YAML that can include:
- Agent configurations with appropriate tools and capabilities
- Team definitions with proper member relationships
- Connection setups for message routing
Follow the provided JSON schema exactly.
Only add stuff asked for by the user.
ONLY RETURN THE ACTUAL YAML. Your Output should ALWAYS be parseable by a YAML parser.
Nver answer with anything else. Dont prepend any sentences. Just return plain YAML.
"""

SCHEMA_URL = "https://raw.githubusercontent.com/phil65/llmling-agent/refs/heads/main/schema/config-schema.json"
README_URL = (
    "https://raw.githubusercontent.com/phil65/llmling-agent/refs/heads/main/README.md"
)


class StatsDisplay(Static):
    """Display for token count and validation status."""

    def update_stats(self, token_count: int, status: str | None = None):
        """Update the stats display."""
        text = f"Context tokens: {token_count:,}"
        if status:
            text = f"{status} | {text}"
        self.update(text)


class YamlDisplay(Static):
    """Display for YAML content with syntax highlighting."""

    def update_yaml(self, content: str):
        """Update the YAML content with syntax highlighting."""
        from rich.syntax import Syntax

        self.update(Syntax(content, "yaml", theme="monokai"))


class ConfigGeneratorApp(App):
    """Application for generating configuration files."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 1;
        padding: 1;
    }

    Input {
        dock: top;
        margin: 1 0;
    }

    YamlDisplay {
        height: 1fr;
        border: solid green;
    }

    StatsDisplay {
        dock: bottom;
        height: 3;
        content-align: center middle;
    }
    """

    BINDINGS: ClassVar = [
        Binding("ctrl+s", "save", "Save Config", show=True),
        ("escape", "quit", "Quit"),
    ]

    def __init__(
        self,
        model: str = "openai:gpt-4o-mini",
        provider: Literal["pydantic_ai", "litellm"] = "pydantic_ai",
        output_path: str | None = None,
        add_to_store: bool = False,
    ):
        super().__init__()
        self.agent = self.setup_agent(model, provider)
        self.current_config: str | None = None
        self.output_path = UPath(output_path) if output_path else None
        self.add_to_store = add_to_store
        self._token_count: int = 0

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="Describe your configuration needs...")
        yield YamlDisplay("")
        yield StatsDisplay("Context tokens: calculating...")

    def setup_agent(
        self, model: str, provider: Literal["pydantic_ai", "litellm"]
    ) -> Agent:
        return Agent(
            "config_generator",
            model=model,
            provider=provider,
            system_prompt=SYS_PROMPT,
        )

    async def on_mount(self):
        """Load schema and calculate token count."""
        schema = await read_path(SCHEMA_URL)
        readme = await read_path(README_URL)

        context = f"Schema:\n{schema}\n\nExample:\n{EXAMPLE}\n\\Readme:\n{readme}"
        self.agent.conversation.add_context_message(context)

        # Calculate token count
        model_name = self.agent.model_name or "gpt-4o-mini"
        model_name = model_name.split(":")[-1]
        self._token_count = count_tokens(context, model_name)

        stats = self.query_one(StatsDisplay)
        stats.update_stats(self._token_count)

    async def on_input_submitted(self, message: Input.Submitted):
        """Generate config when user hits enter."""
        yaml = await self.agent.run(message.value)
        self.current_config = yaml.content

        # Validate
        try:
            AgentsManifest.from_yaml(yaml.content)
            status = "✓ Valid configuration"
        except (ValidationError, YAMLError) as e:
            status = f"✗ Invalid: {e}"

        # Update displays
        content = self.query_one(YamlDisplay)
        content.update_yaml(yaml.content)

        stats = self.query_one(StatsDisplay)
        stats.update_stats(self._token_count, status)

    def action_save(self):
        """Save current config."""
        if not self.current_config:
            self.notify("No configuration generated yet!")
            return

        if not self.output_path:
            self.notify("No output path specified!")
            return

        # Save file
        self.output_path.write_text(self.current_config)

        # Optionally add to store
        if self.add_to_store:
            agent_store.add_config(
                self.output_path.stem,
                str(self.output_path),
            )

        self.notify(f"Saved to {self.output_path}")


if __name__ == "__main__":
    app = ConfigGeneratorApp()
    app.run()
