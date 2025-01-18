from datetime import datetime
from os import PathLike
from typing import Any, ClassVar

from jinja2 import Template
from upath import UPath

from llmling_agent.common_types import JsonValue
from llmling_agent.log import get_logger
from llmling_agent.models.agents import ToolCallInfo
from llmling_agent.models.storage import LogFormat, TextLogConfig
from llmling_agent_storage.base import StorageProvider


CONVERSATIONS_TEMPLATE = """\
=== LLMling Agent Log ===

{%- for conv_id, conv in conversations.items() %}
=== Conversation {{ conv_id }} (agent: {{ conv.agent_name }}, started: {{ conv.start_time.strftime('%Y-%m-%d %H:%M:%S') }}) ===

{%- for msg in messages if msg.conversation_id == conv_id %}
[{{ msg.timestamp.strftime('%Y-%m-%d %H:%M:%S') }}] {{ msg.sender }}{% if msg.model %} ({{ msg.model }}){% endif %}: {{ msg.content }}
{%- if msg.cost_info %}
Tokens: {{ msg.cost_info.token_usage.total }} (prompt: {{ msg.cost_info.token_usage.prompt }}, completion: {{ msg.cost_info.token_usage.completion }})
Cost: ${{ "%.4f"|format(msg.cost_info.total_cost) }}
{%- endif %}
{%- if msg.response_time %}
Response time: {{ "%.1f"|format(msg.response_time) }}s
{%- endif %}
{%- if msg.forwarded_from %}
Forwarded via: {{ msg.forwarded_from|join(' -> ') }}
{%- endif %}

{%- for tool in tool_calls if tool.message_id == msg.id %}
Tool Call: {{ tool.tool_name }}
Args: {{ tool.args|pprint }}
Result: {{ tool.result }}
{%- endfor %}
{%- endfor %}
{%- endfor %}

=== Commands ===
{%- for cmd in commands %}
[{{ cmd.timestamp.strftime('%Y-%m-%d %H:%M:%S') }}] {{ cmd.agent_name }} ({{ cmd.session_id }}): {{ cmd.command }}
{%- endfor %}
"""  # noqa: E501

CHRONOLOGICAL_TEMPLATE = """\
=== LLMling Agent Log ===

{%- for entry in entries|sort(attribute="timestamp") %}
{%- if entry.type == "conversation_start" %}
=== Conversation {{ entry.conversation_id }} (agent: {{ entry.agent_name }}) started ===

{%- elif entry.type == "message" %}
[{{ entry.timestamp.strftime('%Y-%m-%d %H:%M:%S') }}] {{ entry.sender }}{% if entry.model %} ({{ entry.model }}){% endif %}: {{ entry.content }}
{%- if entry.cost_info %}
Tokens: {{ entry.cost_info.token_usage.total }} (prompt: {{ entry.cost_info.token_usage.prompt }}, completion: {{ entry.cost_info.token_usage.completion }})
Cost: ${{ "%.4f"|format(entry.cost_info.total_cost) }}
{%- endif %}
{%- if entry.response_time %}
Response time: {{ "%.1f"|format(entry.response_time) }}s
{%- endif %}
{%- if entry.forwarded_from %}
Forwarded via: {{ entry.forwarded_from|join(' -> ') }}
{%- endif %}

{%- elif entry.type == "tool_call" %}
[{{ entry.timestamp.strftime('%Y-%m-%d %H:%M:%S') }}] Tool Call: {{ entry.tool_name }}
Args: {{ entry.args|pprint }}
Result: {{ entry.result }}

{%- elif entry.type == "command" %}
[{{ entry.timestamp.strftime('%Y-%m-%d %H:%M:%S') }}] Command by {{ entry.agent_name }}: {{ entry.command }}

{%- endif %}
{%- endfor %}
"""  # noqa: E501


logger = get_logger(__name__)


class TextLogProvider(StorageProvider):
    """Human-readable text log provider."""

    TEMPLATES: ClassVar[dict[LogFormat, str]] = {
        "chronological": CHRONOLOGICAL_TEMPLATE,
        "conversations": CONVERSATIONS_TEMPLATE,
    }
    can_load_history = False  # Text logs are write-only

    def __init__(self, config: TextLogConfig):
        """Initialize text log provider.

        Args:
            config: Configuration for provider
            kwargs: Additional arguments to pass to StorageProvider
        """
        super().__init__(config)
        self.path = UPath(config.path)
        self.encoding = config.encoding
        self.template = self._load_template(config.template)
        self._entries: list[dict[str, Any]] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write()  # Create initial empty file

    def _load_template(
        self,
        template: LogFormat | str | PathLike[str] | None,
    ) -> Template:
        """Load template from predefined or file."""
        if template is None:
            template_str = self.TEMPLATES["chronological"]
        elif template in self.TEMPLATES:
            template_str = self.TEMPLATES[template]  # type: ignore
        else:
            # Assume it's a path
            with UPath(template).open() as f:
                template_str = f.read()
        return Template(template_str)

    async def log_message(self, **kwargs):
        """Store message and update log."""
        self._entries.append({
            "type": "message",
            "timestamp": datetime.now(),
            **kwargs,
        })
        self._write()

    async def log_conversation(
        self,
        *,
        conversation_id: str,
        agent_name: str,
        start_time: datetime | None = None,
    ):
        """Store conversation start and update log."""
        self._entries.append({
            "type": "conversation_start",
            "timestamp": start_time or datetime.now(),
            "conversation_id": conversation_id,
            "agent_name": agent_name,
        })
        self._write()

    async def log_tool_call(
        self,
        *,
        conversation_id: str,
        message_id: str,
        tool_call: ToolCallInfo,
    ):
        """Store tool call and update log."""
        self._entries.append({
            "type": "tool_call",
            "timestamp": tool_call.timestamp,
            "conversation_id": conversation_id,
            "message_id": message_id,
            "tool_name": tool_call.tool_name,
            "args": tool_call.args,
            "result": tool_call.result,
        })
        self._write()

    async def log_command(
        self,
        *,
        agent_name: str,
        session_id: str,
        command: str,
        context_type: type | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ):
        """Store command and update log."""
        self._entries.append({
            "type": "command",
            "timestamp": datetime.now(),
            "agent_name": agent_name,
            "session_id": session_id,
            "command": command,
            "context_type": context_type.__name__ if context_type else None,
            "metadata": metadata,
        })
        self._write()

    def _write(self):
        """Write current state to file."""
        try:
            context = {"entries": self._entries}
            text = self.template.render(context)
            self.path.write_text(text, encoding=self.encoding)
        except Exception as e:
            logger.exception("Failed to write to log file: %s", self.path)
            msg = f"Failed to write to log file: {e}"
            raise RuntimeError(msg) from e

    async def get_commands(
        self,
        agent_name: str,
        session_id: str,
        *,
        limit: int | None = None,
        current_session_only: bool = False,
    ) -> list[str]:
        """Not supported for text logs."""
        msg = f"{self.__class__.__name__} does not support retrieving commands"
        raise NotImplementedError(msg)
