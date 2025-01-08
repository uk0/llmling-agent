"""Resource management commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from slashed import Command, CommandContext, CommandError
from slashed.completers import CallbackCompleter

from llmling_agent.log import get_logger
from llmling_agent_commands.completers import get_resource_names


if TYPE_CHECKING:
    from llmling_agent.chat_session.base import AgentPoolView


logger = get_logger(__name__)

LIST_RESOURCES_HELP = """\
Display all resources available to the agent.

Shows:
- Resource names and descriptions
- Resource types and URIs
- Whether parameters are supported
- MIME types

Resource types can be:
- path: Files or URLs
- text: Raw text content
- cli: Command line tools
- source: Python source code
- callable: Python functions
- image: Image files

Use /show-resource for detailed information about specific resources.
"""

SHOW_RESOURCES_HELP = """\
Display detailed information and content of a specific resource.

Shows:
- Resource metadata (type, URI, description)
- MIME type information
- Parameter support status
- Resource content (if loadable)

For resources that support parameters:
- Pass parameters as --param arguments
- Parameters are passed to resource loader\

Examples:
  /show-resource config.yml               # Show configuration file
  /show-resource template --date today    # Template with parameters
  /show-resource image.png               # Show image details
  /show-resource api --key value         # API with parameters

Note: Some resources might require parameters to be viewed.
"""

ADD_RESOURCE_HELP = """\
Add content from a resource to the next message.

Parameters are passed to the resource loader if supported.

Examples:
/add-resource config.yml
/add-resource template --date today
/add-resource api_data --key value"""


async def list_resources(
    ctx: CommandContext[AgentPoolView],
    args: list[str],
    kwargs: dict[str, str],
):
    """List available resources."""
    try:
        resources = ctx.context._agent.runtime.get_resources()

        sections = ["# Available Resources\n"]
        for resource in resources:
            desc = f": {resource.description}" if resource.description else ""
            sections.append(f"- **{resource.name}**{desc}")
            sections.append(f"  Type: {resource.type}")
            if resource.uri:
                sections.append(f"  URI: `{resource.uri}`")

            # Show if resource is templated
            if resource.is_templated():
                sections.append("  *Supports parameters*")

        await ctx.output.print("\n".join(sections))
    except Exception as e:  # noqa: BLE001
        await ctx.output.print(f"Failed to list resources: {e}")


async def show_resource(
    ctx: CommandContext[AgentPoolView],
    args: list[str],
    kwargs: dict[str, str],
):
    """Show details or content of a resource."""
    if not args:
        msg = "Usage: /show-resource <name> [--param1 value1] [--param2 value2]"
        await ctx.output.print(msg)
        return

    name = args[0]
    try:
        # First get resource info
        resources = ctx.context._agent.runtime.get_resources()
        resource_info = next((r for r in resources if r.name == name), None)
        if not resource_info:
            await ctx.output.print(f"Resource '{name}' not found")
            return
        sections = [f"# Resource: {name}\n", f"Type: {resource_info.type}"]
        if resource_info.uri:
            sections.append(f"URI: `{resource_info.uri}`")
        if resource_info.description:
            sections.append(f"Description: {resource_info.description}")
        if resource_info.is_templated():
            sections.append("\nParameters supported")
        sections.append(f"MIME Type: {resource_info.mime_type}")

        # Try to load content with provided parameters
        try:
            content = await ctx.context._agent.runtime.load_resource(name, **kwargs)
            sections.extend(["\n# Content:", "```", str(content), "```"])
        except Exception as e:  # noqa: BLE001
            sections.append(f"\nFailed to load content: {e}")

        await ctx.output.print("\n".join(sections))
    except Exception as e:  # noqa: BLE001
        await ctx.output.print(f"Error accessing resource: {e}")


async def add_resource_command(
    ctx: CommandContext[AgentPoolView],
    args: list[str],
    kwargs: dict[str, str],
):
    """Add resource content as context for the next message.

    The first argument is the resource name, remaining kwargs are passed
    to the resource loader.
    """
    if not args:
        msg = "Usage: /add-resource <name> [param1=value1] [param2=value2]"
        await ctx.output.print(msg)
        return

    name = args[0]
    try:
        await ctx.context._agent.conversation.add_context_from_resource(name, **kwargs)
        await ctx.output.print(f"Added resource '{name}' to next message as context.")
    except Exception as e:
        msg = f"Error loading resource: {e}"
        logger.exception(msg)
        raise CommandError(msg) from e


list_resources_cmd = Command(
    name="list-resources",
    description="List available resources",
    execute_func=list_resources,
    help_text=LIST_RESOURCES_HELP,
    category="resources",
)

show_resource_cmd = Command(
    name="show-resource",
    description="Show details and content of a resource",
    execute_func=show_resource,
    usage="<name> [--param1 value1] [--param2 value2]",
    help_text=SHOW_RESOURCES_HELP,
    category="resources",
    completer=CallbackCompleter(get_resource_names),
)


add_resource_cmd = Command(
    name="add-resource",
    description="Add resource content as context",
    execute_func=add_resource_command,
    usage="<name> [param1=value1] [param2=value2]",
    help_text=ADD_RESOURCE_HELP,
    category="resources",
    completer=CallbackCompleter(get_resource_names),
)
