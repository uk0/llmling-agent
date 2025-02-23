"""Resource management commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from slashed import Command, CommandContext, CommandError
from slashed.completers import CallbackCompleter

from llmling_agent.log import get_logger
from llmling_agent_commands.completers import get_resource_names


if TYPE_CHECKING:
    from llmling_agent.agent.context import AgentContext


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
    ctx: CommandContext[AgentContext],
    args: list[str],
    kwargs: dict[str, str],
):
    """List available resources."""
    try:
        fs = ctx.context.definition.resource_registry.get_fs()
        root = await fs._ls("/", detail=True)

        sections = ["# Available Resources\n"]
        for entry in root:
            protocol = entry["name"].removesuffix("://")
            info = await fs._info(f"{protocol}://")

            desc = f": {info.get('description', '')}" if "description" in info else ""
            sections.append(f"- **{protocol}**{desc}")
            sections.append(f"  Type: {info.get('type', 'unknown')}")
            if uri := info.get("uri"):
                sections.append(f"  URI: `{uri}`")

        await ctx.output.print("\n".join(sections))
    except Exception as e:  # noqa: BLE001
        await ctx.output.print(f"Failed to list resources: {e}")


async def show_resource(
    ctx: CommandContext[AgentContext],
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
        fs = ctx.context.definition.resource_registry.get_fs()

        # Get resource info
        try:
            info = await fs._info(f"{name}://")
        except Exception as e:  # noqa: BLE001
            await ctx.output.print(f"Resource '{name}' not found: {e}")
            return

        sections = [f"# Resource: {name}\n"]
        if typ := info.get("type"):
            sections.append(f"Type: {typ}")
        if uri := info.get("uri"):
            sections.append(f"URI: `{uri}`")
        if desc := info.get("description"):
            sections.append(f"Description: {desc}")
        if mime := info.get("mime_type"):
            sections.append(f"MIME Type: {mime}")

        # Try to list contents
        try:
            listing = await fs._ls(f"{name}://", detail=False)
            if listing:
                sections.extend(["\n# Contents:", "```", *listing, "```"])
        except Exception as e:  # noqa: BLE001
            sections.append(f"\nFailed to list contents: {e}")

        await ctx.output.print("\n".join(sections))
    except Exception as e:  # noqa: BLE001
        await ctx.output.print(f"Error accessing resource: {e}")


async def add_resource_command(
    ctx: CommandContext[AgentContext],
    args: list[str],
    kwargs: dict[str, str],
):
    """Add resource content as context for the next message.

    Examples:
        /add-resource docs              # Add all docs
        /add-resource docs/guide.md     # Add specific file
        /add-resource docs/*.md         # Add all markdown files
    """
    if not args:
        msg = "Usage: /add-resource <resource>[/path] [--pattern pattern]"
        await ctx.output.print(msg)
        return

    try:
        # Parse resource name and path
        parts = args[0].split("/", 1)
        resource_name = parts[0]
        path = parts[1] if len(parts) > 1 else ""

        registry = ctx.context.definition.resource_registry

        if path:
            if "*" in path:
                # It's a pattern - use query
                files = await registry.query(resource_name, pattern=path)
                for file in files:
                    content = await registry.get_content(resource_name, file)
                    ctx.context.agent.conversation.add_context_message(
                        content, source=f"{resource_name}/{file}", **kwargs
                    )
                msg = f"Added {len(files)} files from '{resource_name}' matching '{path}'"
            else:
                # Specific file
                content = await registry.get_content(resource_name, path)
                ctx.context.agent.conversation.add_context_message(
                    content, source=f"{resource_name}/{path}", **kwargs
                )
                msg = f"Added '{resource_name}/{path}' to context"
        else:
            # Add all content from resource root
            files = await registry.query(resource_name)
            for file in files:
                content = await registry.get_content(resource_name, file)
                ctx.context.agent.conversation.add_context_message(
                    content, source=f"{resource_name}/{file}", **kwargs
                )
            msg = f"Added {len(files)} files from '{resource_name}'"

        await ctx.output.print(msg)

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
