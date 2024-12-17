"""Resource management commands."""

from __future__ import annotations

from llmling_agent.commands.base import Command, CommandContext


async def list_resources(
    ctx: CommandContext,
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """List available resources."""
    try:
        resources = ctx.session._agent.runtime.get_resources()

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
    ctx: CommandContext,
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """Show details or content of a resource."""
    if not args:
        await ctx.output.print(
            "Usage: /show-resource <name> [--param1 value1] [--param2 value2]"
        )
        return

    name = args[0]
    try:
        # First get resource info
        resources = ctx.session._agent.runtime.get_resources()
        resource_info = next((r for r in resources if r.name == name), None)
        if not resource_info:
            await ctx.output.print(f"Resource '{name}' not found")
            return

        # Show resource details
        sections = [f"# Resource: {name}\n", f"Type: {resource_info.type}"]
        if resource_info.uri:
            sections.append(f"URI: `{resource_info.uri}`")
        if resource_info.description:
            sections.append(f"Description: {resource_info.description}")

        # Show if resource is templated
        if resource_info.is_templated():
            sections.append("\nParameters supported")

        # Show MIME type if available
        sections.append(f"MIME Type: {resource_info.mime_type}")

        # Try to load content with provided parameters
        try:
            content = await ctx.session._agent.runtime.load_resource(name, **kwargs)
            sections.extend(["\n# Content:", "```", str(content), "```"])
        except Exception as e:  # noqa: BLE001
            sections.append(f"\nFailed to load content: {e}")

        await ctx.output.print("\n".join(sections))
    except Exception as e:  # noqa: BLE001
        await ctx.output.print(f"Error accessing resource: {e}")


list_resources_cmd = Command(
    name="list-resources",
    description="List available resources",
    execute_func=list_resources,
    help_text=(
        "Display all resources available to the agent.\n\n"
        "Shows:\n"
        "- Resource names and descriptions\n"
        "- Resource types and URIs\n"
        "- Whether parameters are supported\n"
        "- MIME types\n\n"
        "Resource types can be:\n"
        "- path: Files or URLs\n"
        "- text: Raw text content\n"
        "- cli: Command line tools\n"
        "- source: Python source code\n"
        "- callable: Python functions\n"
        "- image: Image files\n\n"
        "Use /show-resource for detailed information about specific resources."
    ),
    category="resources",
)

show_resource_cmd = Command(
    name="show-resource",
    description="Show details and content of a resource",
    execute_func=show_resource,
    usage="<name> [--param1 value1] [--param2 value2]",
    help_text=(
        "Display detailed information and content of a specific resource.\n\n"
        "Shows:\n"
        "- Resource metadata (type, URI, description)\n"
        "- MIME type information\n"
        "- Parameter support status\n"
        "- Resource content (if loadable)\n\n"
        "For resources that support parameters:\n"
        "- Pass parameters as --param arguments\n"
        "- Parameters are passed to resource loader\n\n"
        "Examples:\n"
        "  /show-resource config.yml               # Show configuration file\n"
        "  /show-resource template --date today    # Template with parameters\n"
        "  /show-resource image.png               # Show image details\n"
        "  /show-resource api --key value         # API with parameters\n\n"
        "Note: Some resources might require parameters to be viewed."
    ),
    category="resources",
)
