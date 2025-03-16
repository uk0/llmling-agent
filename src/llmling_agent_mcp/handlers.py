"""MCP protocol request handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mcp
from mcp import types

from llmling_agent_mcp import constants, conversions
from llmling_agent_mcp.log import get_logger


if TYPE_CHECKING:
    from pydantic import AnyUrl

    from llmling_agent_mcp.server import LLMLingServer


logger = get_logger(__name__)


def register_handlers(llm_server: LLMLingServer):  # noqa: PLR0915
    """Register all MCP protocol handlers.

    Args:
        llm_server: LLMLing server instance
    """

    @llm_server.server.set_logging_level()
    async def handle_set_level(level: mcp.LoggingLevel):
        """Handle logging level changes."""
        try:
            python_level = constants.MCP_TO_LOGGING[level]
            logger.setLevel(python_level)
            data = f"Log level set to {level}"
            await llm_server.current_session.send_log_message(
                level, data, logger=llm_server.name
            )
        except Exception as exc:
            error_data = mcp.ErrorData(
                message="Error setting log level",
                code=types.INTERNAL_ERROR,
                data=str(exc),
            )
            error = mcp.McpError(error_data)
            raise error from exc

    @llm_server.server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        """Handle tools/list request."""
        tools = await llm_server.provider.get_tools()
        return [conversions.to_mcp_tool(tool) for tool in tools]

    @llm_server.server.call_tool()
    async def handle_call_tool(
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> list[types.TextContent]:
        """Handle tools/call request."""
        arguments = arguments or {}
        # Filter out _meta from arguments
        args = {k: v for k, v in arguments.items() if not k.startswith("_")}

        tools = await llm_server.provider.get_tools()
        tool = next((t for t in tools if t.name == name), None)
        if not tool:
            msg = f"Tool not found: {name}"
            raise ValueError(msg)

        try:
            result = await tool.execute(**args)
            return [types.TextContent(type="text", text=str(result))]
        except Exception as exc:
            logger.exception("Tool execution failed: %s", name)
            error_msg = f"Tool execution failed: {exc}"
            return [types.TextContent(type="text", text=error_msg)]

    @llm_server.server.list_prompts()
    async def handle_list_prompts() -> list[types.Prompt]:
        """Handle prompts/list request."""
        prompts = await llm_server.provider.get_prompts()
        return [conversions.to_mcp_prompt(p) for p in prompts]

    async def handle_get_prompt(
        name: str,
        arguments: dict[str, str] | None = None,
    ) -> types.GetPromptResult:
        """Handle prompts/get request."""
        try:
            message = await llm_server.provider.get_formatted_prompt(name, arguments)
            mcp_msg = conversions.to_mcp_message(message)
            return types.GetPromptResult(
                messages=[mcp_msg],
                description=message.metadata.get("prompt_name"),  # type: ignore
            )
        except KeyError as exc:
            error_data = mcp.ErrorData(code=types.INVALID_PARAMS, message=str(exc))
            raise mcp.McpError(error_data) from exc
        except Exception as exc:
            error_data = mcp.ErrorData(code=types.INTERNAL_ERROR, message=str(exc))
            raise mcp.McpError(error_data) from exc

    # @llm_server.server.list_resources()
    # async def handle_list_resources() -> list[types.Resource]:
    #     """Handle resources/list request."""
    #     resources: list[types.Resource] = []
    #     for name in llm_server.runtime.list_resource_names():
    #         try:
    #             uri = llm_server.runtime.get_resource_uri(name)
    #             mcp_uri = conversions.to_mcp_uri(uri)
    #             dsc = llm_server.runtime._config.resources[name].description
    #             mime = "text/plain"  # Default, could be made more specific
    #             res = types.Resource(
    #                 uri=mcp_uri, name=name, description=dsc, mimeType=mime
    #             )
    #             resources.append(res)
    #         except Exception:
    #             error_msg = "Failed to create resource listing for %r. Config: %r"
    #             cfg = llm_server.runtime._config.resources.get(name)
    #             logger.exception(error_msg, name, cfg)
    #             continue

    #     return resources

    # @llm_server.server.list_resource_templates()
    # async def handle_list_resource_templates() -> list[types.ResourceTemplate]:
    #     """Handle resource template listing request."""
    #     templates: list[types.ResourceTemplate] = []
    #     for resource in llm_server.runtime.get_resources():
    #         if not resource.is_templated():
    #             continue

    #         # Get the loader for proper URI template creation
    #         loader = llm_server.runtime.get_resource_loader(resource)
    #         uri_template = loader.get_uri_template()
    #         templ = types.ResourceTemplate(
    #             uriTemplate=uri_template,
    #             name=resource.uri or "",  # this is fishy, we need display name?
    #             description=resource.description,
    #             mimeType=resource.mime_type,
    #         )
    #         templates.append(templ)

    #     return templates

    # @llm_server.server.read_resource()
    # async def handle_read_resource(uri: AnyUrl) -> str | bytes:
    #     """Handle direct resource content requests."""
    #     try:
    #         internal_uri = conversions.from_mcp_uri(str(uri))
    #         logger.debug("Loading resource from internal URI: %s", internal_uri)

    #         if "://" not in internal_uri:
    #             resource = await llm_server.runtime.load_resource(internal_uri)
    #         else:
    #             resource = await llm_server.runtime.load_resource_by_uri(internal_uri)

    #         if resource.metadata.mime_type.startswith("text/"):
    #             return resource.content
    #         return resource.content_items[0].content.encode()

    #     except Exception as exc:
    #         error_msg = f"Failed to read resource: {exc}"
    #         logger.exception(error_msg)
    #         error_data = mcp.ErrorData(
    #             message=error_msg, code=types.INTERNAL_ERROR, data=exc
    #         )
    #         error = mcp.McpError(error_data)
    #         raise error from exc

    # @llm_server.server.completion()
    # async def handle_completion(
    #     ref: types.PromptReference | types.ResourceReference,
    #     argument: types.CompletionArgument,
    # ) -> types.Completion:
    #     """Handle completion requests."""
    #     try:
    #         match ref:
    #             case types.PromptReference():
    #                 values = await llm_server.runtime.get_prompt_completions(
    #                     current_value=argument.value,
    #                     argument_name=argument.name,
    #                     prompt_name=ref.name,
    #                 )
    #             case types.ResourceReference():
    #                 values = await llm_server.runtime.get_resource_completions(
    #                     uri=ref.uri,
    #                     current_value=argument.value,
    #                     argument_name=argument.name,
    #                 )
    #             case _:
    #                 msg = f"Invalid reference type: {type(ref)}"
    #                 raise ValueError(msg)

    #         return types.Completion(
    #             values=values[:100],
    #             total=len(values),
    #             hasMore=len(values) > 100,
    #         )
    #     except Exception:
    #         logger.exception("Completion failed")
    #         return types.Completion(values=[], total=0, hasMore=False)

    @llm_server.server.progress_notification()
    async def handle_progress(
        token: str | int,
        progress: float,
        total: float | None,
    ):
        """Handle progress notifications from client."""
        msg = "Progress notification: %s %.1f/%.1f"
        logger.debug(msg, token, progress, total or 0.0)

    @llm_server.server.subscribe_resource()
    async def handle_subscribe(uri: AnyUrl):
        """Subscribe to resource updates."""
        uri_str = str(uri)
        llm_server._subscriptions[uri_str].add(llm_server.current_session)
        logger.debug("Added subscription for %s", uri)

    @llm_server.server.unsubscribe_resource()
    async def handle_unsubscribe(uri: AnyUrl):
        """Unsubscribe from resource updates."""
        if (uri_str := str(uri)) in llm_server._subscriptions:
            llm_server._subscriptions[uri_str].discard(llm_server.current_session)
            if not llm_server._subscriptions[uri_str]:
                del llm_server._subscriptions[uri_str]
            msg = "Removed subscription for %s: %s"
            logger.debug(msg, uri, llm_server.current_session)
