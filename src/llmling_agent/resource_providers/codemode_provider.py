"""Meta-resource provider that exposes tools through Python execution."""

from __future__ import annotations

import ast
import json
import subprocess
import tempfile
from typing import TYPE_CHECKING, Any

from schemez import create_schema

from llmling_agent.resource_providers.base import ResourceProvider
from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    from collections.abc import Sequence


class CodeModeResourceProvider(ResourceProvider):
    """Provider that wraps tools into a single Python execution environment."""

    def __init__(
        self,
        wrapped_providers: Sequence[ResourceProvider] | None = None,
        wrapped_tools: Sequence[Tool] | None = None,
        name: str = "meta_tools",
        include_signatures: bool = True,
        include_docstrings: bool = True,
    ):
        """Initialize meta provider.

        Args:
            wrapped_providers: Providers whose tools to wrap
            wrapped_tools: Individual tools to wrap
            name: Provider name
            include_signatures: Include function signatures in documentation
            include_docstrings: Include function docstrings in documentation
        """
        super().__init__(name=name)
        self.wrapped_providers = list(wrapped_providers or [])
        self.wrapped_tools = list(wrapped_tools or [])
        self.include_signatures = include_signatures
        self.include_docstrings = include_docstrings

    async def get_tools(self) -> list[Tool]:
        """Return single meta-tool for Python execution with available tools."""
        desc = await self._build_tool_description()
        return [Tool.from_callable(self.execute_codemode, description_override=desc)]

    async def execute_codemode(
        self, python_code: str, context_vars: dict[str, Any] | None = None
    ) -> Any:
        """Execute Python code with all wrapped tools available as functions.

        Args:
            python_code: Python code to execute
            context_vars: Additional variables to make available

        Returns:
            Result of the last expression or explicit return value
        """
        # Build execution namespace
        namespace = await self._build_execution_namespace()
        if context_vars:
            namespace.update(context_vars)

        # Parse the code to check for return statements or _result assignment
        try:
            tree = ast.parse(python_code)
            has_return = any(isinstance(node, ast.Return) for node in ast.walk(tree))
            any(
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "_result"
                    for target in node.targets
                )
                for node in ast.walk(tree)
            )
        except SyntaxError:
            has_return = False

        # Execute the code
        if has_return:
            # Code has explicit returns, execute as function
            func_code = f"""
async def _exec_func():
{chr(10).join("    " + line for line in python_code.splitlines())}
"""
            exec(func_code, namespace)
            return await namespace["_exec_func"]()
        # Execute directly and check for _result or last expression
        try:
            # Try as expression first for single expressions
            return eval(compile(python_code, "<meta_tool>", "eval"), namespace)
        except SyntaxError:
            # Execute as statements
            exec(compile(python_code, "<meta_tool>", "exec"), namespace)
            # Return _result if set, otherwise None
            return namespace.get("_result")

    async def _build_tool_description(self) -> str:
        """Generate comprehensive tool description with available functions."""
        all_tools = await self._collect_all_tools()

        if not all_tools:
            return "Execute Python code (no tools available)"

        # Generate return type models if datamodel-codegen is available
        return_models = await self._generate_return_models(all_tools)

        parts = [
            "Execute Python code with the following tools available as async functions:",
            "",
        ]

        if return_models:
            parts.extend([
                "# Generated return type models",
                return_models,
                "",
                "# Available functions:",
                "",
            ])

        for tool in all_tools:
            if self.include_signatures:
                # Use schemez to get proper signature
                signature = await self._get_function_signature(tool)
                parts.append(f"async def {signature}:")
            else:
                parts.append(f"async def {tool.name}(...):")

            # Add docstring if available
            if self.include_docstrings and tool.description:
                # Properly indent docstring
                indented_desc = "    " + tool.description.replace("\n", "\n    ")
                parts.append(f'    """{indented_desc}"""')
            parts.append("")

        parts.extend([
            "Usage notes:",
            "- All tool functions are async, use 'await'",
            "- Set '_result' variable to control return value",
            "- Use 'return' statements for early returns",
            "- Last expression value returned if no _result or return",
        ])

        return "\n".join(parts)

    async def _get_function_signature(self, tool: Tool) -> str:
        """Extract function signature using schemez."""
        try:
            # Get the actual callable from the tool
            callable_func = tool.callable.callable

            # Create schema using schemez
            schema = create_schema(callable_func)

            # Convert back to Python signature
            sig = schema.to_python_signature()

            # Try to get return type model name
            return_model_name = await self._get_return_model_name(tool)

            # Format as function signature string
        except Exception:  # noqa: BLE001
            # Fallback to basic signature extraction
            return await self._extract_basic_signature(tool)
        else:
            return f"{tool.name}{sig} -> {return_model_name}"

    async def _extract_basic_signature(self, tool: Tool) -> str:
        """Fallback signature extraction from tool schema."""
        schema = tool.schema["function"]
        params = schema.get("parameters", {}).get("properties", {})
        required = set(schema.get("required", []))  # type: ignore

        param_strs = []
        for name, param_info in params.items():
            # Map JSON Schema types to Python types
            type_map = {
                "string": "str",
                "integer": "int",
                "number": "float",
                "boolean": "bool",
                "array": "list",
                "object": "dict",
            }

            param_type = param_info.get("type", "Any")
            type_hint = type_map.get(param_type, "Any")

            if name not in required:
                param_strs.append(f"{name}: {type_hint} = None")
            else:
                param_strs.append(f"{name}: {type_hint}")

        return f"{tool.name}({', '.join(param_strs)})"

    async def _build_execution_namespace(self) -> dict[str, Any]:
        """Build Python namespace with tool functions."""
        namespace = {
            "__builtins__": __builtins__,
            "_result": None,  # Allow explicit result setting
        }

        all_tools = await self._collect_all_tools()

        # Create async wrapper functions for each tool
        for tool in all_tools:
            # Create closure to capture tool
            def make_tool_func(t: Tool):
                async def tool_func(*args, **kwargs):
                    return await t.execute(*args, **kwargs)

                # Copy function metadata for better introspection
                tool_func.__name__ = t.name
                tool_func.__doc__ = t.description
                return tool_func

            namespace[tool.name] = make_tool_func(tool)

        return namespace

    async def _collect_all_tools(self) -> list[Tool]:
        """Collect all tools from providers and direct tools."""
        all_tools = list(self.wrapped_tools)

        for provider in self.wrapped_providers:
            if provider.requires_async:
                async with provider:
                    provider_tools = await provider.get_tools()
            else:
                provider_tools = await provider.get_tools()
            all_tools.extend(provider_tools)

        return all_tools

    async def _generate_return_models(self, all_tools: list[Tool]) -> str:
        """Generate Pydantic models for tool return types using datamodel-codegen."""
        try:
            # Check if datamodel-codegen is available
            subprocess.run(
                ["datamodel-codegen", "--version"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            # datamodel-codegen not available, skip model generation
            return ""

        model_parts = []

        for tool in all_tools:
            try:
                # Get return schema from schemez
                callable_func = tool.callable.callable
                schema = create_schema(callable_func)
                return_schema = schema.returns

                # Skip if return type is too simple
                if return_schema.get("type") not in {"object", "array"}:
                    continue

                # Create temporary file for schema
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as f:
                    json.dump(return_schema, f)
                    schema_file = f.name

                # Generate model using datamodel-codegen
                result = subprocess.run(
                    [
                        "datamodel-codegen",
                        "--input",
                        schema_file,
                        "--input-file-type",
                        "jsonschema",
                        "--output-model-type",
                        "pydantic.BaseModel",
                        "--class-name",
                        f"{tool.name.title()}Response",
                        "--disable-timestamp",
                        "--use-union-operator",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                if result.stdout.strip():
                    model_parts.append(result.stdout.strip())

            except Exception:  # noqa: BLE001
                # Skip this tool if model generation fails
                continue

        return "\n\n".join(model_parts) if model_parts else ""

    async def _get_return_model_name(self, tool: Tool) -> str:
        """Get the return model name for a tool."""
        try:
            callable_func = tool.callable.callable
            schema = create_schema(callable_func)
            return_schema = schema.returns

            if return_schema.get("type") == "object":
                return f"{tool.name.title()}Response"
            if return_schema.get("type") == "array":
                return f"list[{tool.name.title()}Item]"
            # Map simple types
            type_map = {
                "string": "str",
                "integer": "int",
                "number": "float",
                "boolean": "bool",
            }
            return type_map.get(return_schema.get("type", "string"), "Any")
        except Exception:  # noqa: BLE001
            return "Any"
