# Understanding LLMling's Core Functionality

This document helps you understand how LLMling's core library works when analyzing codebases or helping with integrations.

## Core Concept: RuntimeConfig

RuntimeConfig is the central interface for all LLMling operations. It is created from a Config (loaded from YAML) and acts as a session manager that provides access to:
- Resources (any loadable content: files, API results, CLI output)
- Tools (executable Python functions with schemas)
- Processors (content transformers with pipelines)

Here are its essential methods / types and their purposes:

## Key Types

```python
class LoadedResource(BaseModel):
    """Result of loading any resource."""
    content: str  # The actual content
    source_type: str  # Type of resource (path, text, cli, etc.)
    metadata: ResourceMetadata  # URI, mime-type, etc.

class ProcessorResult(BaseModel):
    """Result of processing content."""
    content: str  # Processed content
    original_content: str  # Input content
    metadata: dict[str, Any]  # Processing info

class Tool(BaseModel):
    """Executable function with schema."""
    name: str
    description: str
    schema: dict  # OpenAI function schema

## Complete RuntimeConfig Interface


All operations that might involve I/O are async. The interface follows a consistent pattern
of sync methods for discovery/configuration and async methods for actual operations.

```python
class RuntimeConfig:
    # Main Entry Points
    @classmethod
    async def open(cls, source: str | Config, *, validate: bool = True) -> RuntimeConfig:
        """Primary way to create and use RuntimeConfig."""

    @classmethod
    def from_config(cls, config: Config) -> RuntimeConfig:
        """Create from existing Config object."""

    # Resource Operations
    async def load_resource(self, name: str) -> LoadedResource:
        """Load a resource by name."""

    async def load_resource_by_uri(self, uri: str) -> LoadedResource:
        """Load a resource using URI."""

    def list_resource_names(self) -> Sequence[str]:
        """Get all available resource names."""

    def list_resource_uris(self) -> Sequence[str]:
        """Get URIs for all resources."""

    def get_resource_uri(self, name: str) -> str:
        """Get URI for a specific resource."""

    def get_resource(self, name: str) -> BaseResource:
        """Get resource configuration."""

    def register_resource(self, name: str, resource: BaseResource, *, replace: bool = False):
        """Register a new resource."""

    # Tool Operations
    async def execute_tool(self, name: str, **params: Any) -> Any:
        """Execute a tool with parameters."""

    def list_tool_names(self) -> Sequence[str]:
        """Get all available tool names."""

    def get_tool(self, name: str) -> Tool:
        """Get tool by name."""

    @property
    def tools(self) -> dict[str, Tool]:
        """Get all registered tools."""

    # Processing Operations
    async def process_content(self, content: str, processor_name: str, **kwargs):
        """Process content through a processor."""

    # Prompt Operations
    async def render_prompt(self, name: str, arguments: dict[str, Any] | None = None):
        """Render a prompt with arguments."""

    def list_prompt_names(self) -> Sequence[str]:
        """Get all available prompt names."""

    def get_prompt(self, name: str) -> Prompt:
        """Get prompt by name."""

    # Completion Support
    async def get_prompt_completions(self, current: str, arg_name: str, prompt_name: str):
        """Get completions for prompt arguments."""

    async def get_resource_completions(self, uri: str, current: str, arg_name: str = None):
        """Get completions for resource arguments."""

    # Configuration Access
    @property
    def original_config(self) -> Config:
        """Access the underlying configuration."""
```

In practice, the most commonly used methods are:
1. `open()` - Creating a runtime instance
2. `load_resource()` / `load_resource_by_uri()` - Loading content
3. `execute_tool()` - Running tools
4. `process_content()` - Transforming content

Other methods are typically used for:
- Discovery (list_* methods)
- Configuration (register_* methods)
- Completions (get_*_completions methods)
- Direct access (get_* methods)


The typical flow is:
1. Load resources
2. Process their content
3. Execute tools with the results

## Configuration Structure

Resources, tools, and processors are defined in YAML:

```yaml
resources:
  repo_contents:              # Resource name
    type: path               # How to load it
    path: "./src"           # Where to load from
    description: "Source"   # What it contains

tools:
  analyze:                   # Tool name
    import_path: "pkg.tool" # Python function to use
    description: "Analyze"  # What it does

processors:
  parse:                    # Processor name
    import_path: "pkg.proc" # Transform function
```

## Integration Pattern

When building integrations with LLMling, you'll typically:
1. Create a RuntimeConfig instance from configuration
2. Use it to access resources/tools/processors
3. Wrap the results in your integration's format

The core library is transport-agnostic - it doesn't care if it's being used by an agent, server, or other integration.

Example of core library usage:
```python
async with RuntimeConfig.open("config.yml") as runtime:
    # All functionality flows through RuntimeConfig
    resource = await runtime.load_resource("repo_contents")
    result = await runtime.execute_tool("analyze", content=resource.content)
```

When you see this pattern, you know you're interacting with LLMling's core functionality.

## Error Handling Patterns

RuntimeConfig operations follow consistent error patterns:

```python
# Core exception hierarchy
class LLMLingError(Exception): ...
class ResourceError(LLMLingError): ...
class ToolError(LLMLingError): ...
class ProcessorError(LLMLingError): ...

# Typical error handling pattern
async with RuntimeConfig.open("config.yml") as runtime:
    try:
        resource = await runtime.load_resource("missing")
    except ResourceError as exc:
        # Resource-specific error handling
    except LLMLingError as exc:
        # Generic error handling
```

Key patterns:
- All errors inherit from LLMLingError
- Specific errors for each subsystem (resources/tools/processors)
- Errors include context about what failed
- Operations fail early and explicitly


## Event System

RuntimeConfig includes an event system for monitoring changes:

```python
class EventType(Enum):
    RESOURCE_ADDED = auto()
    RESOURCE_MODIFIED = auto()
    TOOL_ADDED = auto()
    PROMPT_MODIFIED = auto()
    # etc.

# Register for events
runtime.add_observer(observer, "resource")  # or "tool", "prompt"
```

This is particularly useful when building interactive integrations that need to
react to changes in resources, tools, or prompts.
