"""Storage configuration."""

from pathlib import Path
from typing import Annotated, Final, Literal

from platformdirs import user_data_dir
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, SecretStr


LogFormat = Literal["chronological", "conversations"]
FilterMode = Literal["and", "override"]
SupportedFormats = Literal["yaml", "toml", "json", "ini"]
FormatType = SupportedFormats | Literal["auto"]

APP_NAME: Final = "llmling-agent"
APP_AUTHOR: Final = "llmling"
DATA_DIR: Final = Path(user_data_dir(APP_NAME, APP_AUTHOR))
DEFAULT_DB_NAME: Final = "history.db"


def get_database_path() -> str:
    """Get the database file path, creating directories if needed."""
    db_path = DATA_DIR / DEFAULT_DB_NAME
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path}"


class BaseStorageProviderConfig(BaseModel):
    """Base storage provider configuration."""

    type: str = Field(init=False)
    """Storage provider type."""

    log_messages: bool = True
    """Whether to log messages"""

    agents: set[str] | None = None
    """Optional set of agent names to include. If None, logs all agents."""

    log_conversations: bool = True
    """Whether to log conversations"""

    log_tool_calls: bool = True
    """Whether to log tool calls"""

    log_commands: bool = True
    """Whether to log command executions"""

    log_context: bool = True
    """Whether to log context messages."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True, extra="forbid")


class SQLStorageConfig(BaseStorageProviderConfig):
    """SQL database storage configuration."""

    type: Literal["sql"] = Field("sql", init=False)
    """SQLModel storage configuration."""

    url: str = Field(default_factory=get_database_path)
    """Database URL (e.g. sqlite:///history.db)"""

    pool_size: int = 5
    """Connection pool size"""

    auto_migration: bool = True
    """Whether to automatically add missing columns"""


class TextLogConfig(BaseStorageProviderConfig):
    """Text log configuration."""

    type: Literal["text_file"] = Field("text_file", init=False)
    """Text log storage configuration."""

    path: str
    """Path to log file"""

    format: LogFormat = "chronological"
    """Log format template to use"""

    template: Literal["chronological", "conversations"] | str | None = "chronological"  # noqa: PYI051
    """Template to use: either predefined name or path to custom template"""

    encoding: str = "utf-8"
    """File encoding"""


# Config:
class FileStorageConfig(BaseStorageProviderConfig):
    """File storage configuration."""

    type: Literal["file"] = Field("file", init=False)
    """File storage configuration."""

    path: str
    """Path to storage file (extension determines format unless specified)"""

    format: FormatType = "auto"
    """Storage format (auto=detect from extension)"""

    encoding: str = "utf-8"
    """File encoding"""


class MemoryStorageConfig(BaseStorageProviderConfig):
    """In-memory storage configuration for testing."""

    type: Literal["memory"] = Field("memory", init=False)
    """In-memory storage configuration for testing."""


class Mem0Config(BaseStorageProviderConfig):
    """Configuration for mem0 storage."""

    type: Literal["mem0"] = Field("mem0", init=False)
    """Mem0 storage config."""

    api_key: SecretStr | None = None
    """API key for mem0 service."""

    page_size: int = 100
    """Number of results per page when retrieving paginated data."""

    output_format: Literal["v1.0", "v1.1"] = "v1.1"
    """API output format version. 1.1 is recommended and provides enhanced details."""


class SupabaseConfig(BaseStorageProviderConfig):
    """Configuration for Supabase storage."""

    type: Literal["supabase"] = "supabase"
    """Supabase storage configuration"""

    key: SecretStr
    """Authentication key for Supabase service"""

    url: HttpUrl | None = None
    """Custom Supabase instance URL. If None, uses project_id to construct URL"""

    project_id: str | None = None
    """Supabase project ID. Required if url is not provided"""

    pool_size: int = Field(default=20, gt=0)
    """Maximum number of database connections in the pool"""

    db_schema: str = "public"
    """Database schema name for multi-tenant setups"""

    @property
    def supabase_url(self) -> str | None:
        """Get Supabase URL, defaulting to cloud if not specified."""
        if self.url:
            return str(self.url)
        if self.project_id:
            return f"https://{self.project_id}.supabase.co"
        return None


StorageProviderConfig = Annotated[
    SQLStorageConfig
    | FileStorageConfig
    | TextLogConfig
    | MemoryStorageConfig
    | SupabaseConfig
    | Mem0Config,
    Field(discriminator="type"),
]


class StorageConfig(BaseModel):
    """Global storage configuration."""

    providers: list[StorageProviderConfig] | None = None
    """List of configured storage providers"""

    default_provider: str | None = None
    """Name of default provider for history queries.
    If None, uses first configured provider."""

    agents: set[str] | None = None
    """Global agent filter. Can be overridden by provider-specific filters."""

    filter_mode: FilterMode = "and"
    """How to combine global and provider agent filters:
    - "and": Both global and provider filters must allow the agent
    - "override": Provider filter overrides global filter if set
    """

    log_messages: bool = True
    """Whether to log messages."""

    log_conversations: bool = True
    """Whether to log conversations."""

    log_tool_calls: bool = True
    """Whether to log tool calls."""

    log_commands: bool = True
    """Whether to log command executions."""

    log_context: bool = True
    """Whether to log additions to the context."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True, extra="forbid")

    @property
    def effective_providers(self) -> list[StorageProviderConfig]:
        """Get effective list of providers.

        Returns:
            - Default SQLite provider if providers is None
            - Empty list if providers is empty list
            - Configured providers otherwise
        """
        if self.providers is None:
            cfg = SQLStorageConfig()
            return [cfg]
        return self.providers
