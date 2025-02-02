from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Literal, overload

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from fsspec.asyn import AsyncFileSystem

    from llmling_agent.common_types import StrPath


logger = get_logger(__name__)


@lru_cache(maxsize=32)
def _get_cached_fs(protocol: str) -> AsyncFileSystem:
    """Cached filesystem creation."""
    import fsspec
    from fsspec.asyn import AsyncFileSystem
    from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper
    from morefs.asyn_local import AsyncLocalFileSystem

    if protocol in ("", "file"):
        return AsyncLocalFileSystem()

    fs = fsspec.filesystem(protocol, asynchronous=True)
    if not isinstance(fs, AsyncFileSystem):
        fs = AsyncFileSystemWrapper(fs)
    return fs


async def get_async_fs(path: StrPath) -> AsyncFileSystem:
    """Get appropriate async filesystem for path."""
    from upath import UPath

    path_obj = UPath(path)
    return _get_cached_fs(path_obj.protocol)


@overload
async def read_path(
    path: StrPath, mode: Literal["rt"] = "rt", encoding: str = ...
) -> str: ...


@overload
async def read_path(path: StrPath, mode: Literal["rb"], encoding: str = ...) -> bytes: ...


async def read_path(
    path: StrPath,
    mode: Literal["rt", "rb"] = "rt",
    encoding: str = "utf-8",
) -> str | bytes:
    """Read file content asynchronously when possible.

    Args:
        path: Path to read
        mode: Read mode ("rt" for text, "rb" for binary)
        encoding: File encoding for text files

    Returns:
        File content as string or bytes depending on mode
    """
    from upath import UPath

    path_obj = UPath(path)
    fs = await get_async_fs(path_obj)

    f = await fs.open_async(path_obj.path, mode=mode)
    async with f:
        return await f.read()


if __name__ == "__main__":
    import asyncio

    async def main():
        content = await read_path("README.md")
        print(content)

    asyncio.run(main())
