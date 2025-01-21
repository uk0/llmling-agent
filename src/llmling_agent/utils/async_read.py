from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from fsspec.implementations.asyn_wrapper import AsyncFileSystem, AsyncFileSystemWrapper
from upath import UPath

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from os import PathLike


logger = get_logger(__name__)


@overload
async def read_path(path: str | PathLike[str], mode: Literal["rt"] = "rt") -> str: ...


@overload
async def read_path(path: str | PathLike[str], mode: Literal["rb"]) -> bytes: ...


async def read_path(
    path: str | PathLike[str],
    mode: Literal["rt", "rb"] = "rt",
) -> str | bytes:
    """Read file content asynchronously.

    Args:
        path: Path to read
        mode: Read mode ("rt" for text, "rb" for binary)

    Returns:
        File content as string or bytes depending on mode
    """
    path_obj = UPath(path)
    fs = path_obj.fs
    if not isinstance(fs, AsyncFileSystem):
        fs = AsyncFileSystemWrapper(fs)
    f = await fs.open_async(path_obj._path, mode=mode)
    async with f:
        return await f.read()
