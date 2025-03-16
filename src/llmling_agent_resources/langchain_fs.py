"""LangChain filesystem implementation."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any, Literal

import fsspec
from fsspec.asyn import AsyncFileSystem, sync_wrapper
from upath import UPath, registry


if TYPE_CHECKING:
    from langchain.document_loaders import BaseLoader
    from langchain.schema import Document


class LangchainPath(UPath):
    """UPath implementation for browsing Langchain document loaders."""

    __slots__ = ()

    def iterdir(self):
        if not self.is_dir():
            raise NotADirectoryError(str(self))
        yield from super().iterdir()

    @property
    def path(self) -> str:
        path = super().path
        return "/" if path == "." else path


class LangChainFileSystem(AsyncFileSystem):
    """Filesystem interface for LangChain document loaders."""

    protocol = "langchain"

    def __init__(
        self,
        loader: BaseLoader,
        target_protocol: str | None = None,
        target_options: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the filesystem.

        Args:
            loader: LangChain document loader instance
            target_protocol: Optional protocol for the source files
            target_options: Optional options for the target protocol
            args: Additional arguments for the filesystem
            kwargs: Additional keyword arguments for the filesystem
        """
        super().__init__(*args, **kwargs)
        self.loader = loader
        self.target_protocol = target_protocol
        self.target_options = target_options or {}
        self._documents: dict[str, Document] = {}
        self._loaded = False

    def _make_path(self, path: str) -> UPath:
        """Create a path object from string."""
        return LangchainPath(path)

    async def _load_documents(self):
        """Load documents if not already loaded."""
        if self._loaded:
            return

        # Convert sync loader.load() to async
        assert self.loop
        docs = await self.loop.run_in_executor(None, self.loader.load)
        for i, doc in enumerate(docs):
            path = doc.metadata.get("title", f"doc_{i}.txt")
            self._documents[path] = doc
        self._loaded = True

    async def _ls(
        self,
        path: str,
        detail: bool = True,
        **kwargs: Any,
    ) -> list[dict[str, Any]] | list[str]:
        """List available documents."""
        await self._load_documents()

        if detail:
            return [
                {
                    "name": name,
                    "size": len(doc.page_content),
                    "type": "file",
                    **doc.metadata,
                }
                for name, doc in self._documents.items()
            ]
        return list(self._documents.keys())

    ls = sync_wrapper(_ls)  # type: ignore

    async def _cat_file(self, path: str, **kwargs: Any) -> bytes:
        """Read document content."""
        await self._load_documents()

        path = self._strip_protocol(path).strip("/")  # type: ignore
        if path not in self._documents:
            msg = f"Document not found: {path}"
            raise FileNotFoundError(msg)

        return self._documents[path].page_content.encode()

    async def _open(
        self,
        path: str,
        mode: Literal["rb", "r"] = "rb",
        **kwargs: Any,
    ) -> Any:
        """Provide file-like access to document content."""
        await self._load_documents()

        path = self._strip_protocol(path).strip("/")  # type: ignore
        if path not in self._documents:
            msg = f"Document not found: {path}"
            raise FileNotFoundError(msg)

        content = self._documents[path].page_content.encode()
        return io.BytesIO(content)

    async def _info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """Get document info."""
        await self._load_documents()

        path = self._strip_protocol(path).strip("/")  # type: ignore
        if path not in self._documents:
            msg = f"Document not found: {path}"
            raise FileNotFoundError(msg)

        doc = self._documents[path]
        return {
            "name": path,
            "size": len(doc.page_content),
            "type": "file",
            **doc.metadata,
        }

    async def _isfile(self, path: str) -> bool:
        """Check if path is a file."""
        await self._load_documents()
        return self._strip_protocol(path).strip("/") in self._documents  # type: ignore

    async def _isdir(self, path: str) -> bool:
        """Always False for our flat structure."""
        return False

    @property
    def fsid(self) -> str:
        """Unique identifier for this filesystem instance."""
        return f"langchain_{id(self.loader)}"

    def ukey(self, path: str) -> str:
        """Unique key for caching."""
        return f"{self.fsid}:{path}"


# Register the filesystem implementation
fsspec.register_implementation("langchain", LangChainFileSystem, clobber=True)
registry.register_implementation("langchain", LangchainPath, clobber=True)


if __name__ == "__main__":
    import asyncio
    from pathlib import Path

    from langchain_community.document_loaders import TextLoader

    async def main():
        # Create a sample text file for testing
        test_file = Path("test_doc.txt")
        test_content = "This is a test document.\nIt has multiple lines.\n"
        test_file.write_text(test_content)

        try:
            # Create a loader and filesystem
            loader = TextLoader(str(test_file))
            fs = fsspec.filesystem("langchain", loader=loader)

            # List all documents
            print("\nListing documents:")
            files = fs.ls("")
            print(files)

            # Read content
            print("\nReading content:")
            content = fs.cat(test_file.name).decode()
            print(content)

            # Get file info
            print("\nFile info:")
            info = fs.info(test_file.name)
            print(info)

            # Test file-like access
            print("\nReading using open:")
            with fs.open(test_file.name) as f:
                print(f.read().decode())

        finally:
            # Cleanup
            test_file.unlink()

    # Run the test
    asyncio.run(main())
