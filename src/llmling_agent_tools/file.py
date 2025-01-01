from __future__ import annotations

from upath import UPath


def list_source_files(
    directory: str,
    extensions: list[str] | None = None,
) -> list[str]:
    """Get a list of source code files in a directory.

    Use this tool to find all relevant source files that need documentation.

    Args:
        directory: Path to the source code directory
        extensions: Optional list of file extensions to include (e.g. ['.py', '.js'])
                   If not provided, includes all files

    Returns:
        List of absolute file paths

    Example:
        list_source_files("src", [".py"]) -> ["/full/path/src/models/user.py", ...]
    """
    path = UPath(directory).resolve()  # Get absolute path
    if not path.exists():
        msg = f"Directory not found: {directory}"
        raise ValueError(msg)

    # Default to common source file extensions
    default_extensions = [".py", ".js", ".ts", ".java", ".cpp", ".cs", ".go", ".rs"]
    exts = extensions or default_extensions

    # Find all matching files recursively
    files: list[str] = []
    for ext in exts:
        files.extend(
            str(f.resolve())  # Return absolute paths
            for f in path.rglob(f"*{ext}")
            if not any(p.startswith(".") for p in f.parts)  # Skip hidden dirs
        )

    return sorted(files)


def read_source_file(filepath: str) -> str:
    """Read the content of a source code file.

    Use this tool to get the actual content of source files for documentation.

    Args:
        filepath: Path to the file to read (absolute or relative)

    Returns:
        Content of the file as text

    """
    path = UPath(filepath).resolve()  # Convert to absolute path
    if not path.exists():
        msg = f"File not found: {filepath}"
        raise ValueError(msg)

    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        msg = f"Failed to read {filepath}: {e}"
        raise ValueError(msg) from e
