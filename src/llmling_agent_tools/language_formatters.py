"""Language formatters with dependency-injected command execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LintResult:
    """Result of a linting operation."""

    success: bool
    output: str
    errors: str
    fixed_issues: int = 0
    remaining_issues: int = 0


@dataclass
class FormatResult:
    """Result of a formatting operation."""

    success: bool
    output: str
    errors: str
    formatted: bool = False


@dataclass
class FormatAndLintResult:
    """Combined result of format and lint operations."""

    format_result: FormatResult
    lint_result: LintResult

    @property
    def success(self) -> bool:
        """Overall success if both operations succeeded."""
        return self.format_result.success and self.lint_result.success


# Type alias for command handler
CommandHandler = Callable[[list[str]], Awaitable[tuple[int, str, str]]]


async def default_command_handler(cmd: list[str]) -> tuple[int, str, str]:
    """Default subprocess-based command handler."""
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    return (
        process.returncode or 0,
        stdout.decode() if stdout else "",
        stderr.decode() if stderr else "",
    )


class LanguageFormatter(ABC):
    """Abstract base class for language-specific formatters."""

    def __init__(self, command_handler: CommandHandler = default_command_handler):
        self.command_handler = command_handler

    @property
    @abstractmethod
    def name(self) -> str:
        """Language name (e.g., 'Python', 'TOML')."""

    @property
    @abstractmethod
    def extensions(self) -> list[str]:
        """File extensions this formatter handles (e.g., ['.py', '.pyi'])."""

    @abstractmethod
    async def format(self, path: Path) -> FormatResult:
        """Format a file."""

    @abstractmethod
    async def lint(self, path: Path, fix: bool = False) -> LintResult:
        """Lint a file, optionally fixing issues."""

    async def format_and_lint(self, path: Path, fix: bool = False) -> FormatAndLintResult:
        """Format and then lint a file."""
        format_result = await self.format(path)
        lint_result = await self.lint(path, fix=fix)
        return FormatAndLintResult(format_result, lint_result)

    def can_handle(self, path: Path) -> bool:
        """Check if this formatter can handle the given file."""
        return path.suffix.lower() in self.extensions


class PythonFormatter(LanguageFormatter):
    """Python formatter using ruff."""

    @property
    def name(self) -> str:
        return "Python"

    @property
    def extensions(self) -> list[str]:
        return [".py", ".pyi"]

    async def format(self, path: Path) -> FormatResult:
        cmd = ["uv", "run", "ruff", "format", str(path)]
        return_code, stdout, stderr = await self.command_handler(cmd)

        return FormatResult(
            success=return_code == 0,
            output=stdout,
            errors=stderr,
            formatted=return_code == 0,  # ruff format returns 0 on success
        )

    async def lint(self, path: Path, fix: bool = False) -> LintResult:
        cmd = ["uv", "run", "ruff", "check"]
        if fix:
            cmd.extend(["--fix", "--unsafe-fixes"])
        cmd.append(str(path))

        return_code, stdout, stderr = await self.command_handler(cmd)

        return LintResult(
            success=return_code == 0,
            output=stdout,
            errors=stderr,
            fixed_issues=0,  # Could parse output to count fixes
            remaining_issues=0,  # Could parse output to count remaining issues
        )


class TOMLFormatter(LanguageFormatter):
    """TOML formatter using tombi."""

    @property
    def name(self) -> str:
        return "TOML"

    @property
    def extensions(self) -> list[str]:
        return [".toml"]

    async def format(self, path: Path) -> FormatResult:
        cmd = ["uv", "run", "tombi", "format", str(path)]
        return_code, stdout, stderr = await self.command_handler(cmd)

        return FormatResult(
            success=return_code == 0,
            output=stdout,
            errors=stderr,
            formatted=return_code == 0,
        )

    async def lint(self, path: Path, fix: bool = False) -> LintResult:
        # tombi lint doesn't seem to have a --fix option, so we ignore the fix parameter
        cmd = ["uv", "run", "tombi", "lint", str(path)]
        return_code, stdout, stderr = await self.command_handler(cmd)

        return LintResult(
            success=return_code == 0,
            output=stdout,
            errors=stderr,
            fixed_issues=0,
            remaining_issues=0,
        )


class TypeScriptFormatter(LanguageFormatter):
    """TypeScript/JavaScript formatter using biome."""

    @property
    def name(self) -> str:
        return "TypeScript"

    @property
    def extensions(self) -> list[str]:
        return [".ts", ".tsx", ".js", ".jsx", ".json"]

    async def format(self, path: Path) -> FormatResult:
        cmd = ["biome", "format", "--write", str(path)]
        return_code, stdout, stderr = await self.command_handler(cmd)

        return FormatResult(
            success=return_code == 0,
            output=stdout,
            errors=stderr,
            formatted=return_code == 0,
        )

    async def lint(self, path: Path, fix: bool = False) -> LintResult:
        cmd = ["biome", "lint"]
        if fix:
            cmd.append("--write")
        cmd.append(str(path))

        return_code, stdout, stderr = await self.command_handler(cmd)

        return LintResult(
            success=return_code == 0,
            output=stdout,
            errors=stderr,
            fixed_issues=0,  # Could parse output to count fixes
            remaining_issues=0,  # Could parse output to count remaining issues
        )


class RustFormatter(LanguageFormatter):
    """Rust formatter using rustfmt and clippy."""

    @property
    def name(self) -> str:
        return "Rust"

    @property
    def extensions(self) -> list[str]:
        return [".rs"]

    async def format(self, path: Path) -> FormatResult:
        cmd = ["rustfmt", str(path)]
        return_code, stdout, stderr = await self.command_handler(cmd)

        return FormatResult(
            success=return_code == 0,
            output=stdout,
            errors=stderr,
            formatted=return_code == 0,
        )

    async def lint(self, path: Path, fix: bool = False) -> LintResult:
        cmd = ["cargo", "clippy"]
        if fix:
            cmd.append("--fix")
        cmd.extend(["--", "--", str(path)])

        return_code, stdout, stderr = await self.command_handler(cmd)

        return LintResult(
            success=return_code == 0,
            output=stdout,
            errors=stderr,
            fixed_issues=0,  # Could parse output to count fixes
            remaining_issues=0,  # Could parse output to count remaining issues
        )


class FormatterRegistry:
    """Registry for language formatters."""

    def __init__(self):
        self.formatters: list[LanguageFormatter] = []

    def register(self, formatter: LanguageFormatter) -> None:
        """Register a formatter."""
        self.formatters.append(formatter)

    def get_formatter(self, path: Path) -> LanguageFormatter | None:
        """Get formatter for given file path."""
        return next((f for f in self.formatters if f.can_handle(path)), None)

    def get_supported_extensions(self) -> list[str]:
        """Get all supported file extensions."""
        return sorted({e for formatter in self.formatters for e in formatter.extensions})


# Example usage and testing
if __name__ == "__main__":

    async def main():
        # Create registry and register formatters
        registry = FormatterRegistry()
        registry.register(PythonFormatter())
        registry.register(TOMLFormatter())
        registry.register(TypeScriptFormatter())
        registry.register(RustFormatter())

        print(f"Supported extensions: {registry.get_supported_extensions()}")

        # Create test files
        test_py = Path("test.py")
        test_toml = Path("test.toml")

        test_py.write_text("def hello( ):\n    print('world')")
        test_toml.write_text('[section]\nkey="value"')

        try:
            # Test using registry
            py_formatter = registry.get_formatter(test_py)
            if py_formatter:
                result = await py_formatter.format_and_lint(test_py, fix=True)
                print(f"Python format success: {result.format_result.success}")
                print(f"Python lint success: {result.lint_result.success}")

            toml_formatter = registry.get_formatter(test_toml)
            if toml_formatter:
                result = await toml_formatter.format_and_lint(test_toml)
                print(f"TOML format success: {result.format_result.success}")
                print(f"TOML lint success: {result.lint_result.success}")

        finally:
            # Cleanup
            test_py.unlink(missing_ok=True)
            test_toml.unlink(missing_ok=True)

    asyncio.run(main())
