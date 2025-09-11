#!/usr/bin/env python3
"""Database migration management script for llmling-agent.

This script provides easy commands for managing database migrations using Alembic.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def run_command(cmd: list[str], cwd: Path | None = None) -> int:
    """Run a command and return the exit code."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if result.returncode != 0 and cmd[0] == "alembic":
        print(f"Error: Command '{cmd[0]}' not found. Make sure alembic is installed.")
    return result.returncode


def get_project_root() -> Path:
    """Get the project root directory."""
    # Start from this file and go up to find the project root
    # This file is in src/llmling_agent_storage/sql_provider/cli.py
    return Path(__file__).parent.parent.parent.parent


def main():  # noqa: PLR0911
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Database migration management for llmling-agent"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Status command
    subparsers.add_parser("status", help="Show current migration status")

    # Upgrade command
    upgrade_parser = subparsers.add_parser(
        "upgrade", help="Upgrade database to latest migration"
    )
    upgrade_parser.add_argument(
        "revision", nargs="?", default="head", help="Target revision (default: head)"
    )

    # Downgrade command
    downgrade_parser = subparsers.add_parser("downgrade", help="Downgrade database")
    downgrade_parser.add_argument(
        "revision",
        help="Target revision (e.g., -1 for previous, or specific revision ID)",
    )

    # Create migration command
    create_parser = subparsers.add_parser("create", help="Create a new migration")
    create_parser.add_argument("message", help="Migration message")
    create_parser.add_argument(
        "--autogenerate",
        action="store_true",
        help="Auto-generate migration based on model changes",
    )

    # History command
    subparsers.add_parser("history", help="Show migration history")

    # Current command
    subparsers.add_parser("current", help="Show current revision")

    # Reset command
    reset_parser = subparsers.add_parser(
        "reset", help="Reset database (WARNING: destructive)"
    )
    reset_parser.add_argument(
        "--force", action="store_true", help="Force reset without confirmation"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    project_root = get_project_root()

    # Set up environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")

    # Handle commands
    if args.command == "status":
        return run_command(["alembic", "current", "-v"], cwd=project_root)

    if args.command == "upgrade":
        return run_command(["alembic", "upgrade", args.revision], cwd=project_root)

    if args.command == "downgrade":
        return run_command(["alembic", "downgrade", args.revision], cwd=project_root)

    if args.command == "history":
        return run_command(["alembic", "history", "-v"], cwd=project_root)

    if args.command == "current":
        return run_command(["alembic", "current"], cwd=project_root)

    if args.command == "create":
        cmd = ["alembic", "revision", "-m", args.message]
        if args.autogenerate:
            cmd.append("--autogenerate")
        return run_command(cmd, cwd=project_root)

    if args.command == "reset":
        if not args.force:
            response = input(
                "This will delete all data in the database. Continue? (y/N): "
            )
            if response.lower() != "y":
                print("Aborted.")
                return 0

        # Drop all tables and recreate
        print("Resetting database...")

        # First, drop to base (empty database)
        result = run_command(["alembic", "downgrade", "base"], cwd=project_root)
        if result != 0:
            return result

        # Then upgrade to head
        return run_command(["alembic", "upgrade", "head"], cwd=project_root)

    return 0


if __name__ == "__main__":
    sys.exit(main())
