#!/usr/bin/env bash
# Setups the development environment.

# Stop on errors
set -e

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: 'uv' is not installed or not on PATH." >&2
    echo "Install it from https://github.com/astral-sh/uv and rerun this script." >&2
    exit 1
fi

echo "Installing development dependencies..."
uv sync --group dev --all-extras

echo "Installing pre-commit hooks..."
# Clear stale local core.hooksPath (e.g. from bind-mounted .git/config); pre-commit refuses install otherwise.
git config --local --unset-all core.hooksPath 2>/dev/null || true
uv run pre-commit install
