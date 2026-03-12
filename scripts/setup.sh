#!/usr/bin/env bash
# Setups the development environment.

# Stop on errors
set -e

cd "$(dirname "$0")/.."

echo "Installing development and optional dependencies..."
uv sync --group dev --extra numpy

echo "Installing pre-commit hooks..."
uv run pre-commit install
