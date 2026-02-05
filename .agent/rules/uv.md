---
trigger: always_on
---

# Python Development with `uv`

When working with Python in this project, **always** use `uv` (Astral's Python package manager) for environment management, dependency handling, and script execution.

## Core Commands

### 1. Package Management
- **Add dependencies**: `uv add <package>`
- **Add dev dependencies**: `uv add --dev <package>`
- **Remove dependencies**: `uv remove <package>`
- **Sync environment**: `uv sync` (ensures `.venv` matches `pyproject.toml` and `uv.lock`)
- **Upgrade dependency**: `uv lock --upgrade-package <package>`

### 2. Virtual Environments
- `uv` automatically manages the `.venv` directory.
- You do **not** need to manually activate the environment if you use `uv run`.
- If a `.venv` is missing, run `uv venv` or simply `uv sync`.

### 3. Script & Test Execution
- **Run a Python script**: `uv run <script.py>`
- **Run a command in the environment**: `uv run <command>` (e.g., `uv run ruff check .`)
- **Run tests**: `uv run pytest` or `uv run pytest <test_file.py>`
- **Run a console script**: If a package provides a command (like `black`), use `uv run black .`

## Guidelines for AI Agent
1. **Always prefix Python commands with `uv run`**.
2. **Consult `pyproject.toml` and `uv.lock`** as the source of truth for dependencies.
3. **Never use `pip`** directly unless `uv` is unavailable (which shouldn't happen here).
4. **When adding new packages**, use `uv add` to ensure `pyproject.toml` and `uv.lock` are updated in sync.
5. **If tests fail due to missing dependencies**, check if they need to be added via `uv add --dev`.