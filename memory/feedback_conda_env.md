---
name: Use conda lerobot environment
description: All Python commands must be run using the conda environment named "lerobot"
type: feedback
---

Always use `conda run -n lerobot python` (or `conda run -n lerobot ...`) instead of plain `python` when running any Python scripts or inspecting the lerobot codebase.

**Why:** The project dependencies (lerobot package, torch, etc.) are installed in the conda environment named "lerobot", not the system Python.

**How to apply:** Replace `python` with `conda run -n lerobot python` for all Bash tool calls involving Python in this project.
