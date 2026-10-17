"""
tests/conftest.py

Shared pytest fixtures for all tests.
Adds the project src/ directory to sys.path so modules import cleanly
regardless of whether the package is installed.
"""

import sys
from pathlib import Path

# ── Add src/ to the import path ───────────────────────────────────────────────
_project_root = Path(__file__).resolve().parent.parent
_src = str(_project_root / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
