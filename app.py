"""
Root-level FastAPI entrypoint
Imports the app from backend/main.py for discovery by FastAPI tools
"""

import sys
from pathlib import Path

# Add backend directory to path so imports work
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from main import app

__all__ = ["app"]
