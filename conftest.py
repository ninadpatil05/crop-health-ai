"""
conftest.py — project-level pytest configuration.
Adds the project root to sys.path so that `src.*` imports resolve correctly.
"""
import sys
from pathlib import Path

# Ensure the project root (the dir containing this file) is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
