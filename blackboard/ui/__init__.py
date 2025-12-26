"""
Blackboard UI Module

This module provides user interfaces for Blackboard:

- BlackboardApp: Interactive terminal TUI (Textual-based)
- create_tui: Factory function for the TUI
- is_headless: Check if running in headless/CI mode

For the Streamlit web dashboard, use:
    from blackboard.ui.app import main as run_streamlit_ui
"""

# Textual TUI (interactive terminal)
try:
    from .textual_app import (
        BlackboardApp,
        create_tui,
        is_headless,
    )
except ImportError:
    # Textual not installed
    BlackboardApp = None  # type: ignore
    create_tui = None  # type: ignore
    is_headless = None  # type: ignore

__all__ = [
    "BlackboardApp",
    "create_tui", 
    "is_headless",
]
