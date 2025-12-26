"""
Terminal UI for Blackboard Visualization (Legacy - Deprecated)

This module provides backwards compatibility for the original rich-based TUI.
For the new interactive Textual TUI, use:

    from blackboard.ui import BlackboardApp, create_tui

Warning:
    This module is deprecated and will be removed in v2.0.
    Please migrate to blackboard.ui for the new interactive TUI.
"""

import warnings

# Issue deprecation warning on import
warnings.warn(
    "blackboard.tui is deprecated. Use blackboard.ui for the new Textual TUI, "
    "or blackboard.tui_legacy for the original rich-based viewer.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from legacy module for backwards compatibility
from .tui_legacy import *
