"""
Blackboard Standard Library

Pre-built workers for common tasks.
"""

from blackboard.stdlib.workers import (
    WebSearchWorker,
    BrowserWorker,
    CodeInterpreterWorker,
    HumanProxyWorker,
)

__all__ = [
    "WebSearchWorker",
    "BrowserWorker",
    "CodeInterpreterWorker",
    "HumanProxyWorker",
]
