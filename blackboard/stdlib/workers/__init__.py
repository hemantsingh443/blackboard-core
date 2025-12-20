"""
Standard Library Workers

Pre-built workers for common tasks.
"""

from blackboard.stdlib.workers.search import WebSearchWorker
from blackboard.stdlib.workers.browser import BrowserWorker
from blackboard.stdlib.workers.code import CodeInterpreterWorker
from blackboard.stdlib.workers.human import HumanProxyWorker

__all__ = [
    "WebSearchWorker",
    "BrowserWorker",
    "CodeInterpreterWorker",
    "HumanProxyWorker",
]
