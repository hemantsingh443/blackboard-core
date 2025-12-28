"""
Blackboard Code - Multi-Agent IDE Demo

Configuration and settings.
"""

import os
from pathlib import Path

# LLM Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = "xiaomi/mimo-v2-flash:free"

# Project sandbox
SANDBOX_DIR = Path(os.environ.get("BLACKBOARD_CODE_SANDBOX", "")).resolve() if os.environ.get("BLACKBOARD_CODE_SANDBOX") else None

# Workers that require human approval before execution
REQUIRE_APPROVAL = ["FileWorker", "ExecutorWorker"]

# Max tokens per LLM call
MAX_TOKENS = 4000

# OpenRouter headers
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/hemantsingh443/blackboard-core",
    "X-Title": "Blackboard Code"
}
