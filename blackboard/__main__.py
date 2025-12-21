"""
Blackboard CLI Entry Point

Enables execution via 'python -m blackboard'.
"""

import sys
from blackboard.cli import main

if __name__ == "__main__":
    sys.exit(main())
