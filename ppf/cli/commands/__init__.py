"""
CLI command modules.

Each module implements a single command with:
- register(subparsers): Register the command's argument parser
- run(args, output): Execute the command
"""

from . import discover
from . import detect
from . import stack
from . import hierarchy
from . import hybrid
from . import export
from . import features
from . import info

__all__ = [
    "discover",
    "detect",
    "stack",
    "hierarchy",
    "hybrid",
    "export",
    "features",
    "info",
]
