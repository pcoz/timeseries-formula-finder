"""
Main CLI dispatcher and argparse setup for PPF.

This module sets up the command-line interface with subcommands for:
- discover: Symbolic regression to find formulas
- detect: Find mathematical forms in data windows
- stack: Extract forms iteratively until residuals are noise
- hierarchy: Find nested patterns at multiple timescales
- hybrid: Combine EMD/SSA decomposition with form interpretation
- export: Export expressions to Python/C/JSON
- features: Extract ML-ready features
- info: Show available modes, forms, and macros
"""

import argparse
import sys
from typing import List, Optional

import ppf


class CustomHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Custom formatter that preserves description formatting and adds spacing."""

    def __init__(self, prog, indent_increment=2, max_help_position=30, width=100):
        super().__init__(prog, indent_increment, max_help_position, width)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with all subcommands."""

    # Main parser
    parser = argparse.ArgumentParser(
        prog="ppf",
        description=(
            "PPF - Symbolic Form Discovery\n"
            "================================\n\n"
            "Find interpretable mathematical formulas in your data using\n"
            "symbolic regression, form detection, and signal decomposition.\n\n"
            "Quick Start:\n"
            "  ppf discover data.csv -x time -y signal    # Find formulas\n"
            "  ppf detect data.csv --min-r-squared 0.8    # Detect forms\n"
            "  ppf info modes                              # List discovery modes"
        ),
        epilog=(
            "Examples:\n"
            "  # Discover formulas with genetic programming\n"
            "  ppf discover data.csv -x time -y signal --mode auto -v\n\n"
            "  # Detect forms in windows and extract iteratively\n"
            "  ppf stack data.csv --noise-threshold 0.9\n\n"
            "  # Export discovered expression to Python\n"
            "  ppf discover data.csv --json | ppf export python -f predict\n\n"
            "  # Analyze signal with EMD decomposition\n"
            "  ppf hybrid data.csv --method eemd\n\n"
            "For more information on a command, use: ppf <command> --help\n"
            "Documentation: https://github.com/pcoz/timeseries-formula-finder"
        ),
        formatter_class=CustomHelpFormatter,
    )

    # Global options
    parser.add_argument(
        "--version",
        action="version",
        version=f"ppf {ppf.__version__}",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed progress and debug information",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all output except errors and final result",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format (machine-readable)",
    )
    parser.add_argument(
        "-o", "--output",
        metavar="FILE",
        help="Write output to FILE instead of stdout",
    )

    # Create subparsers
    subparsers = parser.add_subparsers(
        title="Commands",
        dest="command",
        metavar="<command>",
    )

    # Import and register all command modules
    from ppf.cli.commands import (
        discover,
        detect,
        stack,
        hierarchy,
        hybrid,
        export,
        features,
        info,
    )

    # Register each command
    discover.register(subparsers)
    detect.register(subparsers)
    stack.register(subparsers)
    hierarchy.register(subparsers)
    hybrid.register(subparsers)
    export.register(subparsers)
    features.register(subparsers)
    info.register(subparsers)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for the PPF CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Show help if no command given
    if args.command is None:
        parser.print_help()
        return 0

    # Set up output file if specified
    output_file = None
    if args.output:
        try:
            output_file = open(args.output, "w", encoding="utf-8")
        except IOError as e:
            print(f"Error: Cannot open output file '{args.output}': {e}", file=sys.stderr)
            return 1

    try:
        # Each command module has a 'run' function
        exit_code = args.func(args, output_file or sys.stdout)
        return exit_code

    except KeyboardInterrupt:
        if not args.quiet:
            print("\nInterrupted by user", file=sys.stderr)
        return 130

    except BrokenPipeError:
        # Handle pipe closure gracefully (e.g., head, less)
        return 0

    except Exception as e:
        if args.verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1

    finally:
        if output_file:
            output_file.close()


if __name__ == "__main__":
    sys.exit(main())
