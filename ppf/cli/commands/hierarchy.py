"""
Hierarchy command - Find nested patterns at multiple timescales.

Analyzes data at multiple levels: fits forms in windows, then
looks for patterns in how form parameters evolve over time.
"""

import json
from typing import IO

from ppf import HierarchicalDetector, FormType, EntropyMethod
from ppf.cli.utils import (
    add_data_input_args,
    load_data,
    print_error,
    print_progress,
)


DESCRIPTION = """\
WHAT IT DOES:
  Discovers nested patterns at multiple timescales:

  Level 0: Fits forms to windows of your data
           (e.g., finds sine waves in each window)

  Level 1: Looks for patterns in how Level 0 parameters evolve
           (e.g., the amplitude of the sine wave grows linearly)

  Level 2+: Continues if meta-patterns are found

  This reveals hierarchical structure like:
  - "The frequency of oscillations increases quadratically over time"
  - "The trend slope follows a seasonal pattern"

HOW TO USE:
  1. Prepare time series data with enough points for multiple windows
  2. Run: ppf hierarchy data.csv --window-size 50
  3. Review the hierarchy of detected patterns

  The window size determines the base timescale. Smaller windows
  find faster patterns but need more data points.
"""

EPILOG = """\
Examples:
  # Basic hierarchical analysis
  ppf hierarchy vibration.csv --window-size 100

  # Focus on sine patterns
  ppf hierarchy data.csv --preferred-form sine --window-size 50

  # Analyze with overlapping windows
  ppf hierarchy data.csv --window-size 100 --window-overlap 0.5

  # Deeper hierarchy search
  ppf hierarchy data.csv --max-levels 5

  # JSON output for further analysis
  ppf hierarchy data.csv --json

Preferred Forms:
  constant    Look for constant regions
  linear      Look for linear trends
  quadratic   Look for parabolic patterns
  sine        Look for oscillations (default)
  exponential Look for growth/decay
"""


def register(subparsers) -> None:
    """Register the hierarchy command."""
    parser = subparsers.add_parser(
        "hierarchy",
        help="Find nested patterns at multiple timescales",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=lambda prog: __import__("argparse").RawDescriptionHelpFormatter(
            prog, max_help_position=35, width=100
        ),
    )

    # Data input arguments
    add_data_input_args(parser)

    # Hierarchy parameters
    hier_group = parser.add_argument_group("Hierarchy Options")
    hier_group.add_argument(
        "--window-size",
        type=int,
        default=None,
        metavar="N",
        help="Window size for base-level analysis (default: auto)",
    )
    hier_group.add_argument(
        "--window-overlap",
        type=float,
        default=0.0,
        metavar="FRAC",
        help="Fraction of window overlap, 0-0.9 (default: 0)",
    )
    hier_group.add_argument(
        "--min-r-squared",
        type=float,
        default=0.3,
        metavar="R2",
        help="Minimum R² for valid fits (default: 0.3)",
    )
    hier_group.add_argument(
        "--preferred-form",
        choices=["constant", "linear", "quadratic", "sine", "exponential"],
        default=None,
        help="Preferred form type to search for",
    )
    hier_group.add_argument(
        "--max-levels",
        type=int,
        default=3,
        metavar="N",
        help="Maximum hierarchy levels to analyze (default: 3)",
    )
    hier_group.add_argument(
        "--entropy-method",
        choices=["gzip", "spectral"],
        default="spectral",
        help="Entropy method for noise detection (default: spectral)",
    )

    parser.set_defaults(func=run)


def run(args, output: IO[str]) -> int:
    """Execute the hierarchy command."""
    try:
        # Load data
        print_progress("Loading data...", args)
        x, y = load_data(args)
        print_progress(f"Loaded {len(y)} data points", args)

    except ValueError as e:
        print_error(str(e), "Check file path and column specifications")
        return 1

    # Parse optional parameters
    preferred_form = None
    if args.preferred_form:
        preferred_form = FormType[args.preferred_form.upper()]

    entropy_method = EntropyMethod[args.entropy_method.upper()]

    # Create detector
    detector = HierarchicalDetector(
        window_size=args.window_size,
        window_overlap=args.window_overlap,
        min_r_squared=args.min_r_squared,
        preferred_form=preferred_form,
        max_levels=args.max_levels,
        entropy_method=entropy_method,
    )

    # Run analysis
    print_progress("Analyzing hierarchical patterns...", args)
    result = detector.analyze(y)
    print_progress(f"Found {len(result.levels)} hierarchy levels", args)

    # Format output
    if args.json:
        _output_json(result, args, output)
    else:
        _output_human(result, args, output)

    return 0


def _output_json(result, args, output: IO[str]) -> None:
    """Output results as JSON."""
    data = {
        "status": "success",
        "num_levels": len(result.levels),
        "levels": [],
    }

    for i, level in enumerate(result.levels):
        level_data = {
            "level": i,
            "dominant_form": level.dominant_form.name if level.dominant_form else None,
            "coverage": level.coverage,
            "window_fits": [
                {
                    "window_idx": wf.window_idx,
                    "form_type": wf.form_type.name if wf.form_type else None,
                    "r_squared": wf.r_squared,
                    "params": wf.params.tolist() if wf.params is not None and hasattr(wf.params, "tolist") else (list(wf.params) if wf.params is not None else None),
                }
                for wf in level.window_fits
            ],
            "parameter_evolutions": [
                {
                    "param_name": pe.param_name,
                    "has_structure": pe.has_structure,
                    "form_type": pe.form_type.name if pe.form_type else None,
                    "variance_explained": pe.variance_explained,
                }
                for pe in level.parameter_evolutions
            ],
        }
        data["levels"].append(level_data)

    json.dump(data, output, indent=2)
    output.write("\n")


def _output_human(result, args, output: IO[str]) -> None:
    """Output results in human-readable format."""
    output.write("=" * 60 + "\n")
    output.write("HIERARCHICAL PATTERN ANALYSIS\n")
    output.write("=" * 60 + "\n\n")

    if not result.levels:
        output.write("No hierarchical patterns found.\n")
        output.write("\nTips:\n")
        output.write("  - Try adjusting --window-size\n")
        output.write("  - Try lowering --min-r-squared\n")
        return

    # Use built-in summary if available
    if hasattr(result, "summary"):
        output.write(result.summary())
        return

    # Manual formatting
    for i, level in enumerate(result.levels):
        output.write(f"LEVEL {i}:\n")
        output.write("-" * 60 + "\n")

        if level.dominant_form:
            output.write(f"  Dominant form: {level.dominant_form.name}\n")
        output.write(f"  Coverage:      {level.coverage * 100:.1f}%\n")
        output.write(f"  Windows:       {len(level.window_fits)}\n")

        # Show parameter evolutions
        if level.parameter_evolutions:
            output.write("\n  Parameter Evolutions:\n")
            for pe in level.parameter_evolutions:
                structure = "has structure" if pe.has_structure else "no clear pattern"
                form_info = f" ({pe.form_type.name})" if pe.form_type else ""
                var_info = f", {pe.variance_explained * 100:.1f}% explained" if pe.variance_explained else ""
                output.write(f"    {pe.param_name}: {structure}{form_info}{var_info}\n")

        output.write("\n")
