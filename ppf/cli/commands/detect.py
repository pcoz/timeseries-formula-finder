"""
Detect command - Find mathematical forms in data windows.

Scans data for regions that match known mathematical forms (sine, linear,
quadratic, exponential) and validates them through extrapolation.
"""

import json
from typing import IO

from ppf import PPFDetector, FormType
from ppf.cli.utils import (
    add_data_input_args,
    load_data,
    print_error,
    print_progress,
)


DESCRIPTION = """\
WHAT IT DOES:
  Scans your data for windows that match known mathematical forms:
  - Linear (y = ax + b)
  - Quadratic (y = ax² + bx + c)
  - Sinusoidal (y = A*sin(wt + phi) + c)
  - Exponential (y = a*exp(bx) + c)

  Each detected form is validated by testing its extrapolation
  beyond the fitted region. Forms that predict well outside
  their training window are marked as "validated."

HOW TO USE:
  1. Prepare a time series in a CSV file
  2. Run: ppf detect data.csv -x time -y value
  3. Review detected forms and their regions

  Increase --min-r-squared for stricter matching.
  Decrease --min-window if you expect short patterns.
"""

EPILOG = """\
Form Types Detected:
  LINEAR       Straight line trends
  QUADRATIC    Parabolic curves
  SINE         Periodic oscillations
  EXPONENTIAL  Growth/decay patterns
  CONSTANT     Flat regions

Examples:
  # Basic detection with default parameters
  ppf detect sensor_data.csv -x time -y reading

  # Stricter R² threshold
  ppf detect data.csv --min-r-squared 0.9

  # Smaller windows for short patterns
  ppf detect data.csv --min-window 10

  # JSON output for further processing
  ppf detect data.csv --json > detected_forms.json
"""


def register(subparsers) -> None:
    """Register the detect command."""
    parser = subparsers.add_parser(
        "detect",
        help="Detect mathematical forms in data windows",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=lambda prog: __import__("argparse").RawDescriptionHelpFormatter(
            prog, max_help_position=35, width=100
        ),
    )

    # Data input arguments
    add_data_input_args(parser)

    # Detection parameters
    detect_group = parser.add_argument_group("Detection Options")
    detect_group.add_argument(
        "--min-window",
        type=int,
        default=20,
        metavar="N",
        help="Minimum window size for form detection (default: 20)",
    )
    detect_group.add_argument(
        "--min-r-squared",
        type=float,
        default=0.7,
        metavar="R2",
        help="Minimum R² for a valid form fit (default: 0.7)",
    )
    detect_group.add_argument(
        "--extrapolation-window",
        type=int,
        default=10,
        metavar="N",
        help="Points to use for extrapolation validation (default: 10)",
    )
    detect_group.add_argument(
        "--validation-threshold",
        type=float,
        default=0.6,
        metavar="R2",
        help="R² threshold for extrapolation validation (default: 0.6)",
    )

    parser.set_defaults(func=run)


def run(args, output: IO[str]) -> int:
    """Execute the detect command."""
    try:
        # Load data
        print_progress("Loading data...", args)
        x, y = load_data(args)
        print_progress(f"Loaded {len(y)} data points", args)

    except ValueError as e:
        print_error(str(e), "Check file path and column specifications")
        return 1

    # Create detector
    detector = PPFDetector(
        min_window=args.min_window,
        min_r_squared=args.min_r_squared,
        extrapolation_window=args.extrapolation_window,
        validation_threshold=args.validation_threshold,
    )

    # Run detection
    print_progress("Scanning for mathematical forms...", args)
    result = detector.analyze(y)

    n_partial = len(result.partial_forms)
    n_validated = len(result.validated_forms)
    print_progress(f"Found {n_partial} partial forms, {n_validated} validated", args)

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
        "partial_forms": [
            {
                "form_type": pf.fit.form_type.name,
                "start_idx": pf.start_idx,
                "end_idx": pf.end_idx,
                "r_squared": pf.fit.r_squared,
                "params": pf.fit.params.tolist() if hasattr(pf.fit.params, "tolist") else list(pf.fit.params),
                "confidence": pf.confidence,
            }
            for pf in result.partial_forms
        ],
        "validated_forms": [
            {
                "form_type": vf.fit.form_type.name,
                "start_idx": vf.start_idx,
                "end_idx": vf.end_idx,
                "r_squared": vf.fit.r_squared,
                "params": vf.fit.params.tolist() if hasattr(vf.fit.params, "tolist") else list(vf.fit.params),
                "confidence": vf.confidence,
                "extrapolation_r2": getattr(vf, "extrapolation_r2", None),
            }
            for vf in result.validated_forms
        ],
    }
    json.dump(data, output, indent=2)
    output.write("\n")


def _output_human(result, args, output: IO[str]) -> None:
    """Output results in human-readable format."""
    output.write("=" * 60 + "\n")
    output.write("FORM DETECTION RESULTS\n")
    output.write("=" * 60 + "\n\n")

    if not result.partial_forms and not result.validated_forms:
        output.write("No mathematical forms detected.\n")
        output.write("\nTips:\n")
        output.write("  - Try lowering --min-r-squared\n")
        output.write("  - Try reducing --min-window for shorter patterns\n")
        return

    # Validated forms (high confidence)
    if result.validated_forms:
        output.write(f"VALIDATED FORMS ({len(result.validated_forms)}):\n")
        output.write("-" * 60 + "\n")
        for i, vf in enumerate(result.validated_forms, 1):
            output.write(f"\n[{i}] {vf.fit.form_type.name}\n")
            output.write(f"    Region:         [{vf.start_idx}, {vf.end_idx}]\n")
            output.write(f"    Fit R²:         {vf.fit.r_squared:.4f}\n")
            extrap_r2 = getattr(vf, "extrapolation_r2", None)
            if extrap_r2 is not None:
                output.write(f"    Extrapolation:  {extrap_r2:.4f}\n")
            output.write(f"    Confidence:     {vf.confidence:.2f}\n")
            output.write(f"    Parameters:     {_format_params(vf.fit.form_type, vf.fit.params)}\n")
        output.write("\n")

    # Partial forms (detected but not validated)
    partial_only = [pf for pf in result.partial_forms
                    if pf not in result.validated_forms]
    if partial_only:
        output.write(f"PARTIAL FORMS ({len(partial_only)}):\n")
        output.write("-" * 60 + "\n")
        for i, pf in enumerate(partial_only, 1):
            output.write(f"\n[{i}] {pf.fit.form_type.name}\n")
            output.write(f"    Region:     [{pf.start_idx}, {pf.end_idx}]\n")
            output.write(f"    Fit R²:     {pf.fit.r_squared:.4f}\n")
            output.write(f"    Confidence: {pf.confidence:.2f}\n")


def _format_params(form_type: FormType, params) -> str:
    """Format parameters based on form type."""
    param_list = list(params) if hasattr(params, "__iter__") else [params]

    if form_type == FormType.LINEAR:
        if len(param_list) >= 2:
            return f"slope={param_list[0]:.4g}, intercept={param_list[1]:.4g}"
    elif form_type == FormType.QUADRATIC:
        if len(param_list) >= 3:
            return f"a={param_list[0]:.4g}, b={param_list[1]:.4g}, c={param_list[2]:.4g}"
    elif form_type == FormType.SINE:
        if len(param_list) >= 4:
            return f"A={param_list[0]:.4g}, w={param_list[1]:.4g}, phi={param_list[2]:.4g}, c={param_list[3]:.4g}"
    elif form_type == FormType.EXPONENTIAL:
        if len(param_list) >= 3:
            return f"a={param_list[0]:.4g}, b={param_list[1]:.4g}, c={param_list[2]:.4g}"
    elif form_type == FormType.CONSTANT:
        if len(param_list) >= 1:
            return f"value={param_list[0]:.4g}"

    return ", ".join(f"{p:.4g}" for p in param_list)
