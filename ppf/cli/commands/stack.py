"""
Stack command - Extract forms iteratively until residuals are noise.

Implements residual-based form stacking: fits a form, subtracts it,
then fits another to the residuals. Continues until residuals appear
to be pure noise (high entropy).
"""

import json
from typing import IO

from ppf import PPFResidualLayer, EntropyMethod
from ppf.cli.utils import (
    add_data_input_args,
    load_data,
    print_error,
    print_progress,
)


DESCRIPTION = """\
WHAT IT DOES:
  Extracts mathematical forms layer-by-layer from your signal:
  1. Finds the best-fitting form in the data
  2. Subtracts it, leaving residuals
  3. Repeats on residuals until they look like noise

  This "peeling" approach reveals multiple overlapping patterns:
  - A trend line under oscillations
  - A seasonal pattern on top of a polynomial drift
  - Multiple frequency components in a complex signal

  Uses entropy measurement to detect when residuals become noise-like.

HOW TO USE:
  1. Prepare your time series data
  2. Run: ppf stack data.csv
  3. Review the form stack and compression statistics

  The output shows each layer's form and parameters, plus the
  overall compression ratio (how much simpler the forms are
  than the raw data).
"""

EPILOG = """\
Entropy Methods:
  gzip      Compression-based entropy (works well for most signals)
  spectral  Spectral flatness measure (better for audio/vibration)

Examples:
  # Basic stacking with default parameters
  ppf stack sensor_data.csv -x time -y reading

  # Use spectral entropy for audio signals
  ppf stack audio.csv --entropy-method spectral

  # Stricter noise threshold (more layers)
  ppf stack data.csv --noise-threshold 0.95

  # Save residuals to file for inspection
  ppf stack data.csv --save-residuals residuals.csv

  # JSON output showing all layers
  ppf stack data.csv --json
"""


def register(subparsers) -> None:
    """Register the stack command."""
    parser = subparsers.add_parser(
        "stack",
        help="Extract forms iteratively until residuals are noise",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=lambda prog: __import__("argparse").RawDescriptionHelpFormatter(
            prog, max_help_position=35, width=100
        ),
    )

    # Data input arguments
    add_data_input_args(parser)

    # Stacking parameters
    stack_group = parser.add_argument_group("Stacking Options")
    stack_group.add_argument(
        "--entropy-method",
        choices=["gzip", "spectral"],
        default="gzip",
        help="Method for measuring residual entropy (default: gzip)",
    )
    stack_group.add_argument(
        "--noise-threshold",
        type=float,
        default=0.85,
        metavar="T",
        help="Entropy threshold to consider residuals noise (default: 0.85)",
    )
    stack_group.add_argument(
        "--min-compression-gain",
        type=float,
        default=0.05,
        metavar="G",
        help="Minimum compression improvement per layer (default: 0.05)",
    )
    stack_group.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        metavar="N",
        help="Maximum number of form layers to extract (default: 5)",
    )
    stack_group.add_argument(
        "--min-r-squared",
        type=float,
        default=0.5,
        metavar="R2",
        help="Minimum R² for extracted forms (default: 0.5)",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--save-residuals",
        metavar="FILE",
        help="Save final residuals to CSV file",
    )

    parser.set_defaults(func=run)


def run(args, output: IO[str]) -> int:
    """Execute the stack command."""
    try:
        # Load data
        print_progress("Loading data...", args)
        x, y = load_data(args)
        print_progress(f"Loaded {len(y)} data points", args)

    except ValueError as e:
        print_error(str(e), "Check file path and column specifications")
        return 1

    # Parse entropy method
    entropy_method = EntropyMethod[args.entropy_method.upper()]

    # Create residual layer
    layer = PPFResidualLayer(
        entropy_method=entropy_method,
        noise_threshold=args.noise_threshold,
        min_compression_gain=args.min_compression_gain,
        max_iterations=args.max_iterations,
        min_r_squared=args.min_r_squared,
    )

    # Run analysis
    print_progress("Extracting form layers...", args)
    result = layer.analyze(y)
    print_progress(f"Extracted {len(result.form_stack)} layers", args)

    # Save residuals if requested
    if args.save_residuals:
        _save_residuals(x, result.final_residuals, args.save_residuals, args)

    # Format output
    if args.json:
        _output_json(result, args, output)
    else:
        _output_human(result, args, output)

    return 0


def _save_residuals(x, residuals, filename: str, args) -> None:
    """Save residuals to a CSV file."""
    import numpy as np

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("x,residual\n")
            for xi, ri in zip(x, residuals):
                f.write(f"{xi},{ri}\n")
        print_progress(f"Saved residuals to {filename}", args)
    except IOError as e:
        print_error(f"Cannot save residuals: {e}")


def _output_json(result, args, output: IO[str]) -> None:
    """Output results as JSON."""
    data = {
        "status": "success",
        "num_layers": len(result.form_stack),
        "compression_ratio": result.compression_ratio,
        "space_savings": result.space_savings,
        "total_params": result.total_params,
        "form_stack": [
            {
                "layer": i + 1,
                "form_type": layer.form_type.name,
                "params": layer.params.tolist() if hasattr(layer.params, "tolist") else list(layer.params),
                "r_squared": layer.r_squared,
                "residual_entropy": layer.residual_entropy,
                "compression_gain": layer.compression_gain,
            }
            for i, layer in enumerate(result.form_stack)
        ],
        "final_residuals_stats": {
            "mean": float(result.final_residuals.mean()),
            "std": float(result.final_residuals.std()),
            "min": float(result.final_residuals.min()),
            "max": float(result.final_residuals.max()),
        },
    }
    json.dump(data, output, indent=2)
    output.write("\n")


def _output_human(result, args, output: IO[str]) -> None:
    """Output results in human-readable format."""
    output.write("=" * 60 + "\n")
    output.write("FORM STACK RESULTS\n")
    output.write("=" * 60 + "\n\n")

    if not result.form_stack:
        output.write("No forms extracted - data appears to be noise.\n")
        output.write("\nTips:\n")
        output.write("  - Try lowering --noise-threshold\n")
        output.write("  - Try lowering --min-r-squared\n")
        return

    # Summary statistics
    output.write("SUMMARY:\n")
    output.write(f"  Layers extracted:   {len(result.form_stack)}\n")
    output.write(f"  Compression ratio:  {result.compression_ratio:.2f}x\n")
    output.write(f"  Space savings:      {result.space_savings * 100:.1f}%\n")
    output.write(f"  Total parameters:   {result.total_params}\n")
    output.write("\n")

    # Form stack details
    output.write("FORM STACK:\n")
    output.write("-" * 60 + "\n")

    for i, layer in enumerate(result.form_stack, 1):
        output.write(f"\nLayer {i}: {layer.form_type.name}\n")
        output.write(f"  R²:               {layer.r_squared:.4f}\n")
        output.write(f"  Residual entropy: {layer.residual_entropy:.4f}\n")
        output.write(f"  Compression gain: {layer.compression_gain:.4f}\n")
        output.write(f"  Parameters:       {_format_params(layer)}\n")

    # Residual statistics
    output.write("\n" + "-" * 60 + "\n")
    output.write("FINAL RESIDUALS:\n")
    output.write(f"  Mean:  {result.final_residuals.mean():.6g}\n")
    output.write(f"  Std:   {result.final_residuals.std():.6g}\n")
    output.write(f"  Range: [{result.final_residuals.min():.6g}, {result.final_residuals.max():.6g}]\n")


def _format_params(layer) -> str:
    """Format layer parameters."""
    params = list(layer.params) if hasattr(layer.params, "__iter__") else [layer.params]
    return ", ".join(f"{p:.6g}" for p in params)
