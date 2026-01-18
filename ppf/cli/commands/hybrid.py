"""
Hybrid command - Combine EMD/SSA decomposition with form interpretation.

Uses Empirical Mode Decomposition or Singular Spectrum Analysis to
separate signal components, then interprets each component with PPF.
"""

import json
import sys
from typing import IO

from ppf.cli.utils import (
    add_data_input_args,
    load_data,
    print_error,
    print_progress,
    print_warning,
)


DESCRIPTION = """\
WHAT IT DOES:
  Combines signal decomposition methods with form interpretation:

  1. DECOMPOSITION: Separates your signal into components using:
     - EMD:     Empirical Mode Decomposition
     - EEMD:    Ensemble EMD (more robust to noise)
     - CEEMDAN: Complete EEMD with Adaptive Noise
     - SSA:     Singular Spectrum Analysis

  2. INTERPRETATION: Analyzes each component to identify:
     - Its mathematical form (sine, linear, exponential, etc.)
     - Whether it's signal or noise
     - Its contribution to total variance

  This is powerful for complex signals with multiple sources.

HOW TO USE:
  1. Install EMD support: pip install timeseries-formula-finder[hybrid]
  2. Run: ppf hybrid data.csv --method eemd
  3. Review the interpreted components

  SSA doesn't require EMD-signal and works out of the box.
"""

EPILOG = """\
Decomposition Methods:
  emd       Basic EMD - fast but sensitive to noise
  eemd      Ensemble EMD - adds noise, averages results (recommended)
  ceemdan   Complete EEMD - most robust, slowest
  ssa       Singular Spectrum Analysis - algebraic, no EMD needed

SSA-specific Options:
  --ssa-window     Window length for SSA (default: N/4)
  --ssa-components Number of components to extract (default: auto)

EMD-specific Options:
  --max-imfs       Maximum IMFs to extract (default: auto)
  --noise-width    Noise amplitude for EEMD/CEEMDAN (default: 0.2)
  --ensemble-size  Ensemble size for EEMD/CEEMDAN (default: 100)

Examples:
  # Basic EEMD decomposition
  ppf hybrid sensor.csv --method eemd

  # SSA analysis (no extra dependencies)
  ppf hybrid data.csv --method ssa --ssa-window 50

  # Strict noise threshold
  ppf hybrid data.csv --method eemd --noise-threshold 0.7

  # JSON output for further processing
  ppf hybrid data.csv --method ssa --json
"""


def register(subparsers) -> None:
    """Register the hybrid command."""
    parser = subparsers.add_parser(
        "hybrid",
        help="Combine EMD/SSA decomposition with form interpretation",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=lambda prog: __import__("argparse").RawDescriptionHelpFormatter(
            prog, max_help_position=35, width=100
        ),
    )

    # Data input arguments
    add_data_input_args(parser)

    # Method selection
    method_group = parser.add_argument_group("Decomposition Method")
    method_group.add_argument(
        "--method",
        choices=["emd", "eemd", "ceemdan", "ssa"],
        default="eemd",
        help="Decomposition method (default: eemd)",
    )

    # Common parameters
    common_group = parser.add_argument_group("Analysis Options")
    common_group.add_argument(
        "--noise-threshold",
        type=float,
        default=0.5,
        metavar="T",
        help="Entropy threshold for noise classification (default: 0.5)",
    )
    common_group.add_argument(
        "--min-r-squared",
        type=float,
        default=0.3,
        metavar="R2",
        help="Minimum R² for form detection (default: 0.3)",
    )
    common_group.add_argument(
        "--min-variance",
        type=float,
        default=0.01,
        metavar="V",
        help="Minimum variance contribution to include (default: 0.01)",
    )

    # EMD-specific parameters
    emd_group = parser.add_argument_group("EMD Options (emd/eemd/ceemdan)")
    emd_group.add_argument(
        "--max-imfs",
        type=int,
        default=None,
        metavar="N",
        help="Maximum IMFs to extract (default: auto)",
    )
    emd_group.add_argument(
        "--noise-width",
        type=float,
        default=0.2,
        metavar="W",
        help="Noise amplitude for EEMD/CEEMDAN (default: 0.2)",
    )
    emd_group.add_argument(
        "--ensemble-size",
        type=int,
        default=100,
        metavar="N",
        help="Ensemble size for EEMD/CEEMDAN (default: 100)",
    )

    # SSA-specific parameters
    ssa_group = parser.add_argument_group("SSA Options")
    ssa_group.add_argument(
        "--ssa-window",
        type=int,
        default=None,
        metavar="N",
        help="Window length for SSA (default: N/4)",
    )
    ssa_group.add_argument(
        "--ssa-components",
        type=int,
        default=None,
        metavar="N",
        help="Number of SSA components (default: auto)",
    )

    parser.set_defaults(func=run)


def run(args, output: IO[str]) -> int:
    """Execute the hybrid command."""
    # Check for EMD availability if needed
    method = args.method.lower()
    if method in ("emd", "eemd", "ceemdan"):
        try:
            import PyEMD  # noqa: F401
        except ImportError:
            print_error(
                "EMD methods require the EMD-signal package",
                "Install with: pip install timeseries-formula-finder[hybrid]",
            )
            return 1

    try:
        # Load data
        print_progress("Loading data...", args)
        x, y = load_data(args)
        print_progress(f"Loaded {len(y)} data points", args)

    except ValueError as e:
        print_error(str(e), "Check file path and column specifications")
        return 1

    # Import here to avoid issues if EMD not installed
    from ppf import HybridDecomposer

    # Create decomposer with method-specific parameters
    decomposer_kwargs = {
        "method": method,
        "noise_threshold": args.noise_threshold,
        "min_r_squared": args.min_r_squared,
        "min_variance_contribution": args.min_variance,
    }

    # Add EMD-specific parameters
    if method in ("emd", "eemd", "ceemdan"):
        if args.max_imfs:
            decomposer_kwargs["max_imfs"] = args.max_imfs
        if method in ("eemd", "ceemdan"):
            decomposer_kwargs["noise_width"] = args.noise_width
            decomposer_kwargs["ensemble_size"] = args.ensemble_size

    # Add SSA-specific parameters
    if method == "ssa":
        if args.ssa_window:
            decomposer_kwargs["ssa_window"] = args.ssa_window
        if args.ssa_components:
            decomposer_kwargs["ssa_components"] = args.ssa_components

    decomposer = HybridDecomposer(**decomposer_kwargs)

    # Run decomposition
    print_progress(f"Running {method.upper()} decomposition...", args)
    result = decomposer.analyze(y)

    n_signal = len(result.get_signal_components())
    n_noise = len(result.get_noise_components())
    print_progress(f"Found {n_signal} signal components, {n_noise} noise components", args)

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
        "method": args.method.upper(),
        "total_components": len(result.components),
        "signal_components": len(result.get_signal_components()),
        "noise_components": len(result.get_noise_components()),
        "components": [
            {
                "index": i,
                "form_type": comp.form_type.name if comp.form_type else None,
                "is_noise": comp.is_noise,
                "r_squared": comp.r_squared,
                "variance_contribution": comp.variance_contribution,
                "entropy": comp.entropy,
                "interpretation": comp.interpretation,
                "period": comp.period,
                "amplitude": comp.amplitude,
                "params": comp.form_params.tolist() if comp.form_params is not None and hasattr(comp.form_params, "tolist") else (list(comp.form_params) if comp.form_params is not None else None),
            }
            for i, comp in enumerate(result.components)
        ],
    }
    json.dump(data, output, indent=2)
    output.write("\n")


def _output_human(result, args, output: IO[str]) -> None:
    """Output results in human-readable format."""
    output.write("=" * 60 + "\n")
    output.write(f"HYBRID DECOMPOSITION ({args.method.upper()})\n")
    output.write("=" * 60 + "\n\n")

    # Use built-in summary if available
    if hasattr(result, "summary"):
        output.write(result.summary())
        return

    # Manual formatting
    signal_comps = result.get_signal_components()
    noise_comps = result.get_noise_components()

    output.write(f"Total components: {len(result.components)}\n")
    output.write(f"Signal components: {len(signal_comps)}\n")
    output.write(f"Noise components: {len(noise_comps)}\n\n")

    if signal_comps:
        output.write("SIGNAL COMPONENTS:\n")
        output.write("-" * 60 + "\n")
        for i, comp in enumerate(signal_comps):
            output.write(f"\n[{i + 1}] {comp.form_type.name if comp.form_type else 'Unknown'}\n")
            output.write(f"    {comp.interpretation}\n")
            output.write(f"    R²:       {comp.r_squared:.4f}\n")
            output.write(f"    Variance: {comp.variance_contribution * 100:.1f}%\n")
            if comp.period:
                output.write(f"    Period:   {comp.period:.2f}\n")
            if comp.amplitude:
                output.write(f"    Amplitude: {comp.amplitude:.4f}\n")

    if noise_comps:
        output.write("\nNOISE COMPONENTS:\n")
        output.write("-" * 60 + "\n")
        for i, comp in enumerate(noise_comps):
            output.write(f"\n[{i + 1}] Noise (entropy: {comp.entropy:.3f})\n")
            output.write(f"    Variance: {comp.variance_contribution * 100:.1f}%\n")
