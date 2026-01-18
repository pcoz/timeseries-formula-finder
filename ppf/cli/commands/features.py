"""
Features command - Extract ML-ready features from discovery results.

Converts discovered expressions into numerical feature vectors suitable
for machine learning classification or clustering.
"""

import json
import sys
from typing import IO

from ppf import extract_features, get_schema_info, FAMILY_CATEGORIES
from ppf.cli.utils import print_error, print_progress


DESCRIPTION = """\
WHAT IT DOES:
  Extracts numerical features from discovered mathematical expressions
  for use in machine learning pipelines:

  - Structural features: depth, size, operator counts
  - Operator presence: which operations are used
  - Family classification: oscillator, growth, algebraic, etc.
  - Complexity metrics: various complexity measures

  Output can be used for:
  - Clustering similar expressions
  - Training classifiers on expression types
  - Embedding expressions in vector space

HOW TO USE:
  ppf discover data.csv --json | ppf features

  The output is a feature dictionary (or vector with --format vector).
"""

EPILOG = """\
Feature Categories:
  structural    Depth, size, leaf count, etc.
  operators     Presence of each operator type
  family        Expression family indicators
  complexity    Various complexity metrics

Output Formats:
  dict    JSON dictionary with named features (default)
  vector  Flat array of numbers
  csv     CSV row (with header option)

Examples:
  # Extract features as JSON dictionary
  ppf discover data.csv --json | ppf features

  # Get feature vector
  ppf discover data.csv --json | ppf features --format vector

  # Show available feature schema
  ppf features --schema

  # Include residual statistics
  ppf discover data.csv --json | ppf features --include-residuals
"""


def register(subparsers) -> None:
    """Register the features command."""
    parser = subparsers.add_parser(
        "features",
        help="Extract ML-ready features from discovery results",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=lambda prog: __import__("argparse").RawDescriptionHelpFormatter(
            prog, max_help_position=35, width=100
        ),
    )

    # Feature options
    parser.add_argument(
        "--schema",
        action="store_true",
        help="Show feature schema and exit",
    )
    parser.add_argument(
        "--format",
        choices=["dict", "vector", "csv"],
        default="dict",
        help="Output format (default: dict)",
    )
    parser.add_argument(
        "--include-residuals",
        action="store_true",
        help="Include residual statistics in features",
    )
    parser.add_argument(
        "--include-domain-scores",
        action="store_true",
        help="Include domain probe scores if available",
    )

    parser.set_defaults(func=run)


def run(args, output: IO[str]) -> int:
    """Execute the features command."""
    # Handle schema display
    if args.schema:
        _show_schema(output)
        return 0

    # Read input from stdin
    try:
        data = _read_input_json()
    except ValueError as e:
        print_error(str(e))
        return 1

    # Extract features from the best expression
    try:
        features = _extract_from_discover_output(data, args)
    except ValueError as e:
        print_error(str(e))
        return 1

    # Format output
    if args.format == "dict":
        json.dump(features, output, indent=2)
        output.write("\n")
    elif args.format == "vector":
        vector = list(features.values())
        json.dump(vector, output)
        output.write("\n")
    elif args.format == "csv":
        _output_csv(features, output)

    return 0


def _read_input_json() -> dict:
    """Read and parse JSON from stdin."""
    if sys.stdin.isatty():
        raise ValueError(
            "No input provided. Pipe JSON from 'ppf discover --json' or a file."
        )

    content = sys.stdin.read()
    if not content.strip():
        raise ValueError("Empty input received")

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON input: {e}")


def _extract_from_discover_output(data: dict, args) -> dict:
    """Extract features from discover command output."""
    features = {}

    # Check for discover output format
    if "best_tradeoff" not in data:
        raise ValueError(
            "Expected discover output with 'best_tradeoff' key. "
            "Use 'ppf discover --json' to generate input."
        )

    best = data["best_tradeoff"]

    # Basic metrics from the fit result
    features["r_squared"] = best.get("r_squared", 0.0)
    features["mse"] = best.get("mse", 0.0)
    features["complexity"] = best.get("complexity", 0)
    features["depth"] = best.get("depth", 0)
    features["is_noise_like"] = 1 if best.get("is_noise_like", False) else 0

    # Parse expression string for operator features
    expr_str = best.get("expression", "")
    features.update(_parse_expression_features(expr_str))

    # Include residual statistics if requested
    if args.include_residuals and "residuals" in best:
        residuals = best["residuals"]
        if isinstance(residuals, list):
            import numpy as np
            r = np.array(residuals)
            features["residual_mean"] = float(r.mean())
            features["residual_std"] = float(r.std())
            features["residual_max"] = float(r.max())
            features["residual_min"] = float(r.min())

    # Include domain scores if available and requested
    if args.include_domain_scores and "metadata" in data:
        metadata = data["metadata"]
        if "probed_domains" in metadata:
            for domain in metadata["probed_domains"]:
                key = f"domain_{domain.lower()}"
                features[key] = 1

    return features


def _parse_expression_features(expr_str: str) -> dict:
    """Parse expression string for operator presence features."""
    features = {}

    # Operator presence
    operators = {
        "has_sin": "sin(" in expr_str,
        "has_cos": "cos(" in expr_str,
        "has_exp": "exp(" in expr_str,
        "has_log": "log(" in expr_str,
        "has_sqrt": "sqrt(" in expr_str,
        "has_power": "^" in expr_str or "**" in expr_str,
        "has_division": "/" in expr_str,
        "has_multiplication": "*" in expr_str.replace("**", ""),
        "has_addition": "+" in expr_str,
        "has_subtraction": "-" in expr_str,
    }

    for key, value in operators.items():
        features[key] = 1 if value else 0

    # Family classification (heuristic)
    is_oscillator = features["has_sin"] or features["has_cos"]
    is_growth = features["has_exp"] and not is_oscillator
    is_algebraic = not (is_oscillator or is_growth)

    features["family_oscillator"] = 1 if is_oscillator else 0
    features["family_growth"] = 1 if is_growth else 0
    features["family_algebraic"] = 1 if is_algebraic else 0

    # Structural metrics (from string - approximation)
    features["char_length"] = len(expr_str)
    features["paren_depth"] = _max_paren_depth(expr_str)
    features["operator_count"] = sum(1 for c in expr_str if c in "+-*/^")

    return features


def _max_paren_depth(s: str) -> int:
    """Calculate maximum parenthesis nesting depth."""
    depth = 0
    max_depth = 0
    for c in s:
        if c == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif c == ")":
            depth -= 1
    return max_depth


def _output_csv(features: dict, output: IO[str]) -> None:
    """Output features as CSV."""
    keys = list(features.keys())
    values = [str(features[k]) for k in keys]

    output.write(",".join(keys) + "\n")
    output.write(",".join(values) + "\n")


def _show_schema(output: IO[str]) -> None:
    """Display feature schema information."""
    output.write("=" * 60 + "\n")
    output.write("FEATURE SCHEMA\n")
    output.write("=" * 60 + "\n\n")

    output.write("BASIC METRICS:\n")
    output.write("-" * 40 + "\n")
    output.write("  r_squared       Goodness of fit (0-1)\n")
    output.write("  mse             Mean squared error\n")
    output.write("  complexity      Expression complexity (node count)\n")
    output.write("  depth           Expression tree depth\n")
    output.write("  is_noise_like   1 if residuals look like noise\n")
    output.write("\n")

    output.write("OPERATOR PRESENCE:\n")
    output.write("-" * 40 + "\n")
    output.write("  has_sin         Contains sin()\n")
    output.write("  has_cos         Contains cos()\n")
    output.write("  has_exp         Contains exp()\n")
    output.write("  has_log         Contains log()\n")
    output.write("  has_sqrt        Contains sqrt()\n")
    output.write("  has_power       Contains ^ or **\n")
    output.write("  has_division    Contains /\n")
    output.write("  has_multiplication  Contains *\n")
    output.write("  has_addition    Contains +\n")
    output.write("  has_subtraction Contains -\n")
    output.write("\n")

    output.write("FAMILY CLASSIFICATION:\n")
    output.write("-" * 40 + "\n")
    output.write("  family_oscillator  Expression contains sin/cos\n")
    output.write("  family_growth      Expression contains exp (not oscillator)\n")
    output.write("  family_algebraic   Polynomial/algebraic expression\n")
    output.write("\n")

    output.write("STRUCTURAL METRICS:\n")
    output.write("-" * 40 + "\n")
    output.write("  char_length     Length of expression string\n")
    output.write("  paren_depth     Maximum parenthesis nesting\n")
    output.write("  operator_count  Number of operators\n")
    output.write("\n")

    output.write("OPTIONAL (with flags):\n")
    output.write("-" * 40 + "\n")
    output.write("  --include-residuals:\n")
    output.write("    residual_mean, residual_std, residual_max, residual_min\n")
    output.write("  --include-domain-scores:\n")
    output.write("    domain_<name> for each probed domain\n")
