"""
Discover command - Symbolic regression to find mathematical formulas.

Uses genetic programming to search the space of mathematical expressions
and find formulas that fit your data. Returns a Pareto front of solutions
balancing accuracy against complexity.
"""

import json
import sys
from typing import IO

from ppf import (
    SymbolicRegressor,
    DiscoveryMode,
    simplify_expression,
    format_expression_latex,
)
from ppf.cli.utils import (
    add_data_input_args,
    load_data,
    print_error,
    print_progress,
)


DESCRIPTION = """\
WHAT IT DOES:
  Searches the space of mathematical expressions to find formulas that
  fit your data. Uses multi-objective genetic programming to balance
  accuracy against complexity, returning a Pareto front of solutions.

HOW TO USE:
  1. Prepare a CSV file with x (independent) and y (dependent) columns
  2. Run: ppf discover data.csv -x time -y signal
  3. Review the discovered expressions and choose the best fit

  The output shows three key solutions:
  - Most accurate: Best fit regardless of complexity
  - Most parsimonious: Simplest expression with decent fit
  - Best tradeoff: Balanced solution at the Pareto knee
"""

EPILOG = """\
Discovery Modes:
  auto        Automatic domain probing (recommended for unknown data)
  identify    Template-based matching (fast, uses macros)
  discover    Pure discovery (no templates, exploratory)
  oscillator  Damped oscillations, vibrations
  circuit     RC/RLC circuits, charge/discharge
  growth      Population dynamics, saturation curves
  rational    Rational functions (polynomial ratios)
  polynomial  Pure algebraic expressions
  universal   Power laws, Gaussians, step functions

Examples:
  # Basic discovery with auto mode
  ppf discover data.csv -x time -y signal

  # Verbose output with oscillator mode
  ppf discover vibration.csv --mode oscillator -v -g 100

  # JSON output for piping to export
  ppf discover data.csv --json | ppf export python -f predict

  # High-quality search with more generations
  ppf discover data.csv -g 200 -p 1000 --optimize-constants

  # Show full Pareto front
  ppf discover data.csv --show-pareto --max-pareto 20
"""


def register(subparsers) -> None:
    """Register the discover command."""
    parser = subparsers.add_parser(
        "discover",
        help="Discover mathematical formulas using symbolic regression",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=lambda prog: __import__("argparse").RawDescriptionHelpFormatter(
            prog, max_help_position=35, width=100
        ),
    )

    # Data input arguments
    add_data_input_args(parser)

    # Algorithm parameters
    algo_group = parser.add_argument_group("Algorithm Options")
    algo_group.add_argument(
        "-m", "--mode",
        choices=[m.name.lower() for m in DiscoveryMode],
        default="auto",
        help="Discovery mode (default: auto)",
    )
    algo_group.add_argument(
        "-p", "--population-size",
        type=int,
        default=500,
        metavar="N",
        help="Population size for genetic programming (default: 500)",
    )
    algo_group.add_argument(
        "-g", "--generations",
        type=int,
        default=50,
        metavar="N",
        help="Number of generations to evolve (default: 50)",
    )
    algo_group.add_argument(
        "--max-depth",
        type=int,
        default=6,
        metavar="N",
        help="Maximum expression tree depth (default: 6)",
    )
    algo_group.add_argument(
        "--parsimony",
        type=float,
        default=0.001,
        metavar="COEF",
        help="Parsimony pressure coefficient (default: 0.001)",
    )
    algo_group.add_argument(
        "--optimize-constants",
        action="store_true",
        default=True,
        help="Optimize constants with local search (default: on)",
    )
    algo_group.add_argument(
        "--no-optimize-constants",
        action="store_false",
        dest="optimize_constants",
        help="Disable constant optimization",
    )
    algo_group.add_argument(
        "--random-state",
        type=int,
        default=None,
        metavar="SEED",
        help="Random seed for reproducibility",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--show-pareto",
        action="store_true",
        help="Show full Pareto front of solutions",
    )
    output_group.add_argument(
        "--max-pareto",
        type=int,
        default=10,
        metavar="N",
        help="Maximum Pareto solutions to show (default: 10)",
    )
    output_group.add_argument(
        "--latex",
        action="store_true",
        help="Include LaTeX formatted expressions",
    )
    output_group.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify expressions before output",
    )

    parser.set_defaults(func=run)


def run(args, output: IO[str]) -> int:
    """Execute the discover command."""
    try:
        # Load data
        print_progress("Loading data...", args)
        x, y = load_data(args)
        print_progress(f"Loaded {len(y)} data points", args)

    except ValueError as e:
        print_error(str(e), "Check file path and column specifications")
        return 1

    # Parse mode
    mode = DiscoveryMode[args.mode.upper()]

    # Progress callback for verbose mode
    def progress_callback(gen, best_fit, best_expr):
        if args.verbose and not args.quiet:
            print(
                f"Gen {gen:3d}: R²={best_fit:.6f}  {best_expr}",
                file=sys.stderr,
            )

    # Create regressor
    regressor = SymbolicRegressor(
        population_size=args.population_size,
        generations=args.generations,
        max_depth=args.max_depth,
        parsimony_coefficient=args.parsimony,
        optimize_constants=args.optimize_constants,
        random_state=args.random_state,
    )

    # Run discovery
    print_progress(f"Starting symbolic regression (mode={mode.name})...", args)
    callback = progress_callback if args.verbose else None
    result = regressor.discover(x, y, mode=mode, verbose=args.verbose)
    print_progress(f"Completed {result.generations_run} generations", args)

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
        "generations_run": result.generations_run,
        "total_evaluations": result.total_evaluations,
        "best_tradeoff": _fit_to_dict(result.best_tradeoff, args),
        "most_accurate": _fit_to_dict(result.most_accurate, args),
        "most_parsimonious": _fit_to_dict(result.most_parsimonious, args),
    }

    if args.show_pareto:
        data["pareto_front"] = [
            _fit_to_dict(fit, args)
            for fit in result.pareto_front[: args.max_pareto]
        ]

    if result.metadata:
        data["metadata"] = {
            "probed_domains": [d.name for d in result.metadata.get("probed_domains", [])],
            "selected_domain": result.metadata.get("selected_domain", {}).get("name"),
        }

    json.dump(data, output, indent=2)
    output.write("\n")


def _fit_to_dict(fit, args) -> dict:
    """Convert a SymbolicFitResult to a dictionary."""
    expr = fit.expression
    if args.simplify:
        expr = simplify_expression(expr)

    result = {
        "expression": expr.to_string(),
        "r_squared": fit.r_squared,
        "mse": fit.mse,
        "complexity": fit.complexity,
        "depth": expr.depth(),
        "is_noise_like": fit.is_noise_like,
    }

    if args.latex:
        result["latex"] = format_expression_latex(expr)

    return result


def _output_human(result, args, output: IO[str]) -> None:
    """Output results in human-readable format."""
    output.write("=" * 60 + "\n")
    output.write("SYMBOLIC REGRESSION RESULTS\n")
    output.write("=" * 60 + "\n\n")

    output.write(f"Generations: {result.generations_run}\n")
    output.write(f"Evaluations: {result.total_evaluations:,}\n\n")

    # Best tradeoff (main result)
    output.write("BEST TRADEOFF (recommended):\n")
    output.write("-" * 40 + "\n")
    _print_fit(result.best_tradeoff, args, output)

    output.write("\nMOST ACCURATE:\n")
    output.write("-" * 40 + "\n")
    _print_fit(result.most_accurate, args, output)

    output.write("\nSIMPLEST:\n")
    output.write("-" * 40 + "\n")
    _print_fit(result.most_parsimonious, args, output)

    if args.show_pareto:
        output.write("\n" + "=" * 60 + "\n")
        output.write(f"PARETO FRONT (top {args.max_pareto}):\n")
        output.write("=" * 60 + "\n\n")

        for i, fit in enumerate(result.pareto_front[: args.max_pareto], 1):
            output.write(f"[{i}] R²={fit.r_squared:.6f}  C={fit.complexity:2d}  ")
            expr = fit.expression
            if args.simplify:
                expr = simplify_expression(expr)
            output.write(f"{expr.to_string()}\n")


def _print_fit(fit, args, output: IO[str]) -> None:
    """Print a single fit result."""
    expr = fit.expression
    if args.simplify:
        expr = simplify_expression(expr)

    output.write(f"  Expression: {expr.to_string()}\n")
    output.write(f"  R-squared:  {fit.r_squared:.6f}\n")
    output.write(f"  MSE:        {fit.mse:.6e}\n")
    output.write(f"  Complexity: {fit.complexity}\n")
    output.write(f"  Depth:      {expr.depth()}\n")

    if args.latex:
        output.write(f"  LaTeX:      {format_expression_latex(expr)}\n")
