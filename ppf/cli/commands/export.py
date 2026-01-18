"""
Export command - Export expressions to Python, C, or JSON.

Converts discovered expressions into deployable code or portable formats.
"""

import json
import sys
from typing import IO

from ppf import (
    export_python,
    export_c,
    export_json,
    load_json_string,
    bundle_to_json_string,
    SCHEMA_VERSION,
)
from ppf.cli.utils import print_error, print_progress


DESCRIPTION = """\
WHAT IT DOES:
  Converts discovered mathematical expressions into deployable code:

  python  Generate a Python function that evaluates the expression
  c       Generate C code (header-style) for embedded systems
  json    Export to JSON for storage or interchange

HOW TO USE:
  Pipe JSON output from 'ppf discover' to the export command:

    ppf discover data.csv --json | ppf export python -f predict

  Or export from a saved JSON file:

    ppf export python -f predict < expression.json

  The generated code can be used directly in your application.
"""

EPILOG = """\
Subcommands:
  ppf export python   Generate Python function
  ppf export c        Generate C code
  ppf export json     Export/transform JSON

Use 'ppf export <subcommand> --help' for subcommand-specific options.

Examples:
  # Generate Python function
  ppf discover data.csv --json | ppf export python -f my_model

  # Generate C code for Arduino
  ppf discover data.csv --json | ppf export c -f sensor_model --float

  # Transform JSON with additional metadata
  ppf discover data.csv --json | ppf export json --include-tree

Pipeline Example:
  # Full workflow: discover, export, and save
  ppf discover data.csv --json | ppf export python -f predict > model.py
"""


def register(subparsers) -> None:
    """Register the export command with subcommands."""
    parser = subparsers.add_parser(
        "export",
        help="Export expressions to Python/C/JSON",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=lambda prog: __import__("argparse").RawDescriptionHelpFormatter(
            prog, max_help_position=35, width=100
        ),
    )

    # Create subparsers for export formats
    export_subparsers = parser.add_subparsers(
        title="Export Formats",
        dest="export_format",
        metavar="<format>",
    )

    # Python export
    _register_python(export_subparsers)

    # C export
    _register_c(export_subparsers)

    # JSON export
    _register_json(export_subparsers)

    parser.set_defaults(func=run_export_help)


def _register_python(subparsers) -> None:
    """Register the python export subcommand."""
    parser = subparsers.add_parser(
        "python",
        help="Generate Python function",
        description="""\
WHAT IT DOES:
  Generates a Python function that evaluates the discovered expression.
  The function uses only numpy operations and can be used directly.

HOW TO USE:
  ppf discover data.csv --json | ppf export python -f predict

  Import and use the generated function:
    from model import predict
    y = predict(x_values)
""",
        epilog="""\
Examples:
  # Basic Python export
  ppf discover data.csv --json | ppf export python -f model > model.py

  # Custom variable name
  ppf discover data.csv --json | ppf export python -f predict --variable t

  # With bounds checking
  ppf discover data.csv --json | ppf export python -f predict --safe
""",
        formatter_class=lambda prog: __import__("argparse").RawDescriptionHelpFormatter(
            prog, max_help_position=35, width=100
        ),
    )

    parser.add_argument(
        "-f", "--function-name",
        default="evaluate",
        metavar="NAME",
        help="Name for the generated function (default: evaluate)",
    )
    parser.add_argument(
        "--variable",
        default="x",
        metavar="VAR",
        help="Variable name in generated code (default: x)",
    )
    parser.add_argument(
        "--safe",
        action="store_true",
        default=False,
        help="Add bounds checking and error handling",
    )
    parser.add_argument(
        "--no-safe",
        action="store_false",
        dest="safe",
        help="Skip safety checks for performance",
    )

    parser.set_defaults(func=run_python)


def _register_c(subparsers) -> None:
    """Register the C export subcommand."""
    parser = subparsers.add_parser(
        "c",
        help="Generate C code",
        description="""\
WHAT IT DOES:
  Generates C code that evaluates the discovered expression.
  Suitable for embedded systems, microcontrollers, and performance-critical
  applications.

HOW TO USE:
  ppf discover data.csv --json | ppf export c -f model > model.h

  Include and use in your C program:
    #include "model.h"
    double y = model(x);
""",
        epilog="""\
Examples:
  # Basic C export
  ppf discover data.csv --json | ppf export c -f sensor_model > model.h

  # Use float instead of double (for microcontrollers)
  ppf discover data.csv --json | ppf export c -f model --float

  # Macro-style for inline expansion
  ppf discover data.csv --json | ppf export c --macro-style
""",
        formatter_class=lambda prog: __import__("argparse").RawDescriptionHelpFormatter(
            prog, max_help_position=35, width=100
        ),
    )

    parser.add_argument(
        "-f", "--function-name",
        default="evaluate",
        metavar="NAME",
        help="Name for the generated function (default: evaluate)",
    )
    parser.add_argument(
        "--variable",
        default="x",
        metavar="VAR",
        help="Variable name in generated code (default: x)",
    )
    parser.add_argument(
        "--safe",
        action="store_true",
        default=False,
        help="Add bounds checking (increases code size)",
    )
    parser.add_argument(
        "--no-safe",
        action="store_false",
        dest="safe",
        help="Skip safety checks",
    )
    parser.add_argument(
        "--float",
        action="store_true",
        dest="use_float",
        help="Use float instead of double (for embedded)",
    )
    parser.add_argument(
        "--macro-style",
        action="store_true",
        help="Generate as preprocessor macro",
    )

    parser.set_defaults(func=run_c)


def _register_json(subparsers) -> None:
    """Register the JSON export subcommand."""
    parser = subparsers.add_parser(
        "json",
        help="Export/transform JSON",
        description="""\
WHAT IT DOES:
  Transforms or enhances JSON expression data. Can:
  - Add expression tree structure
  - Add metadata (source, variables)
  - Re-format with custom indentation

HOW TO USE:
  ppf discover data.csv --json | ppf export json --include-tree

  Use --source to tag where the expression came from.
""",
        epilog="""\
Examples:
  # Add tree structure
  ppf discover data.csv --json | ppf export json --include-tree

  # Add source metadata
  ppf discover data.csv --json | ppf export json --source "sensor_v1"

  # Compact output
  ppf discover data.csv --json | ppf export json --indent 0
""",
        formatter_class=lambda prog: __import__("argparse").RawDescriptionHelpFormatter(
            prog, max_help_position=35, width=100
        ),
    )

    parser.add_argument(
        "--include-tree",
        action="store_true",
        help="Include full expression tree in output",
    )
    parser.add_argument(
        "--no-tree",
        action="store_false",
        dest="include_tree",
        help="Exclude expression tree (default)",
    )
    parser.add_argument(
        "--source",
        metavar="NAME",
        help="Source identifier for metadata",
    )
    parser.add_argument(
        "--variables",
        metavar="VARS",
        help="Comma-separated variable names",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        metavar="N",
        help="JSON indentation (0 for compact, default: 2)",
    )

    parser.set_defaults(func=run_json)


def run_export_help(args, output: IO[str]) -> int:
    """Show help when no subcommand given."""
    print_error(
        "No export format specified",
        "Use: ppf export python|c|json [options]",
    )
    return 1


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


def _get_expression_from_input(data: dict):
    """Extract expression node from input JSON."""
    # Handle direct expression JSON
    if "node_type" in data:
        return load_json_string(json.dumps(data))

    # Handle discover command output
    if "best_tradeoff" in data:
        expr_str = data["best_tradeoff"].get("expression")
        if expr_str:
            # We need the actual expression node, not just the string
            # The JSON export should include the tree
            if "expression_tree" in data["best_tradeoff"]:
                return load_json_string(json.dumps(data["best_tradeoff"]["expression_tree"]))
            else:
                raise ValueError(
                    "Input JSON doesn't contain expression tree. "
                    "Use 'ppf discover --json' which includes the tree structure."
                )

    raise ValueError(
        "Cannot find expression in input. Expected discover output or expression JSON."
    )


def run_python(args, output: IO[str]) -> int:
    """Execute python export."""
    try:
        data = _read_input_json()

        # For discover output, we can generate code from expression string
        if "best_tradeoff" in data:
            expr_str = data["best_tradeoff"].get("expression", "")
            code = _generate_python_from_string(
                expr_str,
                args.function_name,
                args.variable,
                args.safe,
            )
        else:
            # Direct expression tree
            expr = load_json_string(json.dumps(data))
            code = export_python(expr, var_name=args.variable, safe=args.safe)
            # Wrap in function
            code = _wrap_python_function(code, args.function_name, args.variable)

        output.write(code)
        output.write("\n")
        return 0

    except ValueError as e:
        print_error(str(e))
        return 1


def _generate_python_from_string(expr_str: str, func_name: str, var: str, safe: bool) -> str:
    """Generate Python code from an expression string."""
    lines = [
        '"""',
        f"Auto-generated model function.",
        f"Expression: {expr_str}",
        '"""',
        "",
        "import numpy as np",
        "",
        "",
        f"def {func_name}({var}):",
        f'    """Evaluate the discovered expression."""',
    ]

    if safe:
        lines.extend([
            f"    {var} = np.asarray({var})",
        ])

    # Convert expression string to numpy code
    numpy_expr = expr_str
    # Replace common operators with numpy equivalents
    numpy_expr = numpy_expr.replace("sin(", "np.sin(")
    numpy_expr = numpy_expr.replace("cos(", "np.cos(")
    numpy_expr = numpy_expr.replace("exp(", "np.exp(")
    numpy_expr = numpy_expr.replace("log(", "np.log(")
    numpy_expr = numpy_expr.replace("sqrt(", "np.sqrt(")
    numpy_expr = numpy_expr.replace("abs(", "np.abs(")
    numpy_expr = numpy_expr.replace("^", "**")

    lines.append(f"    return {numpy_expr}")

    return "\n".join(lines)


def _wrap_python_function(code: str, func_name: str, var: str) -> str:
    """Wrap expression code in a function."""
    lines = [
        "import numpy as np",
        "",
        "",
        f"def {func_name}({var}):",
        f'    """Evaluate the discovered expression."""',
        f"    return {code}",
    ]
    return "\n".join(lines)


def run_c(args, output: IO[str]) -> int:
    """Execute C export."""
    try:
        data = _read_input_json()

        # For discover output, generate from expression string
        if "best_tradeoff" in data:
            expr_str = data["best_tradeoff"].get("expression", "")
            code = _generate_c_from_string(
                expr_str,
                args.function_name,
                args.variable,
                args.safe,
                args.use_float,
                args.macro_style,
            )
        else:
            expr = load_json_string(json.dumps(data))
            code = export_c(expr, var_name=args.variable, safe=args.safe)
            code = _wrap_c_function(
                code, args.function_name, args.variable,
                args.use_float, args.macro_style
            )

        output.write(code)
        output.write("\n")
        return 0

    except ValueError as e:
        print_error(str(e))
        return 1


def _generate_c_from_string(
    expr_str: str,
    func_name: str,
    var: str,
    safe: bool,
    use_float: bool,
    macro_style: bool,
) -> str:
    """Generate C code from an expression string."""
    dtype = "float" if use_float else "double"

    # Convert expression to C syntax
    c_expr = expr_str
    c_expr = c_expr.replace("^", "pow(")
    # Handle power operations more carefully
    # This is simplified - real implementation would parse properly

    if macro_style:
        lines = [
            f"/* Auto-generated expression macro */",
            f"/* Expression: {expr_str} */",
            "",
            f"#ifndef {func_name.upper()}_H",
            f"#define {func_name.upper()}_H",
            "",
            "#include <math.h>",
            "",
            f"#define {func_name.upper()}({var}) ({c_expr})",
            "",
            f"#endif /* {func_name.upper()}_H */",
        ]
    else:
        lines = [
            f"/* Auto-generated model function */",
            f"/* Expression: {expr_str} */",
            "",
            f"#ifndef {func_name.upper()}_H",
            f"#define {func_name.upper()}_H",
            "",
            "#include <math.h>",
            "",
            f"static inline {dtype} {func_name}({dtype} {var}) {{",
            f"    return {c_expr};",
            "}",
            "",
            f"#endif /* {func_name.upper()}_H */",
        ]

    return "\n".join(lines)


def _wrap_c_function(code: str, func_name: str, var: str, use_float: bool, macro_style: bool) -> str:
    """Wrap expression code in a C function."""
    dtype = "float" if use_float else "double"

    lines = [
        "#include <math.h>",
        "",
        f"static inline {dtype} {func_name}({dtype} {var}) {{",
        f"    return {code};",
        "}",
    ]
    return "\n".join(lines)


def run_json(args, output: IO[str]) -> int:
    """Execute JSON export/transform."""
    try:
        data = _read_input_json()

        # Add metadata if requested
        if args.source:
            data["source"] = args.source

        if args.variables:
            data["variables"] = [v.strip() for v in args.variables.split(",")]

        data["schema_version"] = SCHEMA_VERSION

        # Format output
        indent = args.indent if args.indent > 0 else None
        json.dump(data, output, indent=indent)
        output.write("\n")
        return 0

    except ValueError as e:
        print_error(str(e))
        return 1
