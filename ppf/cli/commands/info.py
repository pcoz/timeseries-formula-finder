"""
Info command - Show available discovery modes, forms, and macros.

Displays documentation about the available options and capabilities
of the PPF library.
"""

import json
from typing import IO

from ppf import (
    DiscoveryMode,
    FormType,
    MacroOp,
    EntropyMethod,
    MODE_TO_PRIMITIVES,
    FAMILY_CATEGORIES,
)
from ppf.cli.utils import print_error


DESCRIPTION = """\
WHAT IT DOES:
  Displays information about PPF capabilities and options.
  Use this to explore available modes, forms, macros, and more.

HOW TO USE:
  ppf info <topic>

  Available topics:
    modes       Discovery modes and their use cases
    forms       Mathematical form types
    macros      Macro operators (template functions)
    primitives  Primitive operators for GP
    schemas     Feature extraction schemas
    methods     Entropy measurement methods
    all         Show everything
"""

EPILOG = """\
Examples:
  # Show discovery modes
  ppf info modes

  # Show available macros
  ppf info macros

  # Show all information
  ppf info all

  # JSON output for programmatic use
  ppf info modes --json
"""


def register(subparsers) -> None:
    """Register the info command."""
    parser = subparsers.add_parser(
        "info",
        help="Show available discovery modes, forms, and macros",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=lambda prog: __import__("argparse").RawDescriptionHelpFormatter(
            prog, max_help_position=35, width=100
        ),
    )

    parser.add_argument(
        "topic",
        nargs="?",
        default="all",
        choices=["modes", "forms", "macros", "primitives", "schemas", "methods", "all"],
        help="Topic to display (default: all)",
    )

    parser.set_defaults(func=run)


def run(args, output: IO[str]) -> int:
    """Execute the info command."""
    topic = args.topic

    if args.json:
        _output_json(topic, output)
    else:
        _output_human(topic, output)

    return 0


def _output_json(topic: str, output: IO[str]) -> None:
    """Output information as JSON."""
    data = {}

    if topic in ("modes", "all"):
        data["modes"] = _get_modes_info()

    if topic in ("forms", "all"):
        data["forms"] = _get_forms_info()

    if topic in ("macros", "all"):
        data["macros"] = _get_macros_info()

    if topic in ("primitives", "all"):
        data["primitives"] = _get_primitives_info()

    if topic in ("schemas", "all"):
        data["schemas"] = _get_schemas_info()

    if topic in ("methods", "all"):
        data["methods"] = _get_methods_info()

    json.dump(data, output, indent=2)
    output.write("\n")


def _output_human(topic: str, output: IO[str]) -> None:
    """Output information in human-readable format."""
    if topic in ("modes", "all"):
        _print_modes(output)

    if topic in ("forms", "all"):
        _print_forms(output)

    if topic in ("macros", "all"):
        _print_macros(output)

    if topic in ("primitives", "all"):
        _print_primitives(output)

    if topic in ("schemas", "all"):
        _print_schemas(output)

    if topic in ("methods", "all"):
        _print_methods(output)


# ============================================================
# Mode Information
# ============================================================

MODE_DESCRIPTIONS = {
    DiscoveryMode.AUTO: (
        "Automatic domain probing",
        "Probes multiple domains and selects the best fit. Recommended for unknown data."
    ),
    DiscoveryMode.IDENTIFY: (
        "Template-based matching",
        "Uses macro templates for fast identification of common patterns."
    ),
    DiscoveryMode.DISCOVER: (
        "Pure discovery mode",
        "No templates, fully exploratory search for novel formulas."
    ),
    DiscoveryMode.OSCILLATOR: (
        "Damped oscillations",
        "Specialized for vibrations, decaying waves, resonance."
    ),
    DiscoveryMode.CIRCUIT: (
        "RC/RLC circuits",
        "Optimized for electronic circuit responses (charge, discharge)."
    ),
    DiscoveryMode.GROWTH: (
        "Population dynamics",
        "Logistic growth, saturation curves, biological models."
    ),
    DiscoveryMode.RATIONAL: (
        "Rational functions",
        "Polynomial ratios, transfer functions."
    ),
    DiscoveryMode.POLYNOMIAL: (
        "Algebraic expressions",
        "Pure polynomial and algebraic forms."
    ),
    DiscoveryMode.UNIVERSAL: (
        "Universal functions",
        "Power laws, Gaussians, step functions, general purpose."
    ),
}


def _get_modes_info() -> list:
    """Get mode information as structured data."""
    return [
        {
            "name": mode.name,
            "short": desc[0],
            "description": desc[1],
        }
        for mode, desc in MODE_DESCRIPTIONS.items()
    ]


def _print_modes(output: IO[str]) -> None:
    """Print discovery modes."""
    output.write("=" * 60 + "\n")
    output.write("DISCOVERY MODES\n")
    output.write("=" * 60 + "\n\n")

    for mode, (short, desc) in MODE_DESCRIPTIONS.items():
        output.write(f"{mode.name.lower():12s}  {short}\n")
        output.write(f"              {desc}\n\n")


# ============================================================
# Form Information
# ============================================================

FORM_DESCRIPTIONS = {
    FormType.CONSTANT: ("y = c", "Flat, constant value"),
    FormType.LINEAR: ("y = ax + b", "Straight line trend"),
    FormType.QUADRATIC: ("y = ax^2 + bx + c", "Parabolic curve"),
    FormType.SINE: ("y = A*sin(wt + phi) + c", "Periodic oscillation"),
    FormType.EXPONENTIAL: ("y = a*exp(bx) + c", "Growth or decay"),
}


def _get_forms_info() -> list:
    """Get form information as structured data."""
    return [
        {
            "name": form.name,
            "formula": desc[0],
            "description": desc[1],
        }
        for form, desc in FORM_DESCRIPTIONS.items()
    ]


def _print_forms(output: IO[str]) -> None:
    """Print form types."""
    output.write("=" * 60 + "\n")
    output.write("FORM TYPES\n")
    output.write("=" * 60 + "\n\n")

    for form, (formula, desc) in FORM_DESCRIPTIONS.items():
        output.write(f"{form.name:12s}  {formula:25s}  {desc}\n")

    output.write("\n")


# ============================================================
# Macro Information
# ============================================================

MACRO_DESCRIPTIONS = {
    MacroOp.DAMPED_SIN: ("a*exp(-k*t)*sin(w*t+phi)", "Damped sinusoid"),
    MacroOp.DAMPED_COS: ("a*exp(-k*t)*cos(w*t+phi)", "Damped cosinusoid"),
    MacroOp.RC_CHARGE: ("a*(1-exp(-k*t))+c", "RC circuit charging"),
    MacroOp.EXP_DECAY: ("a*exp(-k*t)+c", "Exponential decay"),
    MacroOp.RATIO: ("(a*x+b)/(c*x+d)", "Linear ratio"),
    MacroOp.RATIONAL2: ("(a*x^2+b*x+c)/(d*x+e)", "Quadratic/linear ratio"),
    MacroOp.SIGMOID: ("a/(1+exp(-k*(x-x0)))+c", "Sigmoid function"),
    MacroOp.LOGISTIC: ("K/(1+exp(-r*(x-x0)))", "Logistic growth"),
    MacroOp.HILL: ("a*x^n/(k^n+x^n)", "Hill equation"),
    MacroOp.POWER_LAW: ("a*x^b+c", "Power law"),
    MacroOp.GAUSSIAN: ("a*exp(-((x-mu)/sigma)^2)+c", "Gaussian bell curve"),
    MacroOp.TANH_STEP: ("a*tanh(k*(x-x0))+c", "Smooth step function"),
}


def _get_macros_info() -> list:
    """Get macro information as structured data."""
    return [
        {
            "name": macro.name,
            "formula": desc[0],
            "description": desc[1],
        }
        for macro, desc in MACRO_DESCRIPTIONS.items()
    ]


def _print_macros(output: IO[str]) -> None:
    """Print macro operators."""
    output.write("=" * 60 + "\n")
    output.write("MACRO OPERATORS\n")
    output.write("=" * 60 + "\n\n")

    output.write("Macros are template functions with optimizable parameters.\n")
    output.write("They accelerate discovery by encoding common patterns.\n\n")

    for macro, (formula, desc) in MACRO_DESCRIPTIONS.items():
        output.write(f"{macro.name:12s}\n")
        output.write(f"  Formula: {formula}\n")
        output.write(f"  Use:     {desc}\n\n")


# ============================================================
# Primitive Information
# ============================================================

PRIMITIVE_SETS = {
    "MINIMAL": "Basic operators: +, -, *, /, constants",
    "TRIG": "Sine, cosine, plus polynomial operators",
    "GROWTH": "Exp, log, sqrt for growth/decay curves",
    "POLYNOMIAL": "Pure algebraic operators only",
    "OSCILLATOR": "Damped sinusoids and related",
    "CIRCUIT": "RC charge, exponential decay",
    "PHYSICS": "Combined physics operators",
    "RATIONAL": "Polynomial ratio operators",
    "SATURATION": "Sigmoid, logistic, Hill functions",
    "UNIVERSAL": "Power laws, Gaussians, step functions",
    "ALL_MACROS": "All available template macros",
}


def _get_primitives_info() -> dict:
    """Get primitive set information."""
    return PRIMITIVE_SETS


def _print_primitives(output: IO[str]) -> None:
    """Print primitive sets."""
    output.write("=" * 60 + "\n")
    output.write("PRIMITIVE SETS\n")
    output.write("=" * 60 + "\n\n")

    output.write("Primitive sets define which operators are available during GP.\n")
    output.write("Each discovery mode uses an appropriate primitive set.\n\n")

    for name, desc in PRIMITIVE_SETS.items():
        output.write(f"{name:15s}  {desc}\n")

    output.write("\n")


# ============================================================
# Schema Information
# ============================================================

def _get_schemas_info() -> dict:
    """Get feature schema information."""
    return {
        "families": FAMILY_CATEGORIES,
        "categories": ["structural", "operators", "family", "complexity"],
    }


def _print_schemas(output: IO[str]) -> None:
    """Print feature schemas."""
    output.write("=" * 60 + "\n")
    output.write("FEATURE SCHEMAS\n")
    output.write("=" * 60 + "\n\n")

    output.write("Family Categories:\n")
    for cat in FAMILY_CATEGORIES:
        output.write(f"  - {cat}\n")

    output.write("\nFeature Categories:\n")
    output.write("  - structural (depth, size, etc.)\n")
    output.write("  - operators (presence of each operator)\n")
    output.write("  - family (expression family indicators)\n")
    output.write("  - complexity (various complexity metrics)\n")
    output.write("\n")

    output.write("Use 'ppf features --schema' for detailed feature list.\n")
    output.write("\n")


# ============================================================
# Method Information
# ============================================================

METHOD_DESCRIPTIONS = {
    EntropyMethod.GZIP: (
        "Compression-based entropy",
        "Uses gzip compression ratio to estimate entropy. "
        "Works well for most signals. Lower values = more structure."
    ),
    EntropyMethod.SPECTRAL: (
        "Spectral flatness",
        "Measures how flat the power spectrum is. "
        "Better for audio/vibration signals. Lower values = more tonal."
    ),
}


def _get_methods_info() -> list:
    """Get entropy method information."""
    return [
        {
            "name": method.name,
            "short": desc[0],
            "description": desc[1],
        }
        for method, desc in METHOD_DESCRIPTIONS.items()
    ]


def _print_methods(output: IO[str]) -> None:
    """Print entropy methods."""
    output.write("=" * 60 + "\n")
    output.write("ENTROPY METHODS\n")
    output.write("=" * 60 + "\n\n")

    for method, (short, desc) in METHOD_DESCRIPTIONS.items():
        output.write(f"{method.name.lower():10s}  {short}\n")
        output.write(f"            {desc}\n\n")
