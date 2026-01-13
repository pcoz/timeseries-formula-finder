"""
Python code generation for PPF expressions.

Generates standalone Python functions that evaluate expression trees
without any PPF dependencies at runtime.
"""

from typing import Tuple
import math

from ..symbolic_types import (
    ExprNode, NodeType, UnaryOp, BinaryOp, MacroOp,
)

# Safety wrapper code templates
SAFE_HELPERS = '''
def safe_div(a, b, eps=1e-12):
    """Division protected against zero."""
    return a / (b if abs(b) > eps else (eps if b >= 0 else -eps))


def safe_log(x, eps=1e-12):
    """Logarithm protected against non-positive values."""
    return math.log(max(x, eps))


def clamp_exp(x, lo=-60, hi=60):
    """Exponential with clamped argument to prevent overflow."""
    return math.exp(min(max(x, lo), hi))

'''


def _format_constant(value: float, precision: int = 12) -> str:
    """Format a constant with controlled precision."""
    if value == int(value) and abs(value) < 1e10:
        return str(int(value))

    # Use repr for high precision, then clean up
    formatted = f"{value:.{precision}g}"

    # Remove trailing zeros after decimal point but keep at least one
    if '.' in formatted and 'e' not in formatted.lower():
        formatted = formatted.rstrip('0').rstrip('.')
        if '.' not in formatted:
            formatted += '.0'

    return formatted


def _emit_expr(node: ExprNode, var_name: str, safe: bool) -> str:
    """
    Recursively emit Python expression string for an ExprNode.

    Args:
        node: Expression tree node
        var_name: Variable name (e.g., "t" or "x")
        safe: Whether to use safe wrappers

    Returns:
        Python expression string
    """
    if node.node_type == NodeType.CONSTANT:
        return _format_constant(node.value)

    elif node.node_type == NodeType.VARIABLE:
        return var_name

    elif node.node_type == NodeType.UNARY_OP:
        child = _emit_expr(node.child, var_name, safe)

        if node.unary_op == UnaryOp.SIN:
            return f"math.sin({child})"
        elif node.unary_op == UnaryOp.COS:
            return f"math.cos({child})"
        elif node.unary_op == UnaryOp.EXP:
            if safe:
                return f"clamp_exp({child})"
            return f"math.exp({child})"
        elif node.unary_op == UnaryOp.LOG:
            if safe:
                return f"safe_log({child})"
            return f"math.log({child})"
        elif node.unary_op == UnaryOp.SQRT:
            if safe:
                return f"math.sqrt(abs({child}))"
            return f"math.sqrt({child})"
        elif node.unary_op == UnaryOp.NEG:
            return f"-({child})"
        elif node.unary_op == UnaryOp.ABS:
            return f"abs({child})"
        elif node.unary_op == UnaryOp.SQUARE:
            return f"({child}) ** 2"
        else:
            raise ValueError(f"Unsupported unary operator: {node.unary_op}")

    elif node.node_type == NodeType.BINARY_OP:
        left = _emit_expr(node.left, var_name, safe)
        right = _emit_expr(node.right, var_name, safe)

        if node.binary_op == BinaryOp.ADD:
            return f"({left} + {right})"
        elif node.binary_op == BinaryOp.SUB:
            return f"({left} - {right})"
        elif node.binary_op == BinaryOp.MUL:
            return f"({left} * {right})"
        elif node.binary_op == BinaryOp.DIV:
            if safe:
                return f"safe_div({left}, {right})"
            return f"({left} / {right})"
        elif node.binary_op == BinaryOp.POW:
            if safe:
                return f"(abs({left}) ** {right})"
            return f"({left} ** {right})"
        else:
            raise ValueError(f"Unsupported binary operator: {node.binary_op}")

    elif node.node_type == NodeType.MACRO:
        return _emit_macro(node, var_name, safe)

    raise ValueError(f"Unknown node type: {node.node_type}")


def _emit_macro(node: ExprNode, var_name: str, safe: bool) -> str:
    """Emit Python code for a macro node."""
    params = node.macro_params
    x = var_name

    # Format parameters
    def p(i: int) -> str:
        return _format_constant(params[i])

    exp_func = "clamp_exp" if safe else "math.exp"

    if node.macro_op == MacroOp.DAMPED_SIN:
        # a * exp(-k*x) * sin(w*x + phi)
        a, k, w, phi = params
        return f"({p(0)} * {exp_func}({_format_constant(-abs(k))} * {x}) * math.sin({p(2)} * {x} + {p(3)}))"

    elif node.macro_op == MacroOp.DAMPED_COS:
        # a * exp(-k*x) * cos(w*x + phi)
        a, k, w, phi = params
        return f"({p(0)} * {exp_func}({_format_constant(-abs(k))} * {x}) * math.cos({p(2)} * {x} + {p(3)}))"

    elif node.macro_op == MacroOp.RC_CHARGE:
        # a * (1 - exp(-k*x)) + c
        a, k, c = params
        return f"({p(0)} * (1 - {exp_func}({_format_constant(-abs(k))} * {x})) + {p(2)})"

    elif node.macro_op == MacroOp.EXP_DECAY:
        # a * exp(-k*x) + c
        a, k, c = params
        return f"({p(0)} * {exp_func}({_format_constant(-abs(k))} * {x}) + {p(2)})"

    elif node.macro_op == MacroOp.RATIO:
        # (a*x + b) / (c*x + d)
        a, b, c, d = params
        num = f"({p(0)} * {x} + {p(1)})"
        den = f"({p(2)} * {x} + {p(3)})"
        if safe:
            return f"safe_div({num}, {den})"
        return f"({num} / {den})"

    elif node.macro_op == MacroOp.RATIONAL2:
        # (a*x^2 + b*x + c) / (d*x^2 + e*x + f)
        a, b, c, d, e, f = params
        num = f"({p(0)} * {x}**2 + {p(1)} * {x} + {p(2)})"
        den = f"({p(3)} * {x}**2 + {p(4)} * {x} + {p(5)})"
        if safe:
            return f"safe_div({num}, {den})"
        return f"({num} / {den})"

    elif node.macro_op == MacroOp.SIGMOID:
        # a / (1 + exp(-k*(x - x0)))
        a, k, x0 = params
        return f"({p(0)} / (1 + {exp_func}(-{p(1)} * ({x} - {p(2)}))))"

    elif node.macro_op == MacroOp.LOGISTIC:
        # a / (1 + b*exp(-k*x))
        a, b, k = params
        return f"({p(0)} / (1 + {_format_constant(abs(b))} * {exp_func}(-{p(2)} * {x})))"

    elif node.macro_op == MacroOp.HILL:
        # a * x^n / (k^n + x^n)
        a, k, n = params
        k_abs = _format_constant(abs(k) + 1e-10)
        x_abs = f"(abs({x}) + 1e-10)"
        n_val = _format_constant(max(0.5, min(n, 5.0)))
        return f"({p(0)} * {x_abs}**{n_val} / ({k_abs}**{n_val} + {x_abs}**{n_val}))"

    elif node.macro_op == MacroOp.POWER_LAW:
        # a * x^b + c
        a, b, c = params
        x_abs = f"(abs({x}) + 1e-10)"
        return f"({p(0)} * {x_abs}**{p(1)} + {p(2)})"

    elif node.macro_op == MacroOp.GAUSSIAN:
        # a * exp(-((x-mu)/sigma)^2)
        a, mu, sigma = params
        sigma_abs = _format_constant(abs(sigma) + 1e-10)
        z = f"(({x} - {p(1)}) / {sigma_abs})"
        return f"({p(0)} * {exp_func}(-{z}**2))"

    elif node.macro_op == MacroOp.TANH_STEP:
        # a * tanh(k*(x - x0)) + c
        a, k, x0, c = params
        return f"({p(0)} * math.tanh({p(1)} * ({x} - {p(2)})) + {p(3)})"

    raise ValueError(f"Unsupported macro: {node.macro_op}")


def export_python(
    expr: ExprNode,
    *,
    fn_name: str = "ppf_model",
    signature: Tuple[str, ...] = ("t",),
    safe: bool = True
) -> str:
    """
    Generate a standalone Python function that evaluates the expression.

    Args:
        expr: Expression tree to export
        fn_name: Name for the generated function
        signature: Variable names in function signature
        safe: Whether to emit safety wrappers for div/log/exp

    Returns:
        Python code string that can be executed independently

    Example:
        >>> code = export_python(my_expr, fn_name="predict", signature=("t",))
        >>> exec(code)
        >>> result = predict(1.5)
    """
    # Build function body
    var_name = signature[0] if signature else "x"
    body_expr = _emit_expr(expr, var_name, safe)

    # Build output
    lines = ["import math", ""]

    if safe:
        lines.append(SAFE_HELPERS)

    # Function signature
    sig_str = ", ".join(signature)
    lines.append(f"def {fn_name}({sig_str}):")
    lines.append(f"    return {body_expr}")
    lines.append("")

    return "\n".join(lines)
