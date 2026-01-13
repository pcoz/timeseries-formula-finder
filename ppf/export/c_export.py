"""
C code generation for PPF expressions.

Generates standalone C99 functions that evaluate expression trees
for embedded/edge deployment.
"""

from typing import Tuple

from ..symbolic_types import (
    ExprNode, NodeType, UnaryOp, BinaryOp, MacroOp,
)


# Safety wrappers for double precision
SAFE_HELPERS_DOUBLE = '''
static inline double safe_div(double a, double b) {
    const double eps = 1e-12;
    if (fabs(b) > eps) return a / b;
    return a / (b >= 0 ? eps : -eps);
}

static inline double safe_log(double x) {
    const double eps = 1e-12;
    return log(x > eps ? x : eps);
}

static inline double clamp_exp_arg(double x) {
    if (x < -60.0) return -60.0;
    if (x > 60.0) return 60.0;
    return x;
}

'''

# Safety wrappers for float precision
SAFE_HELPERS_FLOAT = '''
static inline float safe_divf(float a, float b) {
    const float eps = 1e-6f;
    if (fabsf(b) > eps) return a / b;
    return a / (b >= 0 ? eps : -eps);
}

static inline float safe_logf(float x) {
    const float eps = 1e-6f;
    return logf(x > eps ? x : eps);
}

static inline float clamp_exp_argf(float x) {
    if (x < -60.0f) return -60.0f;
    if (x > 60.0f) return 60.0f;
    return x;
}

'''

# Macro helper functions (double precision)
MACRO_HELPERS_DOUBLE = '''
static inline double damped_sin(double a, double k, double t, double w, double phi) {
    return a * exp(clamp_exp_arg(-k * t)) * sin(w * t + phi);
}

static inline double damped_cos(double a, double k, double t, double w, double phi) {
    return a * exp(clamp_exp_arg(-k * t)) * cos(w * t + phi);
}

static inline double rc_charge(double a, double k, double t, double c) {
    return a * (1.0 - exp(clamp_exp_arg(-k * t))) + c;
}

static inline double exp_decay(double a, double k, double t, double c) {
    return a * exp(clamp_exp_arg(-k * t)) + c;
}

static inline double sigmoid(double a, double k, double x, double x0) {
    return a / (1.0 + exp(clamp_exp_arg(-k * (x - x0))));
}

static inline double logistic(double a, double b, double k, double x) {
    return a / (1.0 + fabs(b) * exp(clamp_exp_arg(-k * x)));
}

static inline double hill(double a, double k, double n, double x) {
    const double eps = 1e-10;
    double k_abs = fabs(k) + eps;
    double x_abs = fabs(x) + eps;
    double n_clamped = n < 0.5 ? 0.5 : (n > 5.0 ? 5.0 : n);
    return a * pow(x_abs, n_clamped) / (pow(k_abs, n_clamped) + pow(x_abs, n_clamped));
}

static inline double ratio(double a, double b, double c, double d, double x) {
    return safe_div(a * x + b, c * x + d);
}

static inline double rational2(double a, double b, double c, double d, double e, double f, double x) {
    return safe_div(a * x * x + b * x + c, d * x * x + e * x + f);
}

static inline double power_law(double a, double b, double x, double c) {
    const double eps = 1e-10;
    return a * pow(fabs(x) + eps, b) + c;
}

static inline double gaussian(double a, double mu, double sigma, double x) {
    const double eps = 1e-10;
    double sigma_abs = fabs(sigma) + eps;
    double z = (x - mu) / sigma_abs;
    return a * exp(-z * z);
}

static inline double tanh_step(double a, double k, double x0, double x, double c) {
    return a * tanh(k * (x - x0)) + c;
}

'''


def _format_constant(value: float, use_float: bool = False, precision: int = 12) -> str:
    """Format a constant for C code."""
    suffix = "f" if use_float else ""

    if value == int(value) and abs(value) < 1e10:
        return f"{int(value)}.0{suffix}"

    formatted = f"{value:.{precision}g}"

    # Ensure decimal point for floating point literals
    if '.' not in formatted and 'e' not in formatted.lower():
        formatted += ".0"

    return formatted + suffix


def _emit_expr(node: ExprNode, var_name: str, safe: bool, use_float: bool, macro_style: str) -> str:
    """
    Recursively emit C expression string for an ExprNode.
    """
    if node.node_type == NodeType.CONSTANT:
        return _format_constant(node.value, use_float)

    elif node.node_type == NodeType.VARIABLE:
        return var_name

    elif node.node_type == NodeType.UNARY_OP:
        child = _emit_expr(node.child, var_name, safe, use_float, macro_style)

        # Select function suffix based on precision
        f_suffix = "f" if use_float else ""

        if node.unary_op == UnaryOp.SIN:
            return f"sin{f_suffix}({child})"
        elif node.unary_op == UnaryOp.COS:
            return f"cos{f_suffix}({child})"
        elif node.unary_op == UnaryOp.EXP:
            if safe:
                return f"exp{f_suffix}(clamp_exp_arg{f_suffix}({child}))"
            return f"exp{f_suffix}({child})"
        elif node.unary_op == UnaryOp.LOG:
            if safe:
                return f"safe_log{f_suffix}({child})"
            return f"log{f_suffix}({child})"
        elif node.unary_op == UnaryOp.SQRT:
            if safe:
                return f"sqrt{f_suffix}(fabs{f_suffix}({child}))"
            return f"sqrt{f_suffix}({child})"
        elif node.unary_op == UnaryOp.NEG:
            return f"-({child})"
        elif node.unary_op == UnaryOp.ABS:
            return f"fabs{f_suffix}({child})"
        elif node.unary_op == UnaryOp.SQUARE:
            return f"(({child}) * ({child}))"
        else:
            raise ValueError(f"Unsupported unary operator: {node.unary_op}")

    elif node.node_type == NodeType.BINARY_OP:
        left = _emit_expr(node.left, var_name, safe, use_float, macro_style)
        right = _emit_expr(node.right, var_name, safe, use_float, macro_style)
        f_suffix = "f" if use_float else ""

        if node.binary_op == BinaryOp.ADD:
            return f"({left} + {right})"
        elif node.binary_op == BinaryOp.SUB:
            return f"({left} - {right})"
        elif node.binary_op == BinaryOp.MUL:
            return f"({left} * {right})"
        elif node.binary_op == BinaryOp.DIV:
            if safe:
                return f"safe_div{f_suffix}({left}, {right})"
            return f"({left} / {right})"
        elif node.binary_op == BinaryOp.POW:
            return f"pow{f_suffix}({left}, {right})"
        else:
            raise ValueError(f"Unsupported binary operator: {node.binary_op}")

    elif node.node_type == NodeType.MACRO:
        return _emit_macro(node, var_name, safe, use_float, macro_style)

    raise ValueError(f"Unknown node type: {node.node_type}")


def _emit_macro(node: ExprNode, var_name: str, safe: bool, use_float: bool, macro_style: str) -> str:
    """Emit C code for a macro node."""
    params = node.macro_params
    x = var_name
    f_suffix = "f" if use_float else ""

    def p(i: int) -> str:
        return _format_constant(params[i], use_float)

    # If macro_style is "helpers", emit helper function calls
    if macro_style == "helpers":
        if node.macro_op == MacroOp.DAMPED_SIN:
            return f"damped_sin({p(0)}, {_format_constant(abs(params[1]), use_float)}, {x}, {p(2)}, {p(3)})"
        elif node.macro_op == MacroOp.DAMPED_COS:
            return f"damped_cos({p(0)}, {_format_constant(abs(params[1]), use_float)}, {x}, {p(2)}, {p(3)})"
        elif node.macro_op == MacroOp.RC_CHARGE:
            return f"rc_charge({p(0)}, {_format_constant(abs(params[1]), use_float)}, {x}, {p(2)})"
        elif node.macro_op == MacroOp.EXP_DECAY:
            return f"exp_decay({p(0)}, {_format_constant(abs(params[1]), use_float)}, {x}, {p(2)})"
        elif node.macro_op == MacroOp.SIGMOID:
            return f"sigmoid({p(0)}, {p(1)}, {x}, {p(2)})"
        elif node.macro_op == MacroOp.LOGISTIC:
            return f"logistic({p(0)}, {p(1)}, {p(2)}, {x})"
        elif node.macro_op == MacroOp.HILL:
            return f"hill({p(0)}, {p(1)}, {p(2)}, {x})"
        elif node.macro_op == MacroOp.RATIO:
            return f"ratio({p(0)}, {p(1)}, {p(2)}, {p(3)}, {x})"
        elif node.macro_op == MacroOp.RATIONAL2:
            return f"rational2({p(0)}, {p(1)}, {p(2)}, {p(3)}, {p(4)}, {p(5)}, {x})"
        elif node.macro_op == MacroOp.POWER_LAW:
            return f"power_law({p(0)}, {p(1)}, {x}, {p(2)})"
        elif node.macro_op == MacroOp.GAUSSIAN:
            return f"gaussian({p(0)}, {p(1)}, {p(2)}, {x})"
        elif node.macro_op == MacroOp.TANH_STEP:
            return f"tanh_step({p(0)}, {p(1)}, {p(2)}, {x}, {p(3)})"

    # Inline expansion (default)
    exp_fn = f"exp{f_suffix}"
    clamp = f"clamp_exp_arg{f_suffix}" if safe else ""

    if node.macro_op == MacroOp.DAMPED_SIN:
        k_neg = _format_constant(-abs(params[1]), use_float)
        if safe:
            return f"({p(0)} * {exp_fn}({clamp}({k_neg} * {x})) * sin{f_suffix}({p(2)} * {x} + {p(3)}))"
        return f"({p(0)} * {exp_fn}({k_neg} * {x}) * sin{f_suffix}({p(2)} * {x} + {p(3)}))"

    elif node.macro_op == MacroOp.DAMPED_COS:
        k_neg = _format_constant(-abs(params[1]), use_float)
        if safe:
            return f"({p(0)} * {exp_fn}({clamp}({k_neg} * {x})) * cos{f_suffix}({p(2)} * {x} + {p(3)}))"
        return f"({p(0)} * {exp_fn}({k_neg} * {x}) * cos{f_suffix}({p(2)} * {x} + {p(3)}))"

    elif node.macro_op == MacroOp.RC_CHARGE:
        k_neg = _format_constant(-abs(params[1]), use_float)
        if safe:
            return f"({p(0)} * (1.0{f_suffix} - {exp_fn}({clamp}({k_neg} * {x}))) + {p(2)})"
        return f"({p(0)} * (1.0{f_suffix} - {exp_fn}({k_neg} * {x})) + {p(2)})"

    elif node.macro_op == MacroOp.EXP_DECAY:
        k_neg = _format_constant(-abs(params[1]), use_float)
        if safe:
            return f"({p(0)} * {exp_fn}({clamp}({k_neg} * {x})) + {p(2)})"
        return f"({p(0)} * {exp_fn}({k_neg} * {x}) + {p(2)})"

    elif node.macro_op == MacroOp.SIGMOID:
        if safe:
            return f"({p(0)} / (1.0{f_suffix} + {exp_fn}({clamp}(-{p(1)} * ({x} - {p(2)})))))"
        return f"({p(0)} / (1.0{f_suffix} + {exp_fn}(-{p(1)} * ({x} - {p(2)}))))"

    elif node.macro_op == MacroOp.LOGISTIC:
        b_abs = _format_constant(abs(params[1]), use_float)
        if safe:
            return f"({p(0)} / (1.0{f_suffix} + {b_abs} * {exp_fn}({clamp}(-{p(2)} * {x}))))"
        return f"({p(0)} / (1.0{f_suffix} + {b_abs} * {exp_fn}(-{p(2)} * {x})))"

    elif node.macro_op == MacroOp.HILL:
        k_abs = _format_constant(abs(params[1]) + 1e-10, use_float)
        n_clamped = max(0.5, min(params[2], 5.0))
        n_val = _format_constant(n_clamped, use_float)
        eps = "1e-10f" if use_float else "1e-10"
        x_abs = f"(fabs{f_suffix}({x}) + {eps})"
        return f"({p(0)} * pow{f_suffix}({x_abs}, {n_val}) / (pow{f_suffix}({k_abs}, {n_val}) + pow{f_suffix}({x_abs}, {n_val})))"

    elif node.macro_op == MacroOp.RATIO:
        num = f"({p(0)} * {x} + {p(1)})"
        den = f"({p(2)} * {x} + {p(3)})"
        if safe:
            return f"safe_div{f_suffix}({num}, {den})"
        return f"({num} / {den})"

    elif node.macro_op == MacroOp.RATIONAL2:
        num = f"({p(0)} * {x} * {x} + {p(1)} * {x} + {p(2)})"
        den = f"({p(3)} * {x} * {x} + {p(4)} * {x} + {p(5)})"
        if safe:
            return f"safe_div{f_suffix}({num}, {den})"
        return f"({num} / {den})"

    elif node.macro_op == MacroOp.POWER_LAW:
        eps = "1e-10f" if use_float else "1e-10"
        x_abs = f"(fabs{f_suffix}({x}) + {eps})"
        return f"({p(0)} * pow{f_suffix}({x_abs}, {p(1)}) + {p(2)})"

    elif node.macro_op == MacroOp.GAUSSIAN:
        sigma_abs = _format_constant(abs(params[2]) + 1e-10, use_float)
        z = f"(({x} - {p(1)}) / {sigma_abs})"
        if safe:
            return f"({p(0)} * {exp_fn}({clamp}(-{z} * {z})))"
        return f"({p(0)} * {exp_fn}(-{z} * {z}))"

    elif node.macro_op == MacroOp.TANH_STEP:
        return f"({p(0)} * tanh{f_suffix}({p(1)} * ({x} - {p(2)})) + {p(3)})"

    raise ValueError(f"Unsupported macro: {node.macro_op}")


def _check_needs_macro_helpers(node: ExprNode) -> bool:
    """Check if expression tree contains any macros."""
    if node.node_type == NodeType.MACRO:
        return True
    elif node.node_type == NodeType.UNARY_OP:
        return _check_needs_macro_helpers(node.child)
    elif node.node_type == NodeType.BINARY_OP:
        return _check_needs_macro_helpers(node.left) or _check_needs_macro_helpers(node.right)
    return False


def export_c(
    expr: ExprNode,
    *,
    fn_name: str = "ppf_model",
    signature: Tuple[str, ...] = ("double t",),
    safe: bool = True,
    use_float: bool = False,
    macro_style: str = "inline"
) -> str:
    """
    Generate a standalone C99 function that evaluates the expression.

    Args:
        expr: Expression tree to export
        fn_name: Name for the generated function
        signature: C-style parameter declarations
        safe: Whether to emit safety wrappers
        use_float: Use float instead of double precision
        macro_style: "inline" expands macros, "helpers" emits helper functions

    Returns:
        C code string ready for compilation

    Example:
        >>> code = export_c(my_expr, fn_name="predict", use_float=True)
        >>> # Compile with: gcc -std=c99 -O2 -lm model.c
    """
    if macro_style not in ("inline", "helpers"):
        raise ValueError(f"macro_style must be 'inline' or 'helpers', got '{macro_style}'")

    # Determine variable name from signature and adjust for float mode
    var_name = "t"
    adjusted_signature = list(signature)
    if signature:
        # Extract variable name from "double t" or "float x"
        parts = signature[0].split()
        if len(parts) >= 2:
            var_name = parts[-1]

        # If use_float, convert "double" to "float" in signature
        if use_float:
            adjusted_signature = [s.replace("double", "float") for s in signature]

    # Build expression body
    body_expr = _emit_expr(expr, var_name, safe, use_float, macro_style)

    # Determine return type
    ret_type = "float" if use_float else "double"

    # Build output
    lines = ["#include <math.h>", ""]

    # Add safety helpers if needed
    if safe:
        if use_float:
            lines.append(SAFE_HELPERS_FLOAT)
        else:
            lines.append(SAFE_HELPERS_DOUBLE)

    # Add macro helpers if using helper style
    if macro_style == "helpers" and _check_needs_macro_helpers(expr):
        if safe:
            # Macro helpers depend on clamp_exp_arg which is in safety helpers
            pass
        else:
            # Need clamp_exp_arg even without full safety
            if use_float:
                lines.append("static inline float clamp_exp_argf(float x) {\n    if (x < -60.0f) return -60.0f;\n    if (x > 60.0f) return 60.0f;\n    return x;\n}\n\n")
            else:
                lines.append("static inline double clamp_exp_arg(double x) {\n    if (x < -60.0) return -60.0;\n    if (x > 60.0) return 60.0;\n    return x;\n}\n\n")
        lines.append(MACRO_HELPERS_DOUBLE if not use_float else "")  # TODO: float macro helpers

    # Build function
    sig_str = ", ".join(adjusted_signature)
    lines.append(f"static inline {ret_type} {fn_name}({sig_str}) {{")
    lines.append(f"    return {body_expr};")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)
