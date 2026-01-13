"""
JSON serialization for PPF expressions and model bundles.

Creates portable model bundles for storage, MQTT, REST APIs, etc.
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
import json

from ..symbolic_types import (
    ExprNode, NodeType, UnaryOp, BinaryOp, MacroOp,
    SymbolicFitResult, SymbolicRegressionResult,
)

# Schema version
SCHEMA_VERSION = "ppf.export.model.v1"

# PPF version (would normally come from package metadata)
PPF_VERSION = "0.1.0"

# Macro argument order documentation (from TSD Appendix B)
MACRO_ARG_ORDER = {
    MacroOp.DAMPED_SIN: ["a", "k", "w", "phi"],      # Note: k is decay, w is frequency
    MacroOp.DAMPED_COS: ["a", "k", "w", "phi"],
    MacroOp.RC_CHARGE: ["a", "k", "c"],
    MacroOp.EXP_DECAY: ["a", "k", "c"],
    MacroOp.SIGMOID: ["a", "k", "x0"],
    MacroOp.LOGISTIC: ["a", "b", "k"],
    MacroOp.HILL: ["a", "k", "n"],
    MacroOp.RATIO: ["a", "b", "c", "d"],
    MacroOp.RATIONAL2: ["a", "b", "c", "d", "e", "f"],
    MacroOp.POWER_LAW: ["a", "b", "c"],
    MacroOp.GAUSSIAN: ["a", "mu", "sigma"],
    MacroOp.TANH_STEP: ["a", "k", "x0", "c"],
}


def _serialize_tree(node: ExprNode, var_name: str = "t") -> Dict[str, Any]:
    """
    Serialize an expression tree to a JSON-compatible dict.

    Args:
        node: Expression tree node
        var_name: Variable name to use for VAR nodes

    Returns:
        Dictionary representation of the tree
    """
    if node.node_type == NodeType.CONSTANT:
        return {"type": "CONST", "value": node.value}

    elif node.node_type == NodeType.VARIABLE:
        return {"type": "VAR", "name": var_name}

    elif node.node_type == NodeType.UNARY_OP:
        return {
            "type": "UNARY_OP",
            "op": node.unary_op.name,
            "child": _serialize_tree(node.child, var_name)
        }

    elif node.node_type == NodeType.BINARY_OP:
        return {
            "type": "BINARY_OP",
            "op": node.binary_op.name,
            "children": [
                _serialize_tree(node.left, var_name),
                _serialize_tree(node.right, var_name)
            ]
        }

    elif node.node_type == NodeType.MACRO:
        # Serialize as MACRO_CALL with args array
        args = []
        param_names = MACRO_ARG_ORDER.get(node.macro_op, [])

        for i, val in enumerate(node.macro_params):
            args.append({"type": "CONST", "value": val})

        # Add variable argument (last position for most macros)
        args.append({"type": "VAR", "name": var_name})

        return {
            "type": "MACRO_CALL",
            "name": node.macro_op.name,
            "args": args
        }

    raise ValueError(f"Unknown node type: {node.node_type}")


def _expr_to_latex(node: ExprNode, var_name: str = "t") -> str:
    """
    Convert expression to LaTeX string.

    A basic implementation - could be enhanced for prettier output.
    """
    if node.node_type == NodeType.CONSTANT:
        val = node.value
        if val == int(val) and abs(val) < 1e6:
            return str(int(val))
        return f"{val:.4g}"

    elif node.node_type == NodeType.VARIABLE:
        return var_name

    elif node.node_type == NodeType.UNARY_OP:
        child = _expr_to_latex(node.child, var_name)
        if node.unary_op == UnaryOp.SIN:
            return f"\\sin({child})"
        elif node.unary_op == UnaryOp.COS:
            return f"\\cos({child})"
        elif node.unary_op == UnaryOp.EXP:
            return f"e^{{{child}}}"
        elif node.unary_op == UnaryOp.LOG:
            return f"\\ln({child})"
        elif node.unary_op == UnaryOp.SQRT:
            return f"\\sqrt{{{child}}}"
        elif node.unary_op == UnaryOp.NEG:
            return f"-{child}"
        elif node.unary_op == UnaryOp.ABS:
            return f"|{child}|"
        elif node.unary_op == UnaryOp.SQUARE:
            return f"{child}^2"
        return f"{node.unary_op.name}({child})"

    elif node.node_type == NodeType.BINARY_OP:
        left = _expr_to_latex(node.left, var_name)
        right = _expr_to_latex(node.right, var_name)
        if node.binary_op == BinaryOp.ADD:
            return f"({left} + {right})"
        elif node.binary_op == BinaryOp.SUB:
            return f"({left} - {right})"
        elif node.binary_op == BinaryOp.MUL:
            return f"{left} \\cdot {right}"
        elif node.binary_op == BinaryOp.DIV:
            return f"\\frac{{{left}}}{{{right}}}"
        elif node.binary_op == BinaryOp.POW:
            return f"{left}^{{{right}}}"
        return f"{left} {node.binary_op.name} {right}"

    elif node.node_type == NodeType.MACRO:
        return _macro_to_latex(node, var_name)

    return "?"


def _macro_to_latex(node: ExprNode, var_name: str) -> str:
    """Convert macro to LaTeX representation."""
    params = node.macro_params

    def f(val: float) -> str:
        if val == int(val) and abs(val) < 1e6:
            return str(int(val))
        return f"{val:.4g}"

    t = var_name

    if node.macro_op == MacroOp.DAMPED_SIN:
        a, k, w, phi = params
        return f"{f(a)} e^{{-{f(abs(k))} {t}}} \\sin({f(w)} {t} + {f(phi)})"

    elif node.macro_op == MacroOp.DAMPED_COS:
        a, k, w, phi = params
        return f"{f(a)} e^{{-{f(abs(k))} {t}}} \\cos({f(w)} {t} + {f(phi)})"

    elif node.macro_op == MacroOp.RC_CHARGE:
        a, k, c = params
        return f"{f(a)} (1 - e^{{-{f(abs(k))} {t}}}) + {f(c)}"

    elif node.macro_op == MacroOp.EXP_DECAY:
        a, k, c = params
        return f"{f(a)} e^{{-{f(abs(k))} {t}}} + {f(c)}"

    elif node.macro_op == MacroOp.SIGMOID:
        a, k, x0 = params
        return f"\\frac{{{f(a)}}}{{1 + e^{{-{f(k)}({t} - {f(x0)})}}}}"

    elif node.macro_op == MacroOp.LOGISTIC:
        a, b, k = params
        return f"\\frac{{{f(a)}}}{{1 + {f(abs(b))} e^{{-{f(k)} {t}}}}}"

    elif node.macro_op == MacroOp.HILL:
        a, k, n = params
        return f"\\frac{{{f(a)} {t}^{{{f(n)}}}}}{{{f(abs(k))}^{{{f(n)}}} + {t}^{{{f(n)}}}}}"

    elif node.macro_op == MacroOp.RATIO:
        a, b, c, d = params
        return f"\\frac{{{f(a)} {t} + {f(b)}}}{{{f(c)} {t} + {f(d)}}}"

    elif node.macro_op == MacroOp.RATIONAL2:
        a, b, c, d, e, f_ = params
        return f"\\frac{{{f(a)} {t}^2 + {f(b)} {t} + {f(c)}}}{{{f(d)} {t}^2 + {f(e)} {t} + {f(f_)}}}"

    elif node.macro_op == MacroOp.POWER_LAW:
        a, b, c = params
        return f"{f(a)} {t}^{{{f(b)}}} + {f(c)}"

    elif node.macro_op == MacroOp.GAUSSIAN:
        a, mu, sigma = params
        return f"{f(a)} e^{{-\\left(\\frac{{{t} - {f(mu)}}}{{{f(abs(sigma))}}}\\right)^2}}"

    elif node.macro_op == MacroOp.TANH_STEP:
        a, k, x0, c = params
        return f"{f(a)} \\tanh({f(k)}({t} - {f(x0)})) + {f(c)}"

    return f"{node.macro_op.value}(...)"


def _extract_named_parameters(expr: ExprNode) -> Dict[str, float]:
    """
    Extract named parameters from expression if it's a macro.

    For non-macro expressions, returns empty dict.
    """
    if expr.node_type != NodeType.MACRO:
        return {}

    params = expr.macro_params
    names = MACRO_ARG_ORDER.get(expr.macro_op, [])

    result = {}
    for i, name in enumerate(names):
        if i < len(params):
            result[name] = params[i]

    return result


def export_json(
    model: Union[SymbolicFitResult, SymbolicRegressionResult],
    *,
    include_expr_tree: bool = True,
    include_source: str = "ppf",
    version: Optional[str] = None,
    variables: List[str] = None
) -> Dict[str, Any]:
    """
    Create a portable model bundle for storage or transmission.

    Args:
        model: Result object from symbolic regression
        include_expr_tree: Include full tree serialization
        include_source: Metadata source tag
        version: Version string (defaults to PPF version)
        variables: Variable names (defaults to ["t"])

    Returns:
        Dictionary ready for JSON serialization

    Example:
        >>> bundle = export_json(result)
        >>> json.dumps(bundle)  # For transmission
        >>> # Or save to file
        >>> with open("model.json", "w") as f:
        ...     json.dump(bundle, f, indent=2)
    """
    if variables is None:
        variables = ["t"]
    var_name = variables[0] if variables else "t"

    # Handle both result types
    if isinstance(model, SymbolicRegressionResult):
        fit_result = model.best_tradeoff
        metadata = model.metadata.copy() if model.metadata else {}
    else:
        fit_result = model
        metadata = {}

    if fit_result is None:
        raise ValueError("Model has no fit result to export")

    expr = fit_result.expression

    # Build expression section
    expression_data = {
        "string": expr.to_string().replace("x", var_name),
        "latex": _expr_to_latex(expr, var_name),
    }

    if include_expr_tree:
        expression_data["tree"] = _serialize_tree(expr, var_name)

    # Build bundle
    bundle = {
        "schema": SCHEMA_VERSION,
        "source": include_source,
        "ppf_version": version or PPF_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),

        "metadata": metadata,

        "metrics": {
            "r2": fit_result.r_squared,
            "rmse": float(fit_result.mse ** 0.5) if fit_result.mse else None,
            "complexity": fit_result.complexity,
        },

        "variables": variables,

        "expression": expression_data,

        "parameters": _extract_named_parameters(expr),

        "constraints": {
            "safe": True,
            "div_epsilon": 1e-12,
            "exp_clip": [-60, 60],
            "log_epsilon": 1e-12,
        }
    }

    return bundle


def bundle_to_json_string(bundle: Dict[str, Any], indent: int = 2) -> str:
    """
    Convert bundle dict to JSON string with consistent formatting.

    Args:
        bundle: Model bundle dictionary
        indent: JSON indentation level

    Returns:
        JSON string
    """
    return json.dumps(bundle, indent=indent, sort_keys=False)
