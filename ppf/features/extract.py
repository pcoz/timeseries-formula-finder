"""
Feature extraction from PPF discovery results.

Converts symbolic regression results into interpretable feature
dictionaries for downstream ML pipelines.
"""

from typing import Dict, Any, Optional, List
import numpy as np

from ..symbolic_types import (
    ExprNode, NodeType, UnaryOp, BinaryOp, MacroOp,
    SymbolicFitResult, SymbolicRegressionResult,
)


# Dominant family classification
OSCILLATION_MACROS = {MacroOp.DAMPED_SIN, MacroOp.DAMPED_COS}
GROWTH_MACROS = {MacroOp.SIGMOID, MacroOp.LOGISTIC, MacroOp.HILL, MacroOp.RC_CHARGE}
DECAY_MACROS = {MacroOp.EXP_DECAY}
RATIO_MACROS = {MacroOp.RATIO, MacroOp.RATIONAL2}
PEAK_MACROS = {MacroOp.GAUSSIAN, MacroOp.TANH_STEP, MacroOp.POWER_LAW}


def _classify_dominant_family(expr: ExprNode) -> str:
    """
    Classify the dominant functional family of an expression.

    Returns one of: "oscillation", "growth", "saturation", "ratio", "peaks", "algebraic"
    """
    # Check for macro at root level
    if expr.node_type == NodeType.MACRO:
        if expr.macro_op in OSCILLATION_MACROS:
            return "oscillation"
        elif expr.macro_op in GROWTH_MACROS:
            return "saturation"
        elif expr.macro_op in DECAY_MACROS:
            return "decay"
        elif expr.macro_op in RATIO_MACROS:
            return "ratio"
        elif expr.macro_op in PEAK_MACROS:
            return "peaks"

    # Check for oscillatory operators in tree
    if _contains_ops(expr, {UnaryOp.SIN, UnaryOp.COS}):
        return "oscillation"

    # Check for exponential/growth patterns
    if _contains_ops(expr, {UnaryOp.EXP}):
        if _contains_ops(expr, {BinaryOp.DIV}):
            return "saturation"
        return "growth"

    # Check for rational patterns
    if _contains_binary_op(expr, BinaryOp.DIV):
        return "ratio"

    return "algebraic"


def _contains_ops(expr: ExprNode, ops: set) -> bool:
    """Check if expression contains any of the given unary operators."""
    if expr.node_type == NodeType.UNARY_OP:
        if expr.unary_op in ops:
            return True
        return _contains_ops(expr.child, ops)
    elif expr.node_type == NodeType.BINARY_OP:
        return _contains_ops(expr.left, ops) or _contains_ops(expr.right, ops)
    return False


def _contains_binary_op(expr: ExprNode, op: BinaryOp) -> bool:
    """Check if expression contains a specific binary operator."""
    if expr.node_type == NodeType.BINARY_OP:
        if expr.binary_op == op:
            return True
        return _contains_binary_op(expr.left, op) or _contains_binary_op(expr.right, op)
    elif expr.node_type == NodeType.UNARY_OP:
        return _contains_binary_op(expr.child, op)
    return False


def _extract_oscillator_params(expr: ExprNode) -> Dict[str, float]:
    """Extract oscillator-specific parameters."""
    params = {}

    if expr.node_type == NodeType.MACRO:
        if expr.macro_op == MacroOp.DAMPED_SIN:
            a, k, w, phi = expr.macro_params
            params["amplitude"] = abs(a)
            params["damping_k"] = abs(k)
            params["omega"] = w
            params["phase"] = phi
            # Angular frequency to Hz (assuming x is time in some unit)
            params["freq_hz"] = abs(w) / (2 * np.pi) if w else 0

        elif expr.macro_op == MacroOp.DAMPED_COS:
            a, k, w, phi = expr.macro_params
            params["amplitude"] = abs(a)
            params["damping_k"] = abs(k)
            params["omega"] = w
            params["phase"] = phi
            params["freq_hz"] = abs(w) / (2 * np.pi) if w else 0

    return params


def _extract_logistic_params(expr: ExprNode) -> Dict[str, float]:
    """Extract logistic/sigmoid-specific parameters."""
    params = {}

    if expr.node_type == NodeType.MACRO:
        if expr.macro_op == MacroOp.SIGMOID:
            a, k, x0 = expr.macro_params
            params["K"] = a  # Carrying capacity / upper asymptote
            params["r"] = k  # Growth rate
            params["t0"] = x0  # Midpoint

        elif expr.macro_op == MacroOp.LOGISTIC:
            a, b, k = expr.macro_params
            params["K"] = a
            params["r"] = k

        elif expr.macro_op == MacroOp.HILL:
            a, k, n = expr.macro_params
            params["K"] = a  # Max value
            params["Km"] = abs(k)  # Half-max constant
            params["n"] = n  # Hill coefficient

    return params


def _extract_rational_params(expr: ExprNode) -> Dict[str, float]:
    """Extract rational function parameters."""
    params = {}

    if expr.node_type == NodeType.MACRO:
        if expr.macro_op == MacroOp.RATIO:
            params["numerator_degree"] = 1
            params["denominator_degree"] = 1
            a, b, c, d = expr.macro_params
            # DC gain estimate (at x=0): b/d
            if abs(d) > 1e-10:
                params["gain_estimate"] = b / d

        elif expr.macro_op == MacroOp.RATIONAL2:
            params["numerator_degree"] = 2
            params["denominator_degree"] = 2
            a, b, c, d, e, f = expr.macro_params
            if abs(f) > 1e-10:
                params["gain_estimate"] = c / f

    return params


def _extract_gaussian_params(expr: ExprNode) -> Dict[str, float]:
    """Extract Gaussian/peak parameters."""
    params = {}

    if expr.node_type == NodeType.MACRO:
        if expr.macro_op == MacroOp.GAUSSIAN:
            a, mu, sigma = expr.macro_params
            params["amplitude"] = abs(a)
            params["mu"] = mu
            params["sigma"] = abs(sigma)

        elif expr.macro_op == MacroOp.POWER_LAW:
            a, b, c = expr.macro_params
            params["amplitude"] = abs(a)
            params["exponent"] = b
            params["offset"] = c

    return params


def _compute_residual_stats(residuals: np.ndarray) -> Dict[str, float]:
    """Compute residual statistics."""
    if residuals is None or len(residuals) == 0:
        return {}

    return {
        "residual_rms": float(np.sqrt(np.mean(residuals ** 2))),
        "residual_mad": float(np.median(np.abs(residuals - np.median(residuals)))),
        "residual_max_abs": float(np.max(np.abs(residuals))),
    }


def extract_features(
    result: SymbolicRegressionResult | SymbolicFitResult,
    *,
    include_residual_stats: bool = True,
    include_domain_scores: bool = True
) -> Dict[str, Any]:
    """
    Convert a discovery result into a stable, interpretable feature dictionary.

    Args:
        result: Symbolic regression or fit result object
        include_residual_stats: Include residual statistics
        include_domain_scores: Include domain-specific scores from probing

    Returns:
        Dictionary of features suitable for downstream ML

    Example:
        >>> features = extract_features(result)
        >>> features["dominant_family"]  # "oscillation", "growth", etc.
        >>> features["r2"]  # R-squared value
    """
    # Handle both result types
    if isinstance(result, SymbolicRegressionResult):
        fit_result = result.best_tradeoff
        metadata = result.metadata or {}
    else:
        fit_result = result
        metadata = {}

    if fit_result is None:
        raise ValueError("Result has no fit result to extract features from")

    expr = fit_result.expression

    # Common features (always present)
    features = {
        "mode_chosen": metadata.get("mode", "unknown"),
        "r2": fit_result.r_squared,
        "rmse": float(fit_result.mse ** 0.5) if fit_result.mse else 0.0,
        "complexity": fit_result.complexity,
        "dominant_family": _classify_dominant_family(expr),
    }

    # Residual stats
    if include_residual_stats and fit_result.residuals is not None:
        features.update(_compute_residual_stats(fit_result.residuals))

    # Domain-specific parameters based on family
    family = features["dominant_family"]

    if family == "oscillation":
        osc_params = _extract_oscillator_params(expr)
        features.update(osc_params)

    elif family in ("saturation", "growth"):
        log_params = _extract_logistic_params(expr)
        features.update(log_params)

    elif family == "ratio":
        rat_params = _extract_rational_params(expr)
        features.update(rat_params)

    elif family == "peaks":
        peak_params = _extract_gaussian_params(expr)
        features.update(peak_params)

    # Domain scores from auto-discovery probing
    if include_domain_scores and "probe_results" in metadata:
        for domain, r2 in metadata["probe_results"].items():
            features[f"domain_score_{domain}"] = r2

    return features
