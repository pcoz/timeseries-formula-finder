"""
Utility functions for symbolic regression.

Provides formatted output and expression simplification.
"""

from typing import Optional
import numpy as np

from .symbolic_types import (
    ExprNode, NodeType, UnaryOp, BinaryOp,
    SymbolicRegressionResult, SymbolicFitResult,
)


def print_symbolic_result(
    result: SymbolicRegressionResult,
    show_pareto: bool = True,
    max_pareto_display: int = 5
) -> None:
    """
    Print a formatted symbolic regression result.

    Args:
        result: The SymbolicRegressionResult to display
        show_pareto: Whether to show Pareto front entries
        max_pareto_display: Maximum number of Pareto entries to show
    """
    print("=" * 70)
    print("SYMBOLIC REGRESSION RESULT")
    print("=" * 70)
    print()

    if result.best_tradeoff:
        print("BEST TRADEOFF (accuracy vs complexity):")
        _print_fit_result(result.best_tradeoff, indent=2)
        print()

    if result.most_accurate and result.most_accurate != result.best_tradeoff:
        print("MOST ACCURATE:")
        _print_fit_result(result.most_accurate, indent=2)
        print()

    if result.most_parsimonious and result.most_parsimonious != result.best_tradeoff:
        print("MOST PARSIMONIOUS (simplest):")
        _print_fit_result(result.most_parsimonious, indent=2)
        print()

    if show_pareto and result.pareto_front:
        print(f"PARETO FRONT ({len(result.pareto_front)} solutions):")
        print("-" * 70)

        # Sort by complexity for display
        sorted_front = sorted(result.pareto_front, key=lambda r: r.complexity)

        for i, fit in enumerate(sorted_front[:max_pareto_display]):
            print(f"  [{i+1}] Complexity: {fit.complexity:2d} | "
                  f"R^2: {fit.r_squared:.4f} | {fit.expression_string}")

        if len(sorted_front) > max_pareto_display:
            print(f"  ... and {len(sorted_front) - max_pareto_display} more solutions")

        print()

    print("-" * 70)
    print(f"Generations: {result.generations_run}")
    print(f"Total evaluations: {result.total_evaluations:,}")
    print("=" * 70)


def _print_fit_result(fit: SymbolicFitResult, indent: int = 0) -> None:
    """Print a single fit result with indentation"""
    prefix = " " * indent
    print(f"{prefix}Expression: {fit.expression_string}")
    print(f"{prefix}R-squared:  {fit.r_squared:.6f}")
    print(f"{prefix}MSE:        {fit.mse:.6e}")
    print(f"{prefix}Complexity: {fit.complexity} nodes")
    print(f"{prefix}Noise-like residuals: {fit.is_noise_like}")


def simplify_expression(node: ExprNode) -> ExprNode:
    """
    Apply basic algebraic simplification to an expression tree.

    Simplifications include:
    - x + 0 -> x
    - x * 1 -> x
    - x * 0 -> 0
    - x - 0 -> x
    - 0 - x -> -x
    - x / 1 -> x
    - Constant folding (evaluate constant subtrees)

    Args:
        node: Expression tree to simplify

    Returns:
        Simplified expression tree (new copy)
    """
    node = node.copy()
    return _simplify_recursive(node)


def _simplify_recursive(node: ExprNode) -> ExprNode:
    """Recursive simplification helper"""

    if node.node_type in (NodeType.CONSTANT, NodeType.VARIABLE):
        return node

    if node.node_type == NodeType.UNARY_OP:
        # First simplify child
        node.child = _simplify_recursive(node.child)

        # If child is constant, evaluate
        if node.child.node_type == NodeType.CONSTANT:
            try:
                x = np.array([node.child.value])
                result = node.evaluate(x)[0]
                if np.isfinite(result):
                    return ExprNode(NodeType.CONSTANT, value=float(result))
            except Exception:
                pass

        # Double negation: -(-x) -> x
        if node.unary_op == UnaryOp.NEG:
            if (node.child.node_type == NodeType.UNARY_OP and
                node.child.unary_op == UnaryOp.NEG):
                return node.child.child

        return node

    if node.node_type == NodeType.BINARY_OP:
        # First simplify children
        node.left = _simplify_recursive(node.left)
        node.right = _simplify_recursive(node.right)

        left_const = (node.left.node_type == NodeType.CONSTANT)
        right_const = (node.right.node_type == NodeType.CONSTANT)

        # If both constants, evaluate
        if left_const and right_const:
            try:
                x = np.array([0.0])  # Dummy value
                result = node.evaluate(x)[0]
                if np.isfinite(result):
                    return ExprNode(NodeType.CONSTANT, value=float(result))
            except Exception:
                pass

        # Addition simplifications
        if node.binary_op == BinaryOp.ADD:
            # x + 0 -> x
            if right_const and node.right.value == 0:
                return node.left
            # 0 + x -> x
            if left_const and node.left.value == 0:
                return node.right

        # Subtraction simplifications
        if node.binary_op == BinaryOp.SUB:
            # x - 0 -> x
            if right_const and node.right.value == 0:
                return node.left
            # 0 - x -> -x
            if left_const and node.left.value == 0:
                return ExprNode(
                    NodeType.UNARY_OP,
                    unary_op=UnaryOp.NEG,
                    child=node.right
                )

        # Multiplication simplifications
        if node.binary_op == BinaryOp.MUL:
            # x * 1 -> x
            if right_const and node.right.value == 1:
                return node.left
            # 1 * x -> x
            if left_const and node.left.value == 1:
                return node.right
            # x * 0 -> 0
            if right_const and node.right.value == 0:
                return ExprNode(NodeType.CONSTANT, value=0.0)
            # 0 * x -> 0
            if left_const and node.left.value == 0:
                return ExprNode(NodeType.CONSTANT, value=0.0)

        # Division simplifications
        if node.binary_op == BinaryOp.DIV:
            # x / 1 -> x
            if right_const and node.right.value == 1:
                return node.left
            # 0 / x -> 0 (when x != 0)
            if left_const and node.left.value == 0:
                return ExprNode(NodeType.CONSTANT, value=0.0)

        # Power simplifications
        if node.binary_op == BinaryOp.POW:
            # x ^ 1 -> x
            if right_const and node.right.value == 1:
                return node.left
            # x ^ 0 -> 1
            if right_const and node.right.value == 0:
                return ExprNode(NodeType.CONSTANT, value=1.0)

        return node

    return node


def format_expression_latex(node: ExprNode) -> str:
    """
    Convert expression tree to LaTeX format.

    Args:
        node: Expression tree

    Returns:
        LaTeX string representation
    """
    if node.node_type == NodeType.CONSTANT:
        if node.value == int(node.value):
            return str(int(node.value))
        return f"{node.value:.4g}"

    if node.node_type == NodeType.VARIABLE:
        return "x"

    if node.node_type == NodeType.UNARY_OP:
        child = format_expression_latex(node.child)

        latex_map = {
            UnaryOp.SIN: f"\\sin({child})",
            UnaryOp.COS: f"\\cos({child})",
            UnaryOp.EXP: f"e^{{{child}}}",
            UnaryOp.LOG: f"\\log({child})",
            UnaryOp.SQRT: f"\\sqrt{{{child}}}",
            UnaryOp.NEG: f"-{child}",
            UnaryOp.ABS: f"|{child}|",
            UnaryOp.SQUARE: f"({child})^2",
        }
        return latex_map.get(node.unary_op, f"?({child})")

    if node.node_type == NodeType.BINARY_OP:
        left = format_expression_latex(node.left)
        right = format_expression_latex(node.right)

        if node.binary_op == BinaryOp.ADD:
            return f"({left} + {right})"
        elif node.binary_op == BinaryOp.SUB:
            return f"({left} - {right})"
        elif node.binary_op == BinaryOp.MUL:
            return f"{left} \\cdot {right}"
        elif node.binary_op == BinaryOp.DIV:
            return f"\\frac{{{left}}}{{{right}}}"
        elif node.binary_op == BinaryOp.POW:
            return f"({left})^{{{right}}}"

    return "?"


def expression_complexity_breakdown(node: ExprNode) -> dict:
    """
    Get a breakdown of expression complexity by operator type.

    Args:
        node: Expression tree

    Returns:
        Dict with counts of each operator type
    """
    counts = {
        'constants': 0,
        'variables': 0,
        'unary_ops': {},
        'binary_ops': {},
        'total_nodes': 0,
        'depth': node.depth(),
    }

    def count_nodes(n):
        counts['total_nodes'] += 1

        if n.node_type == NodeType.CONSTANT:
            counts['constants'] += 1
        elif n.node_type == NodeType.VARIABLE:
            counts['variables'] += 1
        elif n.node_type == NodeType.UNARY_OP:
            op_name = n.unary_op.value
            counts['unary_ops'][op_name] = counts['unary_ops'].get(op_name, 0) + 1
            count_nodes(n.child)
        elif n.node_type == NodeType.BINARY_OP:
            op_name = n.binary_op.value
            counts['binary_ops'][op_name] = counts['binary_ops'].get(op_name, 0) + 1
            count_nodes(n.left)
            count_nodes(n.right)

    count_nodes(node)
    return counts
