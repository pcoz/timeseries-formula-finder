"""
Tests for Python code generation from PPF expressions.
"""

import pytest
import numpy as np
import math

from ppf import (
    ExprNode, NodeType, UnaryOp, BinaryOp, MacroOp,
    export_python,
)


class TestPythonExportBasic:
    """Test basic expression export to Python."""

    def test_constant(self):
        """Export a constant node."""
        expr = ExprNode(NodeType.CONSTANT, value=3.14)
        code = export_python(expr, fn_name="f", safe=False)

        assert "def f(t):" in code
        assert "3.14" in code

        # Execute and verify
        exec(code, globals())
        assert abs(f(0) - 3.14) < 1e-10

    def test_variable(self):
        """Export a variable node."""
        expr = ExprNode(NodeType.VARIABLE)
        code = export_python(expr, fn_name="f", safe=False)

        exec(code, globals())
        assert f(5.0) == 5.0

    def test_sin(self):
        """Export sin(x)."""
        expr = ExprNode(
            NodeType.UNARY_OP,
            unary_op=UnaryOp.SIN,
            child=ExprNode(NodeType.VARIABLE)
        )
        code = export_python(expr, fn_name="f", safe=False)

        exec(code, globals())
        assert abs(f(np.pi / 2) - 1.0) < 1e-10

    def test_binary_add(self):
        """Export x + 1."""
        expr = ExprNode(
            NodeType.BINARY_OP,
            binary_op=BinaryOp.ADD,
            left=ExprNode(NodeType.VARIABLE),
            right=ExprNode(NodeType.CONSTANT, value=1.0)
        )
        code = export_python(expr, fn_name="f", safe=False)

        exec(code, globals())
        assert f(5.0) == 6.0

    def test_binary_mul(self):
        """Export 2 * x."""
        expr = ExprNode(
            NodeType.BINARY_OP,
            binary_op=BinaryOp.MUL,
            left=ExprNode(NodeType.CONSTANT, value=2.0),
            right=ExprNode(NodeType.VARIABLE)
        )
        code = export_python(expr, fn_name="f", safe=False)

        exec(code, globals())
        assert f(3.0) == 6.0


class TestPythonExportSafety:
    """Test safety wrappers in Python export."""

    def test_safe_div_nonzero(self):
        """Safe division with non-zero denominator."""
        # x / 2
        expr = ExprNode(
            NodeType.BINARY_OP,
            binary_op=BinaryOp.DIV,
            left=ExprNode(NodeType.VARIABLE),
            right=ExprNode(NodeType.CONSTANT, value=2.0)
        )
        code = export_python(expr, fn_name="f", safe=True)

        exec(code, globals())
        assert f(4.0) == 2.0

    def test_safe_div_zero(self):
        """Safe division by zero doesn't crash."""
        # 1 / x at x=0
        expr = ExprNode(
            NodeType.BINARY_OP,
            binary_op=BinaryOp.DIV,
            left=ExprNode(NodeType.CONSTANT, value=1.0),
            right=ExprNode(NodeType.VARIABLE)
        )
        code = export_python(expr, fn_name="f", safe=True)

        exec(code, globals())
        result = f(0.0)
        assert np.isfinite(result)  # Should not be inf or nan

    def test_safe_log_positive(self):
        """Safe log with positive input."""
        expr = ExprNode(
            NodeType.UNARY_OP,
            unary_op=UnaryOp.LOG,
            child=ExprNode(NodeType.VARIABLE)
        )
        code = export_python(expr, fn_name="f", safe=True)

        exec(code, globals())
        assert abs(f(np.e) - 1.0) < 1e-10

    def test_safe_log_zero(self):
        """Safe log of zero doesn't crash."""
        expr = ExprNode(
            NodeType.UNARY_OP,
            unary_op=UnaryOp.LOG,
            child=ExprNode(NodeType.VARIABLE)
        )
        code = export_python(expr, fn_name="f", safe=True)

        exec(code, globals())
        result = f(0.0)
        assert np.isfinite(result)

    def test_clamp_exp_large(self):
        """Clamped exp doesn't overflow."""
        expr = ExprNode(
            NodeType.UNARY_OP,
            unary_op=UnaryOp.EXP,
            child=ExprNode(NodeType.VARIABLE)
        )
        code = export_python(expr, fn_name="f", safe=True)

        exec(code, globals())
        result = f(1000.0)  # Would overflow without clamping
        assert np.isfinite(result)


class TestPythonExportMacros:
    """Test macro export to Python."""

    def test_damped_sin(self):
        """Export damped sine macro."""
        expr = ExprNode(
            NodeType.MACRO,
            macro_op=MacroOp.DAMPED_SIN,
            macro_params=[1.0, 0.5, 2.0, 0.0]  # a, k, w, phi
        )
        code = export_python(expr, fn_name="f", safe=True)

        exec(code, globals())

        # At t=0: 1.0 * exp(0) * sin(0) = 0
        assert abs(f(0.0)) < 1e-10

        # Check it runs without error at various points
        for t in [0.5, 1.0, 2.0, 5.0]:
            assert np.isfinite(f(t))

    def test_gaussian(self):
        """Export Gaussian macro."""
        expr = ExprNode(
            NodeType.MACRO,
            macro_op=MacroOp.GAUSSIAN,
            macro_params=[1.0, 0.0, 1.0]  # a=1, mu=0, sigma=1
        )
        code = export_python(expr, fn_name="f", safe=True)

        exec(code, globals())

        # At x=0 (center): a * exp(0) = 1
        assert abs(f(0.0) - 1.0) < 1e-10

        # At x=1 (1 sigma): exp(-1) ≈ 0.368
        assert abs(f(1.0) - np.exp(-1)) < 1e-10

    def test_power_law(self):
        """Export power law macro."""
        expr = ExprNode(
            NodeType.MACRO,
            macro_op=MacroOp.POWER_LAW,
            macro_params=[2.0, 3.0, 1.0]  # a=2, b=3, c=1
        )
        code = export_python(expr, fn_name="f", safe=True)

        exec(code, globals())

        # At x=2: 2 * 2^3 + 1 = 17
        assert abs(f(2.0) - 17.0) < 1e-8


class TestPythonExportDeterminism:
    """Test deterministic output."""

    def test_same_expr_same_code(self):
        """Same expression produces identical code."""
        expr = ExprNode(
            NodeType.BINARY_OP,
            binary_op=BinaryOp.MUL,
            left=ExprNode(NodeType.CONSTANT, value=2.5),
            right=ExprNode(
                NodeType.UNARY_OP,
                unary_op=UnaryOp.SIN,
                child=ExprNode(NodeType.VARIABLE)
            )
        )

        code1 = export_python(expr, fn_name="f", safe=True)
        code2 = export_python(expr, fn_name="f", safe=True)

        assert code1 == code2


class TestPythonExportSignature:
    """Test custom function signatures."""

    def test_custom_name(self):
        """Export with custom function name."""
        expr = ExprNode(NodeType.VARIABLE)
        code = export_python(expr, fn_name="predict_temperature", safe=False)

        assert "def predict_temperature(t):" in code

    def test_custom_variable(self):
        """Export with custom variable name."""
        expr = ExprNode(NodeType.VARIABLE)
        code = export_python(expr, fn_name="f", signature=("x",), safe=False)

        assert "def f(x):" in code

        exec(code, globals())
        assert f(5.0) == 5.0
