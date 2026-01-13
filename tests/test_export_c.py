"""
Tests for C code generation from PPF expressions.
"""

import pytest
import numpy as np

from ppf import (
    ExprNode, NodeType, UnaryOp, BinaryOp, MacroOp,
    export_c,
)


class TestCExportBasic:
    """Test basic expression export to C."""

    def test_constant(self):
        """Export a constant node."""
        expr = ExprNode(NodeType.CONSTANT, value=3.14)
        code = export_c(expr, fn_name="f", safe=False)

        assert "#include <math.h>" in code
        assert "static inline double f(double t)" in code
        assert "3.14" in code
        assert "return" in code

    def test_variable(self):
        """Export a variable node."""
        expr = ExprNode(NodeType.VARIABLE)
        code = export_c(expr, fn_name="f", safe=False)

        assert "return t;" in code

    def test_sin(self):
        """Export sin(x)."""
        expr = ExprNode(
            NodeType.UNARY_OP,
            unary_op=UnaryOp.SIN,
            child=ExprNode(NodeType.VARIABLE)
        )
        code = export_c(expr, fn_name="f", safe=False)

        assert "sin(t)" in code

    def test_binary_add(self):
        """Export x + 1."""
        expr = ExprNode(
            NodeType.BINARY_OP,
            binary_op=BinaryOp.ADD,
            left=ExprNode(NodeType.VARIABLE),
            right=ExprNode(NodeType.CONSTANT, value=1.0)
        )
        code = export_c(expr, fn_name="f", safe=False)

        assert "(t + 1.0)" in code

    def test_binary_mul(self):
        """Export 2 * x."""
        expr = ExprNode(
            NodeType.BINARY_OP,
            binary_op=BinaryOp.MUL,
            left=ExprNode(NodeType.CONSTANT, value=2.0),
            right=ExprNode(NodeType.VARIABLE)
        )
        code = export_c(expr, fn_name="f", safe=False)

        assert "(2.0 * t)" in code


class TestCExportSafety:
    """Test safety wrappers in C export."""

    def test_safe_helpers_included(self):
        """Safe mode includes helper functions."""
        expr = ExprNode(NodeType.VARIABLE)
        code = export_c(expr, safe=True)

        assert "safe_div" in code
        assert "safe_log" in code
        assert "clamp_exp_arg" in code

    def test_safe_div_used(self):
        """Division uses safe_div when safe=True."""
        expr = ExprNode(
            NodeType.BINARY_OP,
            binary_op=BinaryOp.DIV,
            left=ExprNode(NodeType.VARIABLE),
            right=ExprNode(NodeType.CONSTANT, value=2.0)
        )
        code = export_c(expr, safe=True)

        assert "safe_div(t, 2.0)" in code

    def test_unsafe_div(self):
        """Division without safe wrapper when safe=False."""
        expr = ExprNode(
            NodeType.BINARY_OP,
            binary_op=BinaryOp.DIV,
            left=ExprNode(NodeType.VARIABLE),
            right=ExprNode(NodeType.CONSTANT, value=2.0)
        )
        code = export_c(expr, safe=False)

        assert "safe_div" not in code
        assert "(t / 2.0)" in code


class TestCExportFloat:
    """Test float precision variant."""

    def test_float_type(self):
        """Float mode uses float type."""
        expr = ExprNode(NodeType.VARIABLE)
        code = export_c(expr, fn_name="f", use_float=True, safe=False)

        assert "static inline float f(float t)" in code

    def test_float_math_functions(self):
        """Float mode uses float math functions."""
        expr = ExprNode(
            NodeType.UNARY_OP,
            unary_op=UnaryOp.SIN,
            child=ExprNode(NodeType.VARIABLE)
        )
        code = export_c(expr, use_float=True, safe=False)

        assert "sinf(t)" in code

    def test_float_constants(self):
        """Float mode uses f suffix on constants."""
        expr = ExprNode(NodeType.CONSTANT, value=3.14)
        code = export_c(expr, use_float=True, safe=False)

        # Should have f suffix
        assert "3.14" in code


class TestCExportMacros:
    """Test macro export to C."""

    def test_damped_sin_inline(self):
        """Export damped sine macro with inline expansion."""
        expr = ExprNode(
            NodeType.MACRO,
            macro_op=MacroOp.DAMPED_SIN,
            macro_params=[1.0, 0.5, 2.0, 0.0]
        )
        code = export_c(expr, safe=True, macro_style="inline")

        assert "exp(" in code
        assert "sin(" in code
        assert "clamp_exp_arg" in code

    def test_gaussian_inline(self):
        """Export Gaussian macro with inline expansion."""
        expr = ExprNode(
            NodeType.MACRO,
            macro_op=MacroOp.GAUSSIAN,
            macro_params=[1.0, 0.0, 1.0]
        )
        code = export_c(expr, safe=True, macro_style="inline")

        assert "exp(" in code

    def test_power_law_inline(self):
        """Export power law macro with inline expansion."""
        expr = ExprNode(
            NodeType.MACRO,
            macro_op=MacroOp.POWER_LAW,
            macro_params=[2.0, 3.0, 1.0]
        )
        code = export_c(expr, safe=True, macro_style="inline")

        assert "pow(" in code


class TestCExportMacroHelpers:
    """Test macro helper function style."""

    def test_macro_helpers_style(self):
        """Macro helpers style emits function calls."""
        expr = ExprNode(
            NodeType.MACRO,
            macro_op=MacroOp.DAMPED_SIN,
            macro_params=[1.0, 0.5, 2.0, 0.0]
        )
        code = export_c(expr, safe=True, macro_style="helpers")

        assert "damped_sin(" in code


class TestCExportDeterminism:
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

        code1 = export_c(expr, fn_name="f", safe=True)
        code2 = export_c(expr, fn_name="f", safe=True)

        assert code1 == code2


class TestCExportSignature:
    """Test custom function signatures."""

    def test_custom_name(self):
        """Export with custom function name."""
        expr = ExprNode(NodeType.VARIABLE)
        code = export_c(expr, fn_name="predict_temp", safe=False)

        assert "predict_temp" in code

    def test_custom_signature(self):
        """Export with custom signature."""
        expr = ExprNode(NodeType.VARIABLE)
        code = export_c(
            expr,
            fn_name="f",
            signature=("double x",),
            safe=False
        )

        assert "f(double x)" in code
        assert "return x;" in code


class TestCExportValidation:
    """Test validation and error handling."""

    def test_invalid_macro_style(self):
        """Invalid macro_style raises error."""
        expr = ExprNode(NodeType.VARIABLE)

        with pytest.raises(ValueError) as exc_info:
            export_c(expr, macro_style="invalid")

        assert "macro_style" in str(exc_info.value)
