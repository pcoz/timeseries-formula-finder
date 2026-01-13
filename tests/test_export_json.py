"""
Tests for JSON serialization of PPF expressions.
"""

import pytest
import json
import numpy as np

from ppf import (
    ExprNode, NodeType, UnaryOp, BinaryOp, MacroOp,
    SymbolicFitResult,
    export_json, load_json, bundle_to_json_string,
    SchemaVersionError, UnsupportedNodeError,
)


def _make_fit_result(expr: ExprNode, r2: float = 0.95) -> SymbolicFitResult:
    """Helper to create a SymbolicFitResult for testing."""
    return SymbolicFitResult(
        expression=expr,
        expression_string=expr.to_string(),
        r_squared=r2,
        mse=0.01,
        complexity=expr.size(),
        residuals=np.array([0.1, -0.1, 0.05]),
        residual_std=0.08,
        is_noise_like=True,
    )


class TestJSONExportBasic:
    """Test basic JSON export."""

    def test_export_constant(self):
        """Export a constant expression."""
        expr = ExprNode(NodeType.CONSTANT, value=3.14)
        result = _make_fit_result(expr)
        bundle = export_json(result)

        assert bundle["schema"] == "ppf.export.model.v1"
        assert "expression" in bundle
        assert "tree" in bundle["expression"]

    def test_export_variable(self):
        """Export a variable expression."""
        expr = ExprNode(NodeType.VARIABLE)
        result = _make_fit_result(expr)
        bundle = export_json(result)

        tree = bundle["expression"]["tree"]
        assert tree["type"] == "VAR"

    def test_export_unary_op(self):
        """Export a unary operation."""
        expr = ExprNode(
            NodeType.UNARY_OP,
            unary_op=UnaryOp.SIN,
            child=ExprNode(NodeType.VARIABLE)
        )
        result = _make_fit_result(expr)
        bundle = export_json(result)

        tree = bundle["expression"]["tree"]
        assert tree["type"] == "UNARY_OP"
        assert tree["op"] == "SIN"

    def test_export_binary_op(self):
        """Export a binary operation."""
        expr = ExprNode(
            NodeType.BINARY_OP,
            binary_op=BinaryOp.ADD,
            left=ExprNode(NodeType.VARIABLE),
            right=ExprNode(NodeType.CONSTANT, value=1.0)
        )
        result = _make_fit_result(expr)
        bundle = export_json(result)

        tree = bundle["expression"]["tree"]
        assert tree["type"] == "BINARY_OP"
        assert tree["op"] == "ADD"
        assert len(tree["children"]) == 2


class TestJSONExportMacros:
    """Test macro serialization."""

    def test_export_damped_sin(self):
        """Export damped sine macro."""
        expr = ExprNode(
            NodeType.MACRO,
            macro_op=MacroOp.DAMPED_SIN,
            macro_params=[1.0, 0.5, 2.0, 0.0]
        )
        result = _make_fit_result(expr)
        bundle = export_json(result)

        tree = bundle["expression"]["tree"]
        assert tree["type"] == "MACRO_CALL"
        assert tree["name"] == "DAMPED_SIN"
        assert "args" in tree

    def test_export_gaussian(self):
        """Export Gaussian macro."""
        expr = ExprNode(
            NodeType.MACRO,
            macro_op=MacroOp.GAUSSIAN,
            macro_params=[1.0, 0.0, 1.0]
        )
        result = _make_fit_result(expr)
        bundle = export_json(result)

        tree = bundle["expression"]["tree"]
        assert tree["name"] == "GAUSSIAN"


class TestJSONExportMetadata:
    """Test metadata export."""

    def test_includes_metrics(self):
        """Bundle includes metrics."""
        expr = ExprNode(NodeType.CONSTANT, value=1.0)
        result = _make_fit_result(expr, r2=0.95)
        bundle = export_json(result)

        assert bundle["metrics"]["r2"] == 0.95
        assert "rmse" in bundle["metrics"]
        assert "complexity" in bundle["metrics"]

    def test_includes_constraints(self):
        """Bundle includes safety constraints."""
        expr = ExprNode(NodeType.CONSTANT, value=1.0)
        result = _make_fit_result(expr)
        bundle = export_json(result)

        assert bundle["constraints"]["safe"] is True
        assert "div_epsilon" in bundle["constraints"]
        assert "exp_clip" in bundle["constraints"]

    def test_includes_variables(self):
        """Bundle includes variable names."""
        expr = ExprNode(NodeType.VARIABLE)
        result = _make_fit_result(expr)
        bundle = export_json(result, variables=["time"])

        assert "time" in bundle["variables"]

    def test_includes_expression_string(self):
        """Bundle includes expression string."""
        expr = ExprNode(
            NodeType.UNARY_OP,
            unary_op=UnaryOp.SIN,
            child=ExprNode(NodeType.VARIABLE)
        )
        result = _make_fit_result(expr)
        bundle = export_json(result)

        assert "string" in bundle["expression"]
        assert "sin" in bundle["expression"]["string"].lower()

    def test_includes_latex(self):
        """Bundle includes LaTeX representation."""
        expr = ExprNode(
            NodeType.UNARY_OP,
            unary_op=UnaryOp.SIN,
            child=ExprNode(NodeType.VARIABLE)
        )
        result = _make_fit_result(expr)
        bundle = export_json(result)

        assert "latex" in bundle["expression"]
        assert "\\sin" in bundle["expression"]["latex"]


class TestJSONRoundtrip:
    """Test JSON round-trip (export -> load)."""

    def test_roundtrip_constant(self):
        """Round-trip a constant."""
        orig_expr = ExprNode(NodeType.CONSTANT, value=3.14)
        result = _make_fit_result(orig_expr)
        bundle = export_json(result)

        loaded_expr, metadata = load_json(bundle)

        assert loaded_expr.node_type == NodeType.CONSTANT
        assert abs(loaded_expr.value - 3.14) < 1e-10

    def test_roundtrip_variable(self):
        """Round-trip a variable."""
        orig_expr = ExprNode(NodeType.VARIABLE)
        result = _make_fit_result(orig_expr)
        bundle = export_json(result)

        loaded_expr, metadata = load_json(bundle)

        assert loaded_expr.node_type == NodeType.VARIABLE

    def test_roundtrip_unary_op(self):
        """Round-trip a unary operation."""
        orig_expr = ExprNode(
            NodeType.UNARY_OP,
            unary_op=UnaryOp.SIN,
            child=ExprNode(NodeType.VARIABLE)
        )
        result = _make_fit_result(orig_expr)
        bundle = export_json(result)

        loaded_expr, metadata = load_json(bundle)

        assert loaded_expr.node_type == NodeType.UNARY_OP
        assert loaded_expr.unary_op == UnaryOp.SIN
        assert loaded_expr.child.node_type == NodeType.VARIABLE

    def test_roundtrip_binary_op(self):
        """Round-trip a binary operation."""
        orig_expr = ExprNode(
            NodeType.BINARY_OP,
            binary_op=BinaryOp.MUL,
            left=ExprNode(NodeType.CONSTANT, value=2.0),
            right=ExprNode(NodeType.VARIABLE)
        )
        result = _make_fit_result(orig_expr)
        bundle = export_json(result)

        loaded_expr, metadata = load_json(bundle)

        assert loaded_expr.node_type == NodeType.BINARY_OP
        assert loaded_expr.binary_op == BinaryOp.MUL
        assert loaded_expr.left.value == 2.0
        assert loaded_expr.right.node_type == NodeType.VARIABLE

    def test_roundtrip_macro(self):
        """Round-trip a macro."""
        orig_expr = ExprNode(
            NodeType.MACRO,
            macro_op=MacroOp.DAMPED_SIN,
            macro_params=[1.0, 0.5, 2.0, 0.0]
        )
        result = _make_fit_result(orig_expr)
        bundle = export_json(result)

        loaded_expr, metadata = load_json(bundle)

        assert loaded_expr.node_type == NodeType.MACRO
        assert loaded_expr.macro_op == MacroOp.DAMPED_SIN
        assert len(loaded_expr.macro_params) == 4

    def test_roundtrip_evaluation(self):
        """Loaded expression evaluates same as original."""
        orig_expr = ExprNode(
            NodeType.BINARY_OP,
            binary_op=BinaryOp.MUL,
            left=ExprNode(NodeType.CONSTANT, value=2.0),
            right=ExprNode(
                NodeType.UNARY_OP,
                unary_op=UnaryOp.SIN,
                child=ExprNode(NodeType.VARIABLE)
            )
        )
        result = _make_fit_result(orig_expr)
        bundle = export_json(result)

        loaded_expr, _ = load_json(bundle)

        x = np.linspace(0, 10, 100)
        orig_y = orig_expr.evaluate(x)
        loaded_y = loaded_expr.evaluate(x)

        np.testing.assert_allclose(orig_y, loaded_y, rtol=1e-10)


class TestJSONValidation:
    """Test JSON validation and error handling."""

    def test_unsupported_schema(self):
        """Unknown schema version raises error."""
        bundle = {"schema": "ppf.export.model.v999", "expression": {"tree": {}}}

        with pytest.raises(SchemaVersionError):
            load_json(bundle)

    def test_missing_tree(self):
        """Missing tree raises error."""
        bundle = {
            "schema": "ppf.export.model.v1",
            "expression": {"string": "sin(x)"}  # No tree
        }

        with pytest.raises(ValueError) as exc_info:
            load_json(bundle)

        assert "tree" in str(exc_info.value).lower()

    def test_unknown_node_type(self):
        """Unknown node type raises error."""
        bundle = {
            "schema": "ppf.export.model.v1",
            "expression": {
                "tree": {"type": "UNKNOWN_NODE", "value": 1}
            }
        }

        with pytest.raises(UnsupportedNodeError):
            load_json(bundle)


class TestJSONSerialization:
    """Test JSON string serialization."""

    def test_to_json_string(self):
        """Bundle can be converted to JSON string."""
        expr = ExprNode(NodeType.CONSTANT, value=1.0)
        result = _make_fit_result(expr)
        bundle = export_json(result)

        json_str = bundle_to_json_string(bundle)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["schema"] == bundle["schema"]

    def test_json_string_deterministic(self):
        """JSON string output is deterministic."""
        expr = ExprNode(NodeType.CONSTANT, value=1.0)
        result = _make_fit_result(expr)
        bundle = export_json(result)

        # Note: created_utc will differ, so we need to fix it for comparison
        bundle["created_utc"] = "2026-01-01T00:00:00Z"

        json1 = bundle_to_json_string(bundle)
        json2 = bundle_to_json_string(bundle)

        assert json1 == json2
