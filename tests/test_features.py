"""
Tests for feature extraction from PPF results.
"""

import pytest
import numpy as np

from ppf import (
    ExprNode, NodeType, UnaryOp, BinaryOp, MacroOp,
    SymbolicFitResult, SymbolicRegressionResult,
    extract_features, feature_vector, feature_matrix, get_schema_info,
)


def _make_fit_result(expr: ExprNode, r2: float = 0.95, residuals=None) -> SymbolicFitResult:
    """Helper to create a SymbolicFitResult for testing."""
    if residuals is None:
        residuals = np.random.randn(100) * 0.1
    return SymbolicFitResult(
        expression=expr,
        expression_string=expr.to_string(),
        r_squared=r2,
        mse=(residuals ** 2).mean(),
        complexity=expr.size(),
        residuals=residuals,
        residual_std=residuals.std(),
        is_noise_like=True,
    )


class TestExtractFeaturesBasic:
    """Test basic feature extraction."""

    def test_common_features_present(self):
        """Common features are always present."""
        expr = ExprNode(NodeType.CONSTANT, value=1.0)
        result = _make_fit_result(expr)
        features = extract_features(result)

        assert "r2" in features
        assert "rmse" in features
        assert "complexity" in features
        assert "dominant_family" in features

    def test_r2_value(self):
        """R2 value is correct."""
        expr = ExprNode(NodeType.CONSTANT, value=1.0)
        result = _make_fit_result(expr, r2=0.87)
        features = extract_features(result)

        assert features["r2"] == 0.87

    def test_complexity_value(self):
        """Complexity is extracted correctly."""
        expr = ExprNode(
            NodeType.BINARY_OP,
            binary_op=BinaryOp.ADD,
            left=ExprNode(NodeType.VARIABLE),
            right=ExprNode(NodeType.CONSTANT, value=1.0)
        )
        result = _make_fit_result(expr)
        features = extract_features(result)

        assert features["complexity"] == expr.size()


class TestExtractFeaturesFamily:
    """Test dominant family classification."""

    def test_oscillation_family_sin(self):
        """Sin expression classified as oscillation."""
        expr = ExprNode(
            NodeType.UNARY_OP,
            unary_op=UnaryOp.SIN,
            child=ExprNode(NodeType.VARIABLE)
        )
        result = _make_fit_result(expr)
        features = extract_features(result)

        assert features["dominant_family"] == "oscillation"

    def test_oscillation_family_damped_sin(self):
        """Damped sine macro classified as oscillation."""
        expr = ExprNode(
            NodeType.MACRO,
            macro_op=MacroOp.DAMPED_SIN,
            macro_params=[1.0, 0.5, 2.0, 0.0]
        )
        result = _make_fit_result(expr)
        features = extract_features(result)

        assert features["dominant_family"] == "oscillation"

    def test_saturation_family_sigmoid(self):
        """Sigmoid macro classified as saturation."""
        expr = ExprNode(
            NodeType.MACRO,
            macro_op=MacroOp.SIGMOID,
            macro_params=[1.0, 1.0, 0.0]
        )
        result = _make_fit_result(expr)
        features = extract_features(result)

        assert features["dominant_family"] == "saturation"

    def test_ratio_family(self):
        """Ratio macro classified as ratio."""
        expr = ExprNode(
            NodeType.MACRO,
            macro_op=MacroOp.RATIO,
            macro_params=[1.0, 0.0, 1.0, 1.0]
        )
        result = _make_fit_result(expr)
        features = extract_features(result)

        assert features["dominant_family"] == "ratio"

    def test_peaks_family_gaussian(self):
        """Gaussian macro classified as peaks."""
        expr = ExprNode(
            NodeType.MACRO,
            macro_op=MacroOp.GAUSSIAN,
            macro_params=[1.0, 0.0, 1.0]
        )
        result = _make_fit_result(expr)
        features = extract_features(result)

        assert features["dominant_family"] == "peaks"

    def test_algebraic_family(self):
        """Pure algebraic expression classified correctly."""
        expr = ExprNode(
            NodeType.BINARY_OP,
            binary_op=BinaryOp.ADD,
            left=ExprNode(NodeType.VARIABLE),
            right=ExprNode(NodeType.CONSTANT, value=1.0)
        )
        result = _make_fit_result(expr)
        features = extract_features(result)

        assert features["dominant_family"] == "algebraic"


class TestExtractFeaturesDomainSpecific:
    """Test domain-specific parameter extraction."""

    def test_oscillator_params(self):
        """Oscillator parameters extracted from damped sine."""
        expr = ExprNode(
            NodeType.MACRO,
            macro_op=MacroOp.DAMPED_SIN,
            macro_params=[2.5, 0.3, 4.0, 1.57]  # a, k, w, phi
        )
        result = _make_fit_result(expr)
        features = extract_features(result)

        assert "amplitude" in features
        assert features["amplitude"] == 2.5
        assert "damping_k" in features
        assert features["damping_k"] == 0.3
        assert "omega" in features
        assert features["omega"] == 4.0
        assert "phase" in features
        assert features["phase"] == 1.57

    def test_gaussian_params(self):
        """Gaussian parameters extracted."""
        expr = ExprNode(
            NodeType.MACRO,
            macro_op=MacroOp.GAUSSIAN,
            macro_params=[3.0, 5.0, 2.0]  # a, mu, sigma
        )
        result = _make_fit_result(expr)
        features = extract_features(result)

        assert features["amplitude"] == 3.0
        assert features["mu"] == 5.0
        assert features["sigma"] == 2.0


class TestExtractFeaturesResiduals:
    """Test residual statistics extraction."""

    def test_residual_stats_included(self):
        """Residual stats included by default."""
        expr = ExprNode(NodeType.CONSTANT, value=1.0)
        residuals = np.array([0.1, -0.1, 0.05, -0.05])
        result = _make_fit_result(expr, residuals=residuals)
        features = extract_features(result)

        assert "residual_rms" in features
        assert "residual_mad" in features
        assert "residual_max_abs" in features

    def test_residual_stats_excluded(self):
        """Residual stats can be excluded."""
        expr = ExprNode(NodeType.CONSTANT, value=1.0)
        result = _make_fit_result(expr)
        features = extract_features(result, include_residual_stats=False)

        assert "residual_rms" not in features
        assert "residual_mad" not in features


class TestFeatureVector:
    """Test feature vectorization."""

    def test_returns_array_and_names(self):
        """Returns both array and names."""
        expr = ExprNode(NodeType.CONSTANT, value=1.0)
        result = _make_fit_result(expr)
        features = extract_features(result)

        vec, names = feature_vector(features)

        assert isinstance(vec, np.ndarray)
        assert isinstance(names, list)
        assert len(vec) == len(names)

    def test_family_one_hot_encoded(self):
        """Family is one-hot encoded."""
        expr = ExprNode(
            NodeType.UNARY_OP,
            unary_op=UnaryOp.SIN,
            child=ExprNode(NodeType.VARIABLE)
        )
        result = _make_fit_result(expr)
        features = extract_features(result)

        vec, names = feature_vector(features)

        # Should have family_* columns
        family_names = [n for n in names if n.startswith("family_")]
        assert len(family_names) > 0

        # oscillation should be 1, others 0
        osc_idx = names.index("family_oscillation")
        assert vec[osc_idx] == 1.0

    def test_missing_features_are_nan(self):
        """Missing domain features filled with NaN."""
        # Algebraic expression won't have oscillator params
        expr = ExprNode(
            NodeType.BINARY_OP,
            binary_op=BinaryOp.ADD,
            left=ExprNode(NodeType.VARIABLE),
            right=ExprNode(NodeType.CONSTANT, value=1.0)
        )
        result = _make_fit_result(expr)
        features = extract_features(result)

        vec, names = feature_vector(features)

        # Oscillator params should be NaN
        if "amplitude" in names:
            amp_idx = names.index("amplitude")
            assert np.isnan(vec[amp_idx])

    def test_different_schemas(self):
        """Different schemas produce different vector lengths."""
        expr = ExprNode(NodeType.CONSTANT, value=1.0)
        result = _make_fit_result(expr)
        features = extract_features(result)

        vec_min, names_min = feature_vector(features, schema="ppf.features.v1.edge_min")
        vec_full, names_full = feature_vector(features, schema="ppf.features.v1.full")

        assert len(vec_min) < len(vec_full)


class TestFeatureMatrix:
    """Test feature matrix for multiple samples."""

    def test_matrix_shape(self):
        """Matrix has correct shape."""
        results = []
        for i in range(5):
            expr = ExprNode(NodeType.CONSTANT, value=float(i))
            results.append(_make_fit_result(expr, r2=0.8 + i * 0.02))

        features_list = [extract_features(r) for r in results]
        matrix, names = feature_matrix(features_list)

        assert matrix.shape[0] == 5  # 5 samples
        assert matrix.shape[1] == len(names)

    def test_empty_list_raises(self):
        """Empty list raises error."""
        with pytest.raises(ValueError):
            feature_matrix([])


class TestSchemaInfo:
    """Test schema information."""

    def test_get_schema_info(self):
        """Schema info returns metadata."""
        info = get_schema_info("ppf.features.v1.full")

        assert "name" in info
        assert "n_features" in info
        assert "fields" in info

    def test_unknown_schema_raises(self):
        """Unknown schema raises error."""
        with pytest.raises(ValueError):
            get_schema_info("unknown.schema.v1")
