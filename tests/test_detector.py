"""
Tests for the PPF Detector module.
"""

import numpy as np
import pytest

from ppf import (
    PPFDetector,
    FormType,
    fit_form,
    find_best_form,
    evaluate_form,
    check_residuals_are_noise,
)


class TestFormFitting:
    """Tests for form fitting functions"""

    def test_fit_constant(self):
        """Test fitting a constant form"""
        np.random.seed(42)
        x = np.arange(50, dtype=float)
        y = np.full(50, 5.0) + np.random.normal(0, 0.1, 50)

        result = fit_form(x, y, FormType.CONSTANT)

        assert result is not None
        assert result.form_type == FormType.CONSTANT
        # For constant fit, R² can be low/0 since SS_tot ≈ noise variance
        # Just verify param is close to 5.0
        assert abs(result.params[0] - 5.0) < 0.5

    def test_fit_linear(self):
        """Test fitting a linear form"""
        x = np.arange(50, dtype=float)
        y = 2.0 * x + 5.0 + np.random.normal(0, 0.5, 50)
        
        result = fit_form(x, y, FormType.LINEAR)
        
        assert result is not None
        assert result.form_type == FormType.LINEAR
        assert result.r_squared > 0.95
        assert abs(result.params[0] - 2.0) < 0.2  # slope
        assert abs(result.params[1] - 5.0) < 1.0  # intercept

    def test_fit_sine(self):
        """Test fitting a sine form"""
        x = np.arange(100, dtype=float)
        y = 3.0 * np.sin(0.15 * x) + 10.0 + np.random.normal(0, 0.2, 100)
        
        result = fit_form(x, y, FormType.SINE)
        
        assert result is not None
        assert result.form_type == FormType.SINE
        assert result.r_squared > 0.9
        assert abs(result.params[0] - 3.0) < 0.5  # amplitude

    def test_fit_insufficient_data(self):
        """Test that fitting fails gracefully with too few points"""
        x = np.array([0, 1])
        y = np.array([1.0, 2.0])
        
        result = fit_form(x, y, FormType.SINE)
        assert result is None

    def test_find_best_form(self):
        """Test finding the best form type"""
        x = np.arange(100, dtype=float)
        y = 2.0 * x + 5.0 + np.random.normal(0, 0.5, 100)
        
        result = find_best_form(x, y, min_r_squared=0.5)
        
        assert result is not None
        assert result.form_type == FormType.LINEAR


class TestEvaluateForm:
    """Tests for form evaluation"""

    def test_evaluate_constant(self):
        """Test evaluating a constant form"""
        x = np.array([0, 1, 2, 3, 4], dtype=float)
        params = np.array([5.0])
        
        result = evaluate_form(FormType.CONSTANT, params, x)
        
        np.testing.assert_array_almost_equal(result, [5.0, 5.0, 5.0, 5.0, 5.0])

    def test_evaluate_linear(self):
        """Test evaluating a linear form"""
        x = np.array([0, 1, 2, 3], dtype=float)
        params = np.array([2.0, 3.0])  # y = 2x + 3
        
        result = evaluate_form(FormType.LINEAR, params, x)
        
        np.testing.assert_array_almost_equal(result, [3.0, 5.0, 7.0, 9.0])


class TestResidualsNoise:
    """Tests for residual noise detection"""

    def test_white_noise_detected(self):
        """Test that white noise is detected as noise-like"""
        np.random.seed(42)
        residuals = np.random.normal(0, 1, 100)
        
        assert check_residuals_are_noise(residuals) is True

    def test_structured_residuals_detected(self):
        """Test that structured residuals are not detected as noise"""
        x = np.arange(100)
        residuals = np.sin(0.2 * x)  # Clear pattern
        
        assert check_residuals_are_noise(residuals) is False


class TestPPFDetector:
    """Tests for the main PPFDetector class"""

    def test_init_validation(self):
        """Test that constructor validates parameters"""
        with pytest.raises(ValueError):
            PPFDetector(min_window=2)  # Too small
        
        with pytest.raises(ValueError):
            PPFDetector(min_r_squared=1.5)  # Out of range

    def test_detect_simple_pattern(self, synthetic_sine_data):
        """Test detecting a simple pattern"""
        x, data = synthetic_sine_data
        
        detector = PPFDetector(
            min_window=30,
            min_r_squared=0.6
        )
        result = detector.analyze(data)
        
        assert result is not None
        assert result.structure_score >= 0
        assert result.noise_level >= 0
