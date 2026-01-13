"""
Tests for the Hierarchical Detection module.
"""

import numpy as np
import pytest

from ppf import (
    HierarchicalDetector,
    HierarchicalResult,
    HierarchyLevel,
    ParameterEvolution,
    FormType,
    EntropyMethod,
)


class TestHierarchicalDetector:
    """Tests for HierarchicalDetector class"""

    def test_init_validation(self):
        """Test that constructor validates parameters"""
        with pytest.raises(ValueError):
            HierarchicalDetector(window_size=2)  # Too small
        
        with pytest.raises(ValueError):
            HierarchicalDetector(window_overlap=1.5)  # Out of range

    def test_detect_single_level(self):
        """Test detecting a single-level pattern"""
        np.random.seed(42)
        x = np.arange(200)
        data = 5.0 * np.sin(0.1 * x) + np.random.normal(0, 0.5, 200)
        
        detector = HierarchicalDetector(
            window_size=50,
            min_r_squared=0.5,
            preferred_form=FormType.SINE
        )
        result = detector.analyze(data)
        
        assert isinstance(result, HierarchicalResult)
        assert len(result.levels) >= 1
        assert result.levels[0].dominant_form == FormType.SINE

    def test_detect_hierarchical_pattern(self, synthetic_hierarchical_data):
        """Test detecting a hierarchical pattern"""
        data = synthetic_hierarchical_data
        
        detector = HierarchicalDetector(
            window_size=40,
            min_r_squared=0.5,
            preferred_form=FormType.SINE,
            max_levels=3
        )
        result = detector.analyze(data)
        
        assert len(result.levels) >= 1
        
        # Check level 1 detected the base form
        level1 = result.levels[0]
        assert level1.coverage > 0.5
        assert level1.dominant_form == FormType.SINE

    def test_auto_window_size(self):
        """Test automatic window size detection"""
        np.random.seed(42)
        # Create data with ~40 point period
        x = np.arange(400)
        data = np.sin(2 * np.pi * x / 40) + np.random.normal(0, 0.1, 400)
        
        detector = HierarchicalDetector(
            window_size=None,  # Auto-detect
            min_r_squared=0.3
        )
        result = detector.analyze(data)
        
        # Should still detect the pattern
        assert len(result.levels) >= 1


class TestHierarchicalResult:
    """Tests for HierarchicalResult class"""

    def test_summary_generation(self, synthetic_hierarchical_data):
        """Test that summary is generated correctly"""
        data = synthetic_hierarchical_data
        
        detector = HierarchicalDetector(
            window_size=40,
            preferred_form=FormType.SINE
        )
        result = detector.analyze(data)
        
        summary = result.summary()
        
        assert isinstance(summary, str)
        assert "Level 1" in summary


class TestParameterEvolution:
    """Tests for ParameterEvolution class"""

    def test_has_structure_detection(self):
        """Test structure detection in parameter evolution"""
        # Evolution with no analysis
        evo = ParameterEvolution(
            param_name="amplitude",
            positions=np.arange(10, dtype=float),
            values=np.random.normal(0, 1, 10)
        )
        
        assert evo.has_structure is False
        assert evo.variance_explained == 0.0
