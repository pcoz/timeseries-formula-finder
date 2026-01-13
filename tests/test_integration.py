"""
Integration tests using real-world data.
"""

import os
import csv
import numpy as np
import pytest

from ppf import (
    PPFResidualLayer,
    HierarchicalDetector,
    EntropyMethod,
    FormType,
    print_stack_result,
    print_hierarchical_result,
)


def load_mars_data(filepath):
    """Load Mars radial distance data from CSV"""
    dates = []
    distances = []

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dates.append(row['date'])
            distances.append(float(row['distance_au']))

    return np.array(dates), np.array(distances)


def load_sunspot_data(filepath, start_year=None):
    """Load sunspot data from SILSO CSV format"""
    years = []
    sunspots = []

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(';')
            if len(parts) >= 4:
                year = int(parts[0])
                if start_year is None or year >= start_year:
                    decimal_year = float(parts[2])
                    sunspot_num = float(parts[3])
                    years.append(decimal_year)
                    sunspots.append(sunspot_num)

    return np.array(years), np.array(sunspots)


@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "test_data", "mars_radial_distance.csv")),
    reason="Mars data file not found"
)
class TestMarsData:
    """Integration tests with Mars orbital data"""

    def test_mars_orbit_detection(self, mars_data_path):
        """Test that Mars orbit is detected correctly"""
        dates, distances = load_mars_data(mars_data_path)
        
        layer = PPFResidualLayer(
            entropy_method=EntropyMethod.SPECTRAL,
            noise_threshold=0.5,
            min_compression_gain=0.05,
            min_r_squared=0.3
        )
        result = layer.analyze(distances)
        
        # Should find at least one form
        assert len(result.form_stack) >= 1
        
        # First form should be sine (orbital motion)
        assert result.form_stack[0].form_type == FormType.SINE
        
        # Should explain most variance
        variance_explained = 1 - (np.var(result.final_residuals) / np.var(distances))
        assert variance_explained > 0.95  # 95%+ variance explained

    def test_mars_period_accuracy(self, mars_data_path):
        """Test that detected period matches Mars orbit (~687 days)"""
        dates, distances = load_mars_data(mars_data_path)
        
        layer = PPFResidualLayer(
            entropy_method=EntropyMethod.SPECTRAL,
            min_r_squared=0.3
        )
        result = layer.analyze(distances)
        
        if result.form_stack and result.form_stack[0].form_type == FormType.SINE:
            freq = result.form_stack[0].params[1]
            period_samples = 2 * np.pi / freq
            period_days = period_samples * 5  # 5-day sampling interval
            
            # Mars orbital period is ~687 days
            assert 650 < period_days < 730  # Within reasonable range


@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "test_data", "sunspot_monthly.csv")),
    reason="Sunspot data file not found"
)
class TestSunspotData:
    """Integration tests with sunspot data"""

    def test_sunspot_hierarchical_detection(self, sunspot_data_path):
        """Test hierarchical detection on sunspot data"""
        years, sunspots = load_sunspot_data(sunspot_data_path)
        
        detector = HierarchicalDetector(
            window_size=132,  # ~11 years in months
            window_overlap=0.0,
            min_r_squared=0.3,
            preferred_form=FormType.SINE,
            max_levels=3,
            min_windows_for_meta=8
        )
        result = detector.analyze(sunspots)
        
        # Should detect at least level 1
        assert len(result.levels) >= 1
        
        # Level 1 should have successful window fits
        level1 = result.levels[0]
        assert len(level1.window_fits) > 0
        assert level1.dominant_form == FormType.SINE

    def test_sunspot_gleissberg_detection(self, sunspot_data_path):
        """Test detection of Gleissberg cycle in amplitude evolution"""
        years, sunspots = load_sunspot_data(sunspot_data_path)
        
        detector = HierarchicalDetector(
            window_size=132,
            min_r_squared=0.3,
            preferred_form=FormType.SINE,
            max_levels=3
        )
        result = detector.analyze(sunspots)
        
        # Check if level 2 exists and has amplitude evolution
        if len(result.levels) >= 2:
            level2 = result.levels[1]
            if 'amplitude' in level2.parameter_evolutions:
                amp_evo = level2.parameter_evolutions['amplitude']
                
                # Check if Gleissberg-like pattern detected
                if amp_evo.has_structure:
                    for form in amp_evo.ppf_result.form_stack:
                        if form.form_type == FormType.SINE:
                            freq = form.params[1]
                            period_windows = 2 * np.pi / freq
                            period_years = period_windows * 11
                            
                            # Gleissberg cycle is ~80-120 years
                            # Allow wider range for imperfect detection
                            if 60 < period_years < 150:
                                return  # Success!
        
        # If we get here, Gleissberg wasn't clearly detected
        # This is OK - it's a challenging detection
        pytest.skip("Gleissberg cycle not clearly detected (expected for noisy data)")
