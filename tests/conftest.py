"""
Pytest configuration and shared fixtures for PPF tests.
"""

import os
import pytest
import numpy as np

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "test_data")


@pytest.fixture
def test_data_dir():
    """Path to test data directory"""
    return TEST_DATA_DIR


@pytest.fixture
def mars_data_path(test_data_dir):
    """Path to Mars orbital data"""
    return os.path.join(test_data_dir, "mars_radial_distance.csv")


@pytest.fixture
def sunspot_data_path(test_data_dir):
    """Path to sunspot data"""
    return os.path.join(test_data_dir, "sunspot_monthly.csv")


@pytest.fixture
def synthetic_sine_data():
    """Generate synthetic sine wave with noise"""
    np.random.seed(42)
    x = np.arange(200, dtype=float)
    clean = 2.0 * np.sin(0.1 * x) + 0.05 * x + 10.0
    noise = np.random.normal(0, 0.2, len(x))
    return x, clean + noise


@pytest.fixture
def synthetic_hierarchical_data():
    """Generate synthetic data with hierarchical structure"""
    np.random.seed(42)
    t = np.arange(1000)
    # Fast oscillation with amplitude modulation
    amplitude = 10 + 5 * np.sin(2 * np.pi * t / 200)
    signal = amplitude * np.sin(2 * np.pi * t / 20)
    noise = np.random.normal(0, 1, len(t))
    return signal + noise
