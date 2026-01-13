"""
Feature vectorization for ML pipelines.

Converts feature dictionaries to fixed-order numpy arrays
suitable for scikit-learn and other ML frameworks.
"""

from typing import Dict, Any, Tuple, List
import numpy as np


# Feature schemas define the order and fields for vectorization
# Each schema is a list of (key, default_value) tuples

# Minimal schema for edge/TinyML deployment
SCHEMA_EDGE_MIN = [
    ("r2", 0.0),
    ("rmse", 0.0),
    ("complexity", 0),
    ("dominant_family", "unknown"),  # Will be one-hot encoded
    ("residual_rms", np.nan),
    ("residual_mad", np.nan),
    ("amplitude", np.nan),
    ("omega", np.nan),
    ("damping_k", np.nan),
    ("K", np.nan),
    ("r", np.nan),
    ("numerator_degree", np.nan),
    ("denominator_degree", np.nan),
]

# Full schema for comprehensive ML pipelines
SCHEMA_FULL = [
    # Core metrics
    ("r2", 0.0),
    ("rmse", 0.0),
    ("complexity", 0),

    # Family indicator (will be one-hot encoded)
    ("dominant_family", "unknown"),

    # Residual stats
    ("residual_rms", np.nan),
    ("residual_mad", np.nan),
    ("residual_max_abs", np.nan),

    # Oscillator parameters
    ("amplitude", np.nan),
    ("omega", np.nan),
    ("freq_hz", np.nan),
    ("damping_k", np.nan),
    ("phase", np.nan),

    # Logistic/growth parameters
    ("K", np.nan),
    ("r", np.nan),
    ("t0", np.nan),
    ("Km", np.nan),
    ("n", np.nan),

    # Rational function parameters
    ("numerator_degree", np.nan),
    ("denominator_degree", np.nan),
    ("gain_estimate", np.nan),

    # Gaussian/peak parameters
    ("mu", np.nan),
    ("sigma", np.nan),
    ("exponent", np.nan),
    ("offset", np.nan),

    # Domain probe scores (if available)
    ("domain_score_oscillator", np.nan),
    ("domain_score_circuit", np.nan),
    ("domain_score_growth", np.nan),
    ("domain_score_rational", np.nan),
    ("domain_score_universal", np.nan),
    ("domain_score_polynomial", np.nan),
]

# Family categories for one-hot encoding
FAMILY_CATEGORIES = ["oscillation", "growth", "saturation", "decay", "ratio", "peaks", "algebraic", "unknown"]

# Schema registry
SCHEMAS = {
    "ppf.features.v1.edge_min": SCHEMA_EDGE_MIN,
    "ppf.features.v1.full": SCHEMA_FULL,
}


def _one_hot_family(family: str) -> Dict[str, float]:
    """One-hot encode the dominant family."""
    result = {}
    for cat in FAMILY_CATEGORIES:
        result[f"family_{cat}"] = 1.0 if family == cat else 0.0
    return result


def feature_vector(
    features: Dict[str, Any],
    schema: str = "ppf.features.v1.full"
) -> Tuple[np.ndarray, List[str]]:
    """
    Convert feature dictionary to a fixed-order numpy array.

    Args:
        features: Feature dictionary from extract_features()
        schema: Schema name defining field order and defaults

    Returns:
        Tuple of (array, names):
        - array: NumPy array of feature values
        - names: List of feature names in corresponding order

    Example:
        >>> features = extract_features(result)
        >>> vec, names = feature_vector(features)
        >>> X = vec.reshape(1, -1)  # For sklearn
    """
    if schema not in SCHEMAS:
        raise ValueError(f"Unknown schema: {schema}. Available: {list(SCHEMAS.keys())}")

    schema_def = SCHEMAS[schema]

    values = []
    names = []

    for key, default in schema_def:
        if key == "dominant_family":
            # Special handling: one-hot encode family
            family = features.get(key, default)
            one_hot = _one_hot_family(family)
            for cat in FAMILY_CATEGORIES:
                names.append(f"family_{cat}")
                values.append(one_hot[f"family_{cat}"])
        else:
            names.append(key)
            val = features.get(key, default)
            # Convert to float if possible
            if isinstance(val, (int, float)):
                values.append(float(val))
            elif val is None:
                values.append(np.nan)
            else:
                # Try to convert, otherwise NaN
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    values.append(np.nan)

    return np.array(values, dtype=np.float64), names


def feature_matrix(
    features_list: List[Dict[str, Any]],
    schema: str = "ppf.features.v1.full"
) -> Tuple[np.ndarray, List[str]]:
    """
    Convert multiple feature dictionaries to a feature matrix.

    Args:
        features_list: List of feature dictionaries
        schema: Schema name defining field order

    Returns:
        Tuple of (matrix, names):
        - matrix: 2D NumPy array (n_samples, n_features)
        - names: List of feature names

    Example:
        >>> features_list = [extract_features(r) for r in results]
        >>> X, names = feature_matrix(features_list)
        >>> clf.fit(X, y)
    """
    if not features_list:
        raise ValueError("features_list cannot be empty")

    # Get names from first sample
    _, names = feature_vector(features_list[0], schema)

    # Build matrix
    rows = []
    for features in features_list:
        vec, _ = feature_vector(features, schema)
        rows.append(vec)

    return np.vstack(rows), names


def get_schema_info(schema: str = "ppf.features.v1.full") -> Dict[str, Any]:
    """
    Get information about a feature schema.

    Args:
        schema: Schema name

    Returns:
        Dictionary with schema metadata
    """
    if schema not in SCHEMAS:
        raise ValueError(f"Unknown schema: {schema}")

    schema_def = SCHEMAS[schema]

    # Count features (accounting for one-hot encoding)
    n_features = 0
    for key, _ in schema_def:
        if key == "dominant_family":
            n_features += len(FAMILY_CATEGORIES)
        else:
            n_features += 1

    return {
        "name": schema,
        "n_features": n_features,
        "n_raw_fields": len(schema_def),
        "family_categories": FAMILY_CATEGORIES,
        "fields": [k for k, _ in schema_def],
    }
