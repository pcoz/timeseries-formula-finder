# Testing Documentation

This document describes the test suite, datasets used, and testing methodology for PPF.

## Test Suite Overview

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_symbolic.py -v          # Symbolic regression core
pytest tests/test_export_*.py -v          # Export layer
pytest tests/test_features.py -v          # Feature extraction
pytest tests/test_detector.py -v          # Form detection
pytest tests/test_hybrid*.py -v           # Hybrid decomposition
```

### Test Count by Module

| Module | Tests | Description |
|--------|-------|-------------|
| `test_symbolic.py` | 45 | GP engine, expression trees, discovery |
| `test_export_python.py` | 16 | Python code generation |
| `test_export_c.py` | 19 | C code generation |
| `test_export_json.py` | 21 | JSON serialization/round-trip |
| `test_features.py` | 22 | Feature extraction and vectorization |
| `test_detector.py` | 12 | Fixed-form fitting |
| `test_hybrid.py` | 8 | EMD/SSA + interpretation |
| **Total** | **143+** | |

---

## Datasets Used

### 1. Synthetic Benchmarks

#### Standard Symbolic Regression Benchmarks

| Name | Formula | Source |
|------|---------|--------|
| Nguyen-1 | `x³ + x² + x` | [Nguyen 2011] |
| Nguyen-4 | `x⁶ + x⁵ + x⁴ + x³ + x² + x` | [Nguyen 2011] |
| Keijzer-4 | `x³ · exp(-x) · cos(x) · sin(x) · (sin²(x)·cos(x) - 1)` | [Keijzer 2003] |
| Korns-12 | `2 - 2.1·cos(9.8·x)·sin(1.3·w)` | [Korns 2011] |
| Damped oscillator | `a·exp(-k·t)·sin(ω·t + φ)` | Physics |
| RC charge | `a·(1 - exp(-k·t)) + c` | Electronics |
| Logistic | `L / (1 + exp(-k·(x - x₀)))` | Biology |

These are generated synthetically with controlled noise levels (typically σ = 0.05-0.1 of signal range).

#### Why These Benchmarks?

1. **Nguyen benchmarks**: Standard in symbolic regression literature, allows comparison with published results
2. **Physics-based (damped oscillator, RC)**: Test macro template discovery
3. **Multi-domain**: Test discovery mode selection

### 2. Real-World Datasets

#### IoT Temperature Sensor Data

**Source**: [Kaggle - Temperature Readings: IOT Devices](https://www.kaggle.com/datasets/atulanandjha/temperature-readings-iot-devices)

| Metric | Value |
|--------|-------|
| Records | 97,606 |
| Duration | 133 days |
| Indoor records | 20,345 |
| Outdoor records | 77,261 |

**What PPF discovered**:
- Daily cycles: Damped sinusoid (R² = 0.58 outdoor, 0.30 indoor)
- First 24h: Power law warming (R² = 0.89)
- Weekly: 2.3°C variation between weekday/weekend

**Why this dataset**:
- Real sensor noise characteristics
- Multiple time scales (hourly, daily, weekly)
- Indoor vs outdoor comparison
- Demonstrates edge AI deployment scenario

#### OEIS Sequences

**Source**: [Online Encyclopedia of Integer Sequences](https://oeis.org/)

| Metric | Value |
|--------|-------|
| Sequences tested | 386,000+ |
| Detection rate | ~15% |

**Why this dataset**:
- Ground truth (sequences have known forms)
- Tests limits of form discovery
- Validates false positive rate

### 3. Simulated Datasets

#### Edge AI Sensor Simulation

Generated vibration signals simulating rotating machinery:

| Condition | Signature |
|-----------|-----------|
| Normal | Clean sinusoid at rotation frequency |
| Imbalance | Dominant 1x RPM |
| Bearing fault | Damped oscillations (impact response) |
| Misalignment | Strong 2x harmonic |
| Looseness | Multiple harmonics + sub-harmonic |

**Why these simulations**:
- Controlled ground truth for fault classification
- Tests damped oscillation macro
- Demonstrates predictive maintenance use case

#### ECG Waveform Simulation

Synthetic ECG with controllable morphology:

| Component | Model |
|-----------|-------|
| P-wave | Gaussian |
| QRS complex | Sum of Gaussians |
| T-wave | Asymmetric Gaussian / Damped cosine |

**Why these simulations**:
- Tests multi-component signal analysis
- Demonstrates biomedical application
- Ground truth for wave parameters

---

## Test Categories

### Unit Tests

**Purpose**: Verify individual components work correctly.

```python
# Example: Expression evaluation
def test_damped_sin_evaluation():
    expr = ExprNode(
        NodeType.MACRO,
        macro_op=MacroOp.DAMPED_SIN,
        macro_params=[1.0, 0.5, 2.0, 0.0]
    )
    x = np.array([0.0, 1.0, 2.0])
    y = expr.evaluate(x)
    # Verify against analytical formula
    expected = np.exp(-0.5 * x) * np.sin(2.0 * x)
    np.testing.assert_allclose(y, expected, rtol=1e-10)
```

**Coverage**:
- All node types (CONSTANT, VARIABLE, UNARY_OP, BINARY_OP, MACRO)
- All operators (sin, cos, exp, log, sqrt, +, -, *, /, ^)
- All macros (DAMPED_SIN, GAUSSIAN, SIGMOID, etc.)
- Edge cases (division by zero, log(0), exp overflow)

### Integration Tests

**Purpose**: Verify end-to-end workflows.

```python
# Example: Discover → Export → Evaluate
def test_discover_export_evaluate():
    # Generate data
    t = np.linspace(0, 10, 200)
    y = 2.0 * np.sin(3.0 * t) + 0.1 * np.random.randn(200)

    # Discover
    result = regressor.discover(t, y, mode=DiscoveryMode.OSCILLATOR)
    assert result.best_tradeoff.r_squared > 0.9

    # Export to Python
    code = export_python(result.best_tradeoff.expression)
    exec(code, globals())

    # Evaluate and compare
    y_pred = np.array([ppf_model(ti) for ti in t])
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
    assert r2 > 0.9
```

### Round-Trip Tests

**Purpose**: Verify serialization preserves structure.

```python
def test_json_roundtrip():
    # Original expression
    orig_expr = ExprNode(...)

    # Export to JSON
    bundle = export_json(result)

    # Load back
    loaded_expr, metadata = load_json(bundle)

    # Verify identical evaluation
    x = np.linspace(0, 10, 100)
    np.testing.assert_allclose(
        orig_expr.evaluate(x),
        loaded_expr.evaluate(x),
        rtol=1e-10
    )
```

### Safety Tests

**Purpose**: Verify numerical safety.

```python
def test_safe_division_by_zero():
    expr = ExprNode(
        NodeType.BINARY_OP,
        binary_op=BinaryOp.DIV,
        left=ExprNode(NodeType.CONSTANT, value=1.0),
        right=ExprNode(NodeType.VARIABLE)  # Will be 0
    )
    code = export_python(expr, safe=True)
    exec(code, globals())

    result = ppf_model(0.0)
    assert np.isfinite(result)  # Must not be inf or nan
```

### Determinism Tests

**Purpose**: Verify reproducible output.

```python
def test_export_determinism():
    expr = ExprNode(...)

    code1 = export_python(expr)
    code2 = export_python(expr)

    assert code1 == code2  # Identical output
```

---

## Golden Tests

Golden tests compare output against known-good references stored in `tests/golden/`.

| File | Purpose |
|------|---------|
| `damped_oscillator.py` | Reference Python export |
| `damped_oscillator.c` | Reference C export |
| `damped_oscillator.json` | Reference JSON bundle |

**Why golden tests**:
- Catch unintended output changes
- Document expected behavior
- Regression detection

---

## Benchmark Tests

Located in `tests/benchmark_*.py`:

| Benchmark | Tests |
|-----------|-------|
| `benchmark_symbolic.py` | Recovery of known forms |
| `benchmark_comparison.py` | Comparison with alternatives |

**Metrics tracked**:
- R² on test set
- Expression complexity
- Runtime
- False positive rate (wrong forms on noise)

---

## Running Tests

### Prerequisites

```bash
pip install pytest pytest-cov numpy scipy
```

### Commands

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=ppf --cov-report=html

# Specific test file
pytest tests/test_symbolic.py -v

# Specific test function
pytest tests/test_symbolic.py::test_discover_damped_sine -v

# Only fast tests (skip benchmarks)
pytest tests/ -v -m "not slow"

# Parallel execution
pytest tests/ -v -n auto
```

### Expected Output

```
============================= test session starts =============================
collected 143 items

tests/test_detector.py::test_fit_sine PASSED
tests/test_detector.py::test_fit_linear PASSED
...
tests/test_symbolic.py::test_discover_polynomial PASSED
tests/test_symbolic.py::test_discover_damped_sine PASSED
...
tests/test_export_python.py::TestPythonExportBasic::test_constant PASSED
tests/test_export_python.py::TestPythonExportBasic::test_variable PASSED
...

============================= 143 passed in 12.34s ============================
```

---

## Continuous Integration

Recommended CI configuration (GitHub Actions):

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install numpy scipy pytest pytest-cov
    - name: Run tests
      run: pytest tests/ -v --cov=ppf
```

---

## Adding New Tests

### Test Naming Convention

```
test_<module>_<functionality>.py
    test_<specific_case>()
```

### Test Template

```python
import pytest
import numpy as np
from ppf import SymbolicRegressor, DiscoveryMode

class TestNewFeature:
    """Tests for new feature X."""

    def test_basic_case(self):
        """Basic functionality works."""
        # Arrange
        x = np.linspace(0, 10, 100)
        y = some_formula(x)

        # Act
        result = new_feature(x, y)

        # Assert
        assert result.metric > threshold

    def test_edge_case(self):
        """Edge case is handled."""
        pass

    @pytest.mark.slow
    def test_benchmark(self):
        """Performance benchmark (slow)."""
        pass
```

---

## Known Limitations

1. **Stochastic tests**: GP is stochastic, so some discovery tests may occasionally fail. Use fixed seeds for reproducibility.

2. **Benchmark timing**: Timing tests may vary across hardware. Focus on relative comparisons.

3. **Large data tests**: Some tests with OEIS data require the `oeis_stripped.txt` file (~80MB).
