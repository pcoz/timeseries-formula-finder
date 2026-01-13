# PPF - Symbolic Form Discovery

[![PyPI version](https://img.shields.io/pypi/v/timeseries-formula-finder.svg)](https://pypi.org/project/timeseries-formula-finder/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Discover interpretable mathematical formulas from data, then deploy them anywhere.**

PPF (Promising Partial Form) is a **law-aware** symbolic regression library that finds compact mathematical expressions describing your data. Unlike black-box neural networks, PPF produces human-readable formulas grounded in physical reality that can be deployed to edge devices in under 100 bytes.

## Core Philosophy

A **Promising Partial Form** is a mathematical expression that explains a meaningful portion of a signal with high explanatory power and low complexity. Complex signals are modeled as layered sums:

```
y(t) ≈ f₁(t) + f₂(t) + ... + fₖ(t) + ε(t)
```

where each `fᵢ(t)` is a discovered form (oscillation, decay, trend, etc.) and `ε(t)` is noise-like residual.

PPF repeatedly **discovers → subtracts → analyzes residuals** to reveal the layered mathematical structure hidden in data. This approach applies to any 1D time-series or signal analysis task - from real-time sensor monitoring to scientific data exploration to serving as an interpretable front-end to downstream AI systems.

## Why PPF?

| Traditional ML | PPF Approach |
|----------------|--------------|
| Train neural network | Discover formula |
| Deploy 10-100KB model | Deploy 50-byte expression |
| 1000s of MACs/inference | 10 FLOPs/evaluation |
| Black box | Interpretable |
| Requires TensorFlow Lite | Runs on any microcontroller |

**Key insight**: Many real-world signals (sensor data, biological rhythms, physical systems) follow simple mathematical forms. PPF attempts to discover those forms automatically.

**Information-theoretic view**: Where neural networks compress data into opaque weight matrices, PPF compresses data into explicit equations. This enables a layered architecture: `Raw Signal → PPF (laws) → ML (meaning) → Decisions`, where PPF converts high-entropy measurements into low-entropy symbolic representations.

## Quick Start

```bash
pip install timeseries-formula-finder
```

```python
import numpy as np
from ppf import SymbolicRegressor, export_python, export_c

# Generate example data
t = np.linspace(0, 10, 200)
y = 2.5 * np.exp(-0.3 * t) * np.sin(4.0 * t + 0.5) + np.random.randn(200) * 0.1

# Discover the underlying formula
regressor = SymbolicRegressor(generations=30)
result = regressor.discover(t, y, verbose=True)

print(f"Discovered: {result.best_tradeoff.expression_string}")
print(f"R-squared:  {result.best_tradeoff.r_squared:.4f}")
# Output: Discovered: 2.498*exp(-0.301*x)*sin(4.002*x + 0.497)
# Output: R-squared:  0.9847

# Export to standalone Python (no PPF dependency)
python_code = export_python(result.best_tradeoff.expression, fn_name="predict")
exec(python_code)
print(predict(1.5))  # Works without importing ppf

# Export to C for embedded deployment
c_code = export_c(result.best_tradeoff.expression, use_float=True)
# Compile with: gcc -std=c99 -O2 -lm model.c
```

## Features

### Multi-Domain Discovery

PPF includes domain-specific "vocabularies" that guide the search:

```python
from ppf import SymbolicRegressor, DiscoveryMode

regressor = SymbolicRegressor()

# Let PPF auto-detect the best domain
result = regressor.discover(x, y, mode=DiscoveryMode.AUTO)

# Or specify a domain if you know it
result = regressor.discover(x, y, mode=DiscoveryMode.OSCILLATOR)   # Vibrations, waves
result = regressor.discover(x, y, mode=DiscoveryMode.CIRCUIT)      # RC charging, decay
result = regressor.discover(x, y, mode=DiscoveryMode.GROWTH)       # Sigmoids, saturation
result = regressor.discover(x, y, mode=DiscoveryMode.RATIONAL)     # Ratios, feedback
result = regressor.discover(x, y, mode=DiscoveryMode.UNIVERSAL)    # Power laws, Gaussians
```

### Macro Templates

PPF includes "macros" - pre-composed functional forms that capture common physics:

| Macro | Formula | Use Case |
|-------|---------|----------|
| `DAMPED_SIN` | `a·exp(-k·t)·sin(ω·t + φ)` | Vibration decay |
| `RC_CHARGE` | `a·(1 - exp(-k·t)) + c` | Capacitor charging |
| `GAUSSIAN` | `a·exp(-((x-μ)/σ)²)` | Peaks, distributions |
| `SIGMOID` | `a / (1 + exp(-k·(x-x₀)))` | Saturation curves |
| `POWER_LAW` | `a·x^b + c` | Scaling phenomena |
| `HILL` | `a·x^n / (k^n + x^n)` | Enzyme kinetics |

These let PPF find complex forms (multiplicative compositions across function families) that pure GP struggles with.

### Export to Edge Devices

Discovered formulas export to production-ready code:

```python
from ppf import export_python, export_c, export_json

# Standalone Python evaluator
code = export_python(expr, fn_name="predict_temp", safe=True)
# Includes safety wrappers for div-by-zero, log(0), exp overflow

# C99 for microcontrollers
code = export_c(expr, use_float=True, safe=True)
# Compiles on ESP32, STM32, Arduino, etc.

# JSON bundle for storage/transmission
bundle = export_json(result, variables=["t"])
# Send via MQTT, store in database, audit trail
```

### Feature Extraction for Downstream ML

Use discovered forms as features for classification/regression:

```python
from ppf import extract_features, feature_vector

# Extract interpretable features
features = extract_features(result)
print(features["dominant_family"])  # "oscillation"
print(features["damping_k"])        # 0.301
print(features["omega"])            # 4.002

# Convert to ML-ready vector
vec, names = feature_vector(features, schema="ppf.features.v1.full")
# Feed to sklearn, XGBoost, etc.
```

## Installation

### From PyPI (recommended)

```bash
pip install timeseries-formula-finder
```

### From source

```bash
git clone https://github.com/pcoz/timeseries-formula-finder.git
cd timeseries-formula-finder
pip install -e .
```

### Optional dependencies

```bash
# For hybrid decomposition (EMD/SSA)
pip install timeseries-formula-finder[hybrid]

# For development (tests)
pip install timeseries-formula-finder[dev]
```

## Documentation

| Document | Description |
|----------|-------------|
| [USER_GUIDE.md](USER_GUIDE.md) | Complete usage guide with examples |
| [docs/PPF_Paper.md](docs/PPF_Paper.md) | Core PPF concepts and architecture |
| [docs/PPF_Information_Theory_Paper.md](docs/PPF_Information_Theory_Paper.md) | Information-theoretic foundations |
| [COMPARISON.md](COMPARISON.md) | How PPF compares to PySR, gplearn, Eureqa, AI Feynman |
| [USE_CASES.md](USE_CASES.md) | Edge AI, ECG analysis, predictive maintenance |
| [TESTING.md](TESTING.md) | Test suite details and datasets used |
| [docs/PPF_EXPORT_LAYER_TSD.md](docs/PPF_EXPORT_LAYER_TSD.md) | Export layer technical specification |
| [docs/IOT_SENSOR_ANALYSIS.md](docs/IOT_SENSOR_ANALYSIS.md) | Real IoT sensor case study |
| [docs/ECG_ANALYSIS.md](docs/ECG_ANALYSIS.md) | ECG waveform analysis case study |
| [DOCUMENTATION.md](DOCUMENTATION.md) | Comprehensive API documentation |

## Use Cases

### Edge AI / IoT Sensors

Deploy temperature prediction models to ESP32 in 50 bytes instead of TensorFlow Lite:

```python
# Discover daily temperature cycle from sensor data
result = regressor.discover(hours, temp, mode=DiscoveryMode.OSCILLATOR)
# Found: T(t) = 36 + 5·cos(2π·t/24 - 2)  [R² = 0.58]

# Export to C for microcontroller
c_code = export_c(result.best_tradeoff.expression, use_float=True)
# 3 parameters, 10 FLOPs, runs at 100kHz on ESP32
```

### Biomedical Signal Analysis

Extract interpretable features from ECG waveforms:

```python
# Analyze T-wave morphology
result = regressor.discover(t_wave_time, t_wave_amplitude, mode=DiscoveryMode.UNIVERSAL)
# Found: Damped cosine (R² = 0.96) - indicates repolarization dynamics

# Use parameters as cardiac health features
features = extract_features(result)
# damping_k, omega → feed to arrhythmia classifier
```

### Predictive Maintenance

Classify machine health from vibration signatures:

```python
# Healthy machine: clean sinusoid
# Bearing fault: damped oscillations (impact response)
# Imbalance: dominant 1x RPM

# The discovered FORM is the diagnosis
result = regressor.discover(t, vibration, mode=DiscoveryMode.OSCILLATOR)
if "exp(-" in result.best_tradeoff.expression_string:
    print("Bearing fault detected - damped oscillation signature")
```

## Architecture

```
ppf/
├── symbolic_types.py     # Expression trees, macros, primitives
├── symbolic.py           # GP engine, symbolic regressor
├── symbolic_utils.py     # Printing, simplification
├── detector.py           # Fixed-form fitting (legacy)
├── residual_layer.py     # Multi-layer decomposition
├── hierarchical.py       # Windowed parameter evolution
├── hybrid.py             # EMD/SSA + PPF interpretation
├── export/
│   ├── python_export.py  # export_python()
│   ├── c_export.py       # export_c()
│   ├── json_export.py    # export_json()
│   └── load.py           # load_json()
└── features/
    ├── extract.py        # extract_features()
    └── vectorize.py      # feature_vector()
```

## Benchmarks

Performance on standard symbolic regression benchmarks:

| Benchmark | PPF R² | Complexity | Notes |
|-----------|--------|------------|-------|
| Kepler's 3rd Law | 0.9999 | 3 | `T² = a³` |
| Damped Oscillator | 0.9847 | 8 | Macro template |
| Logistic Growth | 0.9923 | 5 | Sigmoid macro |
| Nguyen-1 (x³+x²+x) | 0.9998 | 7 | Polynomial mode |

See [BENCHMARK_COMPARISON.md](BENCHMARK_COMPARISON.md) for detailed results.

## License

MIT License - see [LICENSE](LICENSE) for details.

Copyright (c) 2026 Edward Chalk ([fleetingswallow.com](https://fleetingswallow.com))

## Citation

If you use PPF in research, please cite:

```bibtex
@software{timeseries-formula-finder,
  author = {Chalk, Edward},
  title = {Timeseries Formula Finder: Discover Mathematical Forms in Time-Series Data},
  year = {2026},
  url = {https://github.com/pcoz/timeseries-formula-finder}
}
```

## Contact

- Website: [fleetingswallow.com](https://fleetingswallow.com)
- Email: edward@fleetingswallow.com
