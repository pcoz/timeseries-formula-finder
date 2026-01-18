# PPF User Guide

A complete guide to using PPF for time-series analysis and symbolic form discovery.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Command-Line Interface](#2-command-line-interface)
3. [Core Concepts](#3-core-concepts)
4. [API Reference](#4-api-reference)
5. [Discovery Modes](#5-discovery-modes)
6. [Understanding Results](#6-understanding-results)
7. [Export for Deployment](#7-export-for-deployment)
8. [Feature Extraction](#8-feature-extraction)
9. [Common Use Cases](#9-common-use-cases)
10. [Troubleshooting](#10-troubleshooting)
11. [Best Practices](#11-best-practices)

---

## 1. Quick Start

### Installation

```bash
pip install timeseries-formula-finder

# Optional: for hybrid EMD/SSA decomposition
pip install timeseries-formula-finder[hybrid]
```

### Your First Discovery

```python
import numpy as np
from ppf import SymbolicRegressor

# Create some data with a known pattern
t = np.linspace(0, 10, 200)
y = 2.5 * np.sin(3.0 * t + 0.5) + 0.1 * np.random.randn(200)

# Discover the underlying form
regressor = SymbolicRegressor(generations=30)
result = regressor.discover(t, y, verbose=True)

# See what was found
print(f"Discovered: {result.best_tradeoff.expression_string}")
print(f"R-squared:  {result.best_tradeoff.r_squared:.4f}")
```

**Expected output:**
```
Gen   0: R²=0.15, MSE=2.1e+00, Complexity=3
Gen  10: R²=0.78, MSE=5.2e-01, Complexity=5
Gen  20: R²=0.94, MSE=1.4e-01, Complexity=7
Gen  30: R²=0.97, MSE=6.8e-02, Complexity=6
Discovered: 2.49*sin(3.01*x + 0.51)
R-squared:  0.9734
```

---

## 2. Command-Line Interface

PPF includes a command-line interface for quick analysis without writing Python code.

### Installation

After installing PPF, the `ppf` command is available:

```bash
pip install timeseries-formula-finder

# Verify installation
ppf --help
```

You can also run via Python module:

```bash
python -m ppf --help
```

### Quick Examples

```bash
# Discover formulas from a CSV file
ppf discover data.csv -x time -y signal

# Use a specific discovery mode with verbose output
ppf discover data.csv --mode oscillator -v -g 100

# Detect mathematical forms in data windows
ppf detect sensor.csv --min-r-squared 0.8

# Extract forms layer by layer until residuals are noise
ppf stack data.csv --entropy-method spectral

# Hierarchical pattern analysis at multiple timescales
ppf hierarchy data.csv --window-size 100

# Signal decomposition with SSA (no extra dependencies)
ppf hybrid data.csv --method ssa

# Export discovered formula to Python code
ppf --json discover data.csv | ppf export python -f predict > model.py

# Export to C for embedded systems
ppf --json discover data.csv | ppf export c -f sensor_model --float > model.h

# Show available discovery modes
ppf info modes

# Show all macro templates
ppf info macros
```

### Available Commands

| Command | Description |
|---------|-------------|
| `discover` | Symbolic regression to find mathematical formulas |
| `detect` | Detect mathematical forms in data windows |
| `stack` | Extract forms iteratively until residuals are noise |
| `hierarchy` | Find nested patterns at multiple timescales |
| `hybrid` | Combine EMD/SSA decomposition with form interpretation |
| `export` | Export expressions to Python, C, or JSON |
| `features` | Extract ML-ready features from discovery results |
| `info` | Show available modes, forms, macros, and methods |

### Global Options

These options work with all commands:

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Show detailed progress and debug information |
| `-q, --quiet` | Suppress all output except errors and results |
| `--json` | Output results in JSON format (for piping) |
| `-o, --output FILE` | Write output to FILE instead of stdout |
| `--version` | Show version information |
| `-h, --help` | Show help message |

### Data Input Options

Most commands accept these data input options:

```bash
ppf discover data.csv                    # From file
ppf discover data.csv -x time -y signal  # Specify columns by name
ppf discover data.csv -x 0 -y 1          # Specify columns by index
cat data.csv | ppf discover --stdin      # From stdin
ppf discover data.csv --delimiter ";"    # Custom delimiter
ppf discover data.csv --skip-header 2    # Skip header rows
```

### The Discover Command

The most commonly used command for symbolic regression:

```bash
ppf discover data.csv [options]
```

**Key options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-m, --mode` | auto | Discovery mode (auto, oscillator, circuit, growth, etc.) |
| `-g, --generations` | 50 | Number of evolutionary generations |
| `-p, --population-size` | 500 | Population size for genetic programming |
| `--max-depth` | 6 | Maximum expression tree depth |
| `--parsimony` | 0.001 | Complexity penalty coefficient |
| `--random-state` | None | Random seed for reproducibility |
| `--show-pareto` | off | Show full Pareto front of solutions |
| `--simplify` | off | Simplify expressions before output |
| `--latex` | off | Include LaTeX-formatted expressions |

**Example output:**

```
============================================================
SYMBOLIC REGRESSION RESULTS
============================================================

Generations: 50
Evaluations: 25,000

BEST TRADEOFF (recommended):
----------------------------------------
  Expression: 2.499*exp(-0.301*x)*sin(4.002*x + 0.497)
  R-squared:  0.9847
  MSE:        1.53e-02
  Complexity: 8
  Depth:      3

MOST ACCURATE:
----------------------------------------
  Expression: 2.501*exp(-0.300*x)*sin(4.001*x + 0.498) + 0.002*x
  R-squared:  0.9851
  ...
```

### The Export Command

Export discovered expressions to deployable code:

```bash
# Export to Python
ppf --json discover data.csv | ppf export python -f my_model

# Export to C for embedded systems
ppf --json discover data.csv | ppf export c -f sensor_model --float

# Export/transform JSON
ppf --json discover data.csv | ppf export json --source "experiment_1"
```

**Python export options:**

| Option | Description |
|--------|-------------|
| `-f, --function-name` | Name for generated function (default: evaluate) |
| `--variable` | Variable name in code (default: x) |
| `--safe` | Add bounds checking and error handling |

**C export options:**

| Option | Description |
|--------|-------------|
| `-f, --function-name` | Name for generated function |
| `--variable` | Variable name in code |
| `--float` | Use float instead of double |
| `--macro-style` | Generate as preprocessor macro |
| `--safe` | Add bounds checking |

### Workflow Examples

**Basic discovery and export:**

```bash
# 1. Discover a formula
ppf discover sensor_data.csv -x time -y temp --mode auto -v

# 2. If satisfied, export to code
ppf --json discover sensor_data.csv | ppf export python -f predict_temp > model.py
```

**Batch processing:**

```bash
# Process multiple files
for file in data/*.csv; do
    echo "Processing $file..."
    ppf --json discover "$file" > "${file%.csv}_model.json"
done
```

**Inspection workflow:**

```bash
# Check what modes are available
ppf info modes

# Check what macros exist
ppf info macros

# Try different modes
ppf discover data.csv --mode oscillator -v
ppf discover data.csv --mode growth -v
ppf discover data.csv --mode universal -v
```

For complete CLI documentation, see [docs/CLI.md](docs/CLI.md).

---

## 3. Core Concepts

### What is a "Promising Partial Form"?

A **Promising Partial Form (PPF)** is a mathematical expression that:
- Explains a **meaningful portion** of a signal
- Has **high explanatory power** (R² above threshold)
- Has **low complexity** (parsimony)
- Maps to a **recognizable mathematical family**

### Layered Decomposition

Complex signals are modeled as sums of partial forms:

```
y(t) ≈ f₁(t) + f₂(t) + ... + fₖ(t) + ε(t)
```

Where:
- `fᵢ(t)` = Discovered partial form (oscillation, decay, trend, etc.)
- `ε(t)` = Noise-like residual
- `k` = Number of layers (determined automatically)

### The Discovery Loop

```
Input Signal → DISCOVER (find best form)
                    ↓
              VALIDATE (check R² > threshold)
                    ↓
              SUBTRACT (compute residual)
                    ↓
              TEST (is residual noise-like?)
                    ↓
         YES: Done    NO: Repeat with residual
```

---

## 4. API Reference

### SymbolicRegressor

The main entry point for form discovery.

```python
from ppf import SymbolicRegressor, DiscoveryMode

regressor = SymbolicRegressor(
    population_size=500,      # Number of candidate expressions per generation
    generations=50,           # Number of evolutionary generations
    max_depth=6,              # Maximum expression tree depth
    parsimony_coefficient=0.001,  # Complexity penalty (higher = simpler forms)
    tournament_size=7,        # Selection pressure
    crossover_prob=0.9,       # Probability of crossover vs mutation
    random_state=None         # For reproducibility
)
```

#### Parameters Explained

| Parameter | Default | Effect |
|-----------|---------|--------|
| `population_size` | 500 | More = better exploration, slower |
| `generations` | 50 | More = better convergence, slower |
| `max_depth` | 6 | Higher = more complex expressions possible |
| `parsimony_coefficient` | 0.001 | Higher = prefer simpler expressions |
| `tournament_size` | 7 | Higher = more selection pressure |
| `random_state` | None | Set for reproducible results |

#### The `discover()` Method

```python
result = regressor.discover(
    x,                        # Independent variable (1D numpy array)
    y,                        # Dependent variable (1D numpy array)
    mode=DiscoveryMode.AUTO,  # Discovery mode (see Section 4)
    verbose=False             # Print progress during search
)
```

**Returns:** `SymbolicRegressionResult` object (see Section 5)

### Export Functions

```python
from ppf import export_python, export_c, export_json, load_json

# Export to standalone Python
code = export_python(result.best_tradeoff.expression, fn_name="predict")

# Export to C99
code = export_c(result.best_tradeoff.expression, use_float=True)

# Export to JSON (for storage/transmission)
bundle = export_json(result, variables=["t"])

# Load from JSON
loaded_result = load_json(bundle)
```

### Feature Extraction

```python
from ppf import extract_features, feature_vector

# Extract interpretable features
features = extract_features(result)

# Convert to ML-ready vector
vec, names = feature_vector(features)
```

---

## 5. Discovery Modes

PPF supports multiple discovery modes optimized for different signal types.

### Mode Overview

| Mode | Best For | Primitives Used |
|------|----------|-----------------|
| `AUTO` | Unknown signals | Probes all, selects best |
| `OSCILLATOR` | Vibrations, waves | sin, cos, damped forms |
| `CIRCUIT` | Electronics | exp, RC charge/discharge |
| `GROWTH` | Biology, adoption | sigmoid, logistic, Hill |
| `RATIONAL` | Feedback systems | ratio, rational functions |
| `POLYNOMIAL` | Algebraic relationships | add, mul, powers |
| `UNIVERSAL` | General patterns | power law, Gaussian, tanh |
| `DISCOVER` | Novel forms | All primitives, no macros |
| `IDENTIFY` | Known form families | All macros |

### When to Use Each Mode

#### AUTO (Recommended Default)
```python
result = regressor.discover(x, y, mode=DiscoveryMode.AUTO)
```
- **Use when:** You don't know the signal type
- **How it works:** Runs quick probes on each domain, then full search on the best
- **Tradeoff:** Slightly slower but most robust

#### OSCILLATOR
```python
result = regressor.discover(x, y, mode=DiscoveryMode.OSCILLATOR)
```
- **Use when:** Signal appears periodic or has ringing behavior
- **Includes macros:** `DAMPED_SIN`, `DAMPED_COS`
- **Good for:** Mechanical vibrations, audio envelopes, RLC circuits

#### CIRCUIT
```python
result = regressor.discover(x, y, mode=DiscoveryMode.CIRCUIT)
```
- **Use when:** Signal looks like charging/discharging curves
- **Includes macros:** `RC_CHARGE`, `EXP_DECAY`
- **Good for:** RC circuits, temperature equilibration, chemical reactions

#### GROWTH
```python
result = regressor.discover(x, y, mode=DiscoveryMode.GROWTH)
```
- **Use when:** Signal shows S-curve or saturation behavior
- **Includes macros:** `SIGMOID`, `LOGISTIC`, `HILL`
- **Good for:** Population growth, technology adoption, enzyme kinetics

#### RATIONAL
```python
result = regressor.discover(x, y, mode=DiscoveryMode.RATIONAL)
```
- **Use when:** Signal may involve ratios or transfer functions
- **Includes macros:** `RATIO`, `RATIONAL2`
- **Good for:** Control systems, Michaelis-Menten kinetics

#### UNIVERSAL
```python
result = regressor.discover(x, y, mode=DiscoveryMode.UNIVERSAL)
```
- **Use when:** Signal might be a peak, power law, or step
- **Includes macros:** `POWER_LAW`, `GAUSSIAN`, `TANH_STEP`
- **Good for:** Spectra, scaling phenomena, threshold effects

#### DISCOVER
```python
result = regressor.discover(x, y, mode=DiscoveryMode.DISCOVER)
```
- **Use when:** You want to find novel compositions
- **No macros:** Pure GP from primitives
- **Good for:** Research, finding unexpected patterns
- **Note:** Slower, may not find multiplicative compositions

#### IDENTIFY
```python
result = regressor.discover(x, y, mode=DiscoveryMode.IDENTIFY)
```
- **Use when:** You know the data follows a standard form
- **All macros:** Maximum template coverage
- **Good for:** Quick identification of known patterns

---

## 6. Understanding Results

### The SymbolicRegressionResult Object

```python
result = regressor.discover(x, y)

# Three key solutions from the Pareto front:
result.most_accurate       # Highest R², may be complex
result.most_parsimonious   # Simplest form above threshold
result.best_tradeoff       # Recommended: best accuracy/complexity balance

# Full Pareto front (all non-dominated solutions)
result.pareto_front        # List of SymbolicFitResult objects
```

### The SymbolicFitResult Object

Each solution contains:

```python
fit = result.best_tradeoff

fit.expression            # ExprNode: the expression tree
fit.expression_string     # str: human-readable formula
fit.r_squared            # float: coefficient of determination (0-1)
fit.mse                  # float: mean squared error
fit.complexity           # int: number of nodes in tree

# Evaluate on new data
y_pred = fit.evaluate(x_new)
```

### Interpreting R-squared

| R² Value | Interpretation |
|----------|---------------|
| > 0.99 | Excellent fit, likely correct form |
| 0.95-0.99 | Very good fit |
| 0.90-0.95 | Good fit, may have noise or missing components |
| 0.80-0.90 | Moderate fit, consider layered decomposition |
| < 0.80 | Poor fit, signal may not follow discoverable form |

### The Pareto Front

The Pareto front shows the accuracy-complexity tradeoff:

```
    R²
     ^
 1.0 |           * most_accurate (complex)
     |         *
     |       *   <- Pareto front
 0.8 |     *
     |   * <- best_tradeoff (knee)
 0.6 | *
     |* most_parsimonious (simple)
 0.4 +-------------------------> Complexity
     1   3   5   7   9   11
```

**Choosing a solution:**
- `most_accurate`: When prediction accuracy is paramount
- `most_parsimonious`: When interpretability/deployment size matters
- `best_tradeoff`: General recommendation (knee of the curve)

---

## 7. Export for Deployment

### Python Export

Generate standalone Python code with no PPF dependency:

```python
from ppf import export_python

code = export_python(
    result.best_tradeoff.expression,
    fn_name="predict_temperature",  # Function name
    signature=("t",),               # Variable names
    safe=True                       # Include safety wrappers
)

print(code)
```

**Output:**
```python
import math

def safe_div(a, b):
    return a / b if abs(b) > 1e-10 else a / 1e-10

def safe_log(x):
    return math.log(max(abs(x), 1e-10))

def predict_temperature(t):
    return 2.49 * math.sin(3.01 * t + 0.51)
```

**Using the exported code:**
```python
exec(code)
print(predict_temperature(1.5))  # Works without importing ppf
```

### C Export

Generate C99 code for embedded deployment:

```python
from ppf import export_c

code = export_c(
    result.best_tradeoff.expression,
    fn_name="predict",
    signature=("double t",),  # C type declarations
    use_float=True,           # Use float instead of double
    safe=True,                # Include safety wrappers
    macro_style="inline"      # "inline" or "helper"
)

print(code)
```

**Output:**
```c
#include <math.h>

float predict(float t) {
    return 2.49f * sinf(3.01f * t + 0.51f);
}
```

**Compilation:**
```bash
gcc -std=c99 -O2 -lm -c model.c
```

### JSON Export

Serialize for storage, transmission, or audit trail:

```python
from ppf import export_json, load_json
import json

# Export
bundle = export_json(
    result,
    variables=["t"],
    include_metadata=True
)

# Save to file
with open("model.json", "w") as f:
    json.dump(bundle, f, indent=2)

# Load back
with open("model.json", "r") as f:
    loaded = json.load(f)

restored = load_json(loaded)
y_pred = restored.evaluate(x_new)
```

**JSON structure:**
```json
{
  "schema": "ppf.expression.v1",
  "tree": { ... },
  "metrics": {
    "r_squared": 0.9734,
    "mse": 0.068,
    "complexity": 6
  },
  "expression_string": "2.49*sin(3.01*x + 0.51)",
  "variables": ["t"]
}
```

---

## 8. Feature Extraction

Use discovered forms as interpretable features for downstream ML.

### Extracting Features

```python
from ppf import extract_features

features = extract_features(result)

print(features)
```

**Output:**
```python
{
    'dominant_family': 'oscillation',
    'r2': 0.9734,
    'complexity': 6,
    'amplitude': 2.49,
    'omega': 3.01,
    'phase': 0.51,
    'damping_k': None,  # Not a damped oscillation
    'K': None,          # No saturation
    'mu': None,         # No Gaussian peak
    'sigma': None
}
```

### Feature Meanings

| Feature | Type | Meaning |
|---------|------|---------|
| `dominant_family` | str | Form type: "oscillation", "decay", "growth", "polynomial", "peak" |
| `r2` | float | Explanatory power (0-1) |
| `complexity` | int | Expression size |
| `amplitude` | float | Signal magnitude |
| `omega` / `frequency` | float | Oscillation rate (rad/s or Hz) |
| `damping_k` | float | Decay rate (1/time) |
| `phase` | float | Phase offset (radians) |
| `K` / `L` | float | Saturation limit |
| `mu` | float | Peak/center location |
| `sigma` | float | Peak width |

### Creating Feature Vectors

```python
from ppf import feature_vector

# Single result
vec, names = feature_vector(features)
print(f"Feature names: {names}")
print(f"Feature values: {vec}")

# Multiple results (for classification)
all_features = [extract_features(r) for r in results]
X = np.array([feature_vector(f)[0] for f in all_features])

# Use with sklearn
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, labels)
```

### Feature Schemas

```python
from ppf import feature_vector

# Minimal schema (fewer features, no NaNs)
vec, names = feature_vector(features, schema="ppf.features.v1.minimal")

# Full schema (all possible features)
vec, names = feature_vector(features, schema="ppf.features.v1.full")
```

---

## 9. Common Use Cases

### 8.1 IoT Sensor Analysis

**Scenario:** Predict temperature from historical sensor data.

```python
import numpy as np
from ppf import SymbolicRegressor, DiscoveryMode, export_c

# Load sensor data (hours, temperature)
hours = np.array([0, 1, 2, ..., 23, 24, ...])
temp = np.array([18.2, 17.8, 17.5, ..., 25.1, 24.8, ...])

# Discover daily pattern
regressor = SymbolicRegressor(generations=40)
result = regressor.discover(hours, temp, mode=DiscoveryMode.OSCILLATOR, verbose=True)

print(f"Model: {result.best_tradeoff.expression_string}")
print(f"R²: {result.best_tradeoff.r_squared:.3f}")

# Export for ESP32
c_code = export_c(result.best_tradeoff.expression, use_float=True)
# Typical result: T(h) = 21.5 + 4.2*cos(0.262*h - 2.1)
# 4 parameters, ~50 bytes, runs at 100kHz
```

### 8.2 Vibration Analysis / Predictive Maintenance

**Scenario:** Classify machine health from vibration signatures.

```python
import numpy as np
from ppf import SymbolicRegressor, DiscoveryMode, extract_features

def analyze_machine(vibration_signal, time):
    regressor = SymbolicRegressor(generations=30)
    result = regressor.discover(time, vibration_signal,
                                 mode=DiscoveryMode.OSCILLATOR)

    features = extract_features(result)
    expr_str = result.best_tradeoff.expression_string

    # Diagnosis based on form type
    if features['damping_k'] is not None and features['damping_k'] > 0.1:
        return "BEARING_FAULT", "Damped oscillation indicates impact response"
    elif features['dominant_family'] == 'oscillation':
        return "HEALTHY", "Clean sinusoidal vibration"
    else:
        return "UNKNOWN", "Unusual vibration pattern"

# Use
status, reason = analyze_machine(vibration_data, time_data)
print(f"Machine status: {status}")
print(f"Reason: {reason}")
```

### 8.3 ECG/Biomedical Waveform Analysis

**Scenario:** Extract morphology features from ECG T-waves.

```python
import numpy as np
from ppf import SymbolicRegressor, DiscoveryMode, extract_features

def analyze_t_wave(t_wave_time, t_wave_amplitude):
    regressor = SymbolicRegressor(generations=30)
    result = regressor.discover(t_wave_time, t_wave_amplitude,
                                 mode=DiscoveryMode.UNIVERSAL)

    features = extract_features(result)

    # T-wave morphology features for arrhythmia detection
    return {
        'form_type': features['dominant_family'],
        'amplitude': features.get('amplitude'),
        'width': features.get('sigma'),  # If Gaussian
        'symmetry': features.get('damping_k'),  # If damped
        'r_squared': features['r2']
    }

# Use in classification pipeline
ecg_features = [analyze_t_wave(t, amp) for t, amp in t_waves]
X = np.array([[f['amplitude'], f['width'], f['r_squared']] for f in ecg_features])
# Feed to arrhythmia classifier
```

### 8.4 Scientific Law Discovery

**Scenario:** Discover physical relationships from experimental data.

```python
import numpy as np
from ppf import SymbolicRegressor, DiscoveryMode

# Orbital data: period vs semi-major axis
periods = np.array([0.24, 0.62, 1.0, 1.88, 11.86, 29.46])  # years
distances = np.array([0.39, 0.72, 1.0, 1.52, 5.2, 9.54])   # AU

# Discover Kepler's Third Law: T² ∝ a³
regressor = SymbolicRegressor(generations=50)
result = regressor.discover(distances, periods**2,
                            mode=DiscoveryMode.POLYNOMIAL, verbose=True)

print(f"Discovered: T² = {result.best_tradeoff.expression_string}")
# Expected: T² ≈ a³ (Kepler's Third Law)
```

### 8.5 Signal Denoising via Form Extraction

**Scenario:** Extract clean signal by fitting a form and evaluating it.

```python
import numpy as np
from ppf import SymbolicRegressor

# Noisy signal
t = np.linspace(0, 10, 500)
y_clean = 3.0 * np.exp(-0.2 * t) * np.sin(2.5 * t)
y_noisy = y_clean + 0.3 * np.random.randn(500)

# Discover form
regressor = SymbolicRegressor(generations=40)
result = regressor.discover(t, y_noisy, mode=DiscoveryMode.OSCILLATOR)

# Evaluate form to get denoised signal
y_denoised = result.best_tradeoff.evaluate(t)

# Compare
print(f"Noise reduction: {np.std(y_noisy - y_clean):.3f} -> {np.std(y_denoised - y_clean):.3f}")
```

### 8.6 Real-Time Monitoring with Deployed Model

**Scenario:** Deploy discovered model for online prediction.

```python
# During development: discover and export
from ppf import SymbolicRegressor, export_python

result = regressor.discover(historical_time, historical_values)
code = export_python(result.best_tradeoff.expression, fn_name="predict")

with open("deployed_model.py", "w") as f:
    f.write(code)

# In production: use exported model (no PPF dependency)
from deployed_model import predict

while True:
    current_time = get_current_time()
    predicted_value = predict(current_time)

    actual_value = read_sensor()
    residual = actual_value - predicted_value

    if abs(residual) > threshold:
        alert("Anomaly detected!")
```

---

## 10. Troubleshooting

### Low R² Values

**Symptom:** R² < 0.8 despite visible pattern in data.

**Possible causes and solutions:**

1. **Wrong discovery mode**
   ```python
   # Try AUTO mode to let PPF choose
   result = regressor.discover(x, y, mode=DiscoveryMode.AUTO)
   ```

2. **Insufficient generations**
   ```python
   regressor = SymbolicRegressor(generations=100)  # Increase from default 50
   ```

3. **Multi-component signal** - use layered decomposition
   ```python
   from ppf import PPFResidualLayer
   layer = PPFResidualLayer()
   result = layer.analyze(y)
   # Returns stack of forms
   ```

4. **Non-stationary parameters** - use hierarchical decomposition
   ```python
   from ppf import HierarchicalDetector
   detector = HierarchicalDetector(window_size=100)
   result = detector.analyze(y)
   ```

### Search Doesn't Converge

**Symptom:** R² stays low across generations.

**Solutions:**

1. **Increase population size**
   ```python
   regressor = SymbolicRegressor(population_size=1000)
   ```

2. **Adjust parsimony coefficient**
   ```python
   # Lower = allow more complex solutions
   regressor = SymbolicRegressor(parsimony_coefficient=0.0001)
   ```

3. **Try different mode**
   ```python
   for mode in [DiscoveryMode.OSCILLATOR, DiscoveryMode.GROWTH, DiscoveryMode.UNIVERSAL]:
       result = regressor.discover(x, y, mode=mode)
       print(f"{mode}: R²={result.best_tradeoff.r_squared:.3f}")
   ```

### Export Produces NaN/Inf

**Symptom:** Exported function returns NaN or Inf for some inputs.

**Solution:** Enable safety wrappers
```python
code = export_python(expr, safe=True)  # Includes safe_div, safe_log, clamp_exp
code = export_c(expr, safe=True)
```

### Memory Issues with Large Data

**Symptom:** Out of memory errors with large datasets.

**Solution:** Subsample the data
```python
# Subsample to 1000 points
indices = np.linspace(0, len(x)-1, 1000, dtype=int)
x_sub = x[indices]
y_sub = y[indices]

result = regressor.discover(x_sub, y_sub)

# Verify on full data
y_pred = result.best_tradeoff.evaluate(x)
r2_full = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
```

---

## 11. Best Practices

### 1. Start with AUTO Mode

Unless you know the signal type, let PPF probe all domains:
```python
result = regressor.discover(x, y, mode=DiscoveryMode.AUTO)
```

### 2. Use Verbose Mode for Debugging

Watch the search progress to understand convergence:
```python
result = regressor.discover(x, y, verbose=True)
```

### 3. Check the Pareto Front

Don't just use `best_tradeoff` - examine alternatives:
```python
for fit in result.pareto_front:
    print(f"R²={fit.r_squared:.3f}, Complexity={fit.complexity}: {fit.expression_string}")
```

### 4. Validate on Held-Out Data

```python
# Split data
train_idx = np.random.choice(len(x), size=int(0.8*len(x)), replace=False)
test_idx = np.setdiff1d(np.arange(len(x)), train_idx)

# Discover on training data
result = regressor.discover(x[train_idx], y[train_idx])

# Validate on test data
y_pred = result.best_tradeoff.evaluate(x[test_idx])
test_r2 = 1 - np.sum((y[test_idx] - y_pred)**2) / np.sum((y[test_idx] - np.mean(y[test_idx]))**2)
print(f"Train R²: {result.best_tradeoff.r_squared:.3f}, Test R²: {test_r2:.3f}")
```

### 5. Set Random State for Reproducibility

```python
regressor = SymbolicRegressor(random_state=42)
```

### 6. Consider Physical Constraints

If you know constraints (e.g., decay rate must be positive), verify:
```python
features = extract_features(result)
if features.get('damping_k', 0) < 0:
    print("Warning: Negative decay rate - physically unrealistic")
```

### 7. Export with Safety for Production

Always use safety wrappers in production exports:
```python
code = export_c(expr, safe=True)  # Prevents NaN/Inf crashes
```

### 8. Document Discovered Forms

When you deploy a model, document what was found:
```python
bundle = export_json(result, variables=["time_hours"])
bundle['metadata'] = {
    'discovered_date': '2026-01-13',
    'data_source': 'temperature_sensor_001',
    'interpretation': 'Daily temperature cycle'
}
```

---

## Appendix: Quick Reference

### Import Statements

```python
# Core functionality
from ppf import SymbolicRegressor, DiscoveryMode

# Export
from ppf import export_python, export_c, export_json, load_json

# Features
from ppf import extract_features, feature_vector

# Advanced
from ppf import PPFResidualLayer, HierarchicalDetector, HybridDecomposer
```

### Discovery Modes Cheat Sheet

| Signal Looks Like... | Use Mode |
|---------------------|----------|
| Don't know | `AUTO` |
| Sine wave | `OSCILLATOR` |
| Ringing/damped | `OSCILLATOR` |
| Exponential curve | `CIRCUIT` |
| S-curve / saturation | `GROWTH` |
| Power law / scaling | `UNIVERSAL` |
| Gaussian peak | `UNIVERSAL` |
| Ratio / feedback | `RATIONAL` |
| Polynomial | `POLYNOMIAL` |
| Something new | `DISCOVER` |

### Result Access Cheat Sheet

```python
result = regressor.discover(x, y)

# Best solution (recommended)
result.best_tradeoff.expression_string  # "2.5*sin(3.0*x + 0.5)"
result.best_tradeoff.r_squared          # 0.97
result.best_tradeoff.evaluate(x_new)    # Predict on new data

# Alternative solutions
result.most_accurate                    # Highest R²
result.most_parsimonious               # Simplest
result.pareto_front                    # All non-dominated solutions

# Export
export_python(result.best_tradeoff.expression)
export_c(result.best_tradeoff.expression, use_float=True)
export_json(result)
```

---

*For more information, see the [PPF Paper](docs/PPF_Paper.md) and [API Documentation](DOCUMENTATION.md).*
