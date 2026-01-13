# PPF Use Cases

This document describes practical applications of PPF with examples and implementation guidance.

---

## 1. Edge AI / IoT Sensor Deployment

### The Problem

Traditional ML approaches for IoT sensors face severe constraints:

| Constraint | TensorFlow Lite | PPF Formula |
|------------|-----------------|-------------|
| Model size | 10-100 KB | 50-100 bytes |
| Flash usage | Significant | Negligible |
| RAM at inference | ~10 KB | ~100 bytes |
| CPU cycles | 1000s MACs | ~10 FLOPs |
| Power consumption | Higher | Minimal |

For battery-powered sensors or cheap microcontrollers (ESP8266, ATtiny), neural networks often aren't feasible.

### Solution: Discover → Export → Deploy

```python
from ppf import SymbolicRegressor, export_c, DiscoveryMode
import numpy as np

# 1. Collect sensor data during development
hours = np.array([...])  # Time in hours
temperature = np.array([...])  # Temperature readings

# 2. Discover the underlying pattern
regressor = SymbolicRegressor(generations=30, population_size=300)
result = regressor.discover(hours, temperature, mode=DiscoveryMode.OSCILLATOR)

print(f"Discovered: {result.best_tradeoff.expression_string}")
# Example output: 36.2 + 5.1*cos(0.262*x - 2.1)
# This is a daily temperature cycle!

# 3. Export to C for your microcontroller
c_code = export_c(
    result.best_tradeoff.expression,
    fn_name="predict_temp",
    use_float=True,  # Use float for embedded
    safe=True        # Include safety wrappers
)

# Save to file
with open("temperature_model.h", "w") as f:
    f.write(c_code)
```

### Generated C Code

```c
#include <math.h>

static inline float clamp_exp_argf(float x) {
    if (x < -60.0f) return -60.0f;
    if (x > 60.0f) return 60.0f;
    return x;
}

static inline float predict_temp(float t) {
    return (36.2f + 5.1f * cosf(0.262f * t - 2.1f));
}
```

### Integration with Arduino/ESP32

```cpp
// In your Arduino sketch
#include "temperature_model.h"

void loop() {
    float current_hour = millis() / 3600000.0;  // Hours since boot
    float predicted_temp = predict_temp(current_hour);

    // Use prediction for anomaly detection
    float actual_temp = readSensor();
    float error = fabs(actual_temp - predicted_temp);

    if (error > THRESHOLD) {
        sendAlert("Anomaly detected!");
    }

    delay(60000);  // Check every minute
}
```

### Real-World Results

From analysis of 97,606 IoT temperature readings:

| Analysis | Form Discovered | R² | Parameters |
|----------|-----------------|-----|------------|
| Daily cycle (outdoor) | Damped sinusoid | 0.58 | 4 |
| Daily cycle (indoor) | Damped sinusoid | 0.30 | 4 |
| First 24h warming | Power law x^1.59 | 0.89 | 3 |

**Key insight**: Real sensor data can be compressed to 4-8 parameters instead of thousands of neural network weights.

---

## 2. Biomedical Signal Analysis (ECG)

### The Problem

ECG analysis traditionally uses either:
1. **Rule-based algorithms**: Hand-crafted thresholds, miss subtle patterns
2. **Deep learning**: Black-box, hard to validate clinically

PPF offers a middle ground: data-driven discovery with interpretable outputs.

### Solution: Multi-Perspective Analysis

```python
from ppf import SymbolicRegressor, DiscoveryMode, extract_features
import numpy as np

# Load ECG data (e.g., from PhysioNet)
t_wave_time = np.array([...])  # Time axis for T-wave segment
t_wave_amplitude = np.array([...])  # T-wave voltage

# Analyze from multiple "perspectives"
perspectives = {
    "universal": DiscoveryMode.UNIVERSAL,    # Gaussians, power laws
    "oscillator": DiscoveryMode.OSCILLATOR,  # Damped oscillations
    "polynomial": DiscoveryMode.POLYNOMIAL,  # Algebraic
}

results = {}
for name, mode in perspectives.items():
    regressor = SymbolicRegressor(generations=30)
    result = regressor.discover(t_wave_time, t_wave_amplitude, mode=mode)
    results[name] = result
    print(f"{name}: R²={result.best_tradeoff.r_squared:.4f}")
    print(f"  Form: {result.best_tradeoff.expression_string}")
```

### Example Output

```
universal: R²=0.9234
  Form: 0.85*exp(-((x - 5.2)/1.8)^2)  # Gaussian

oscillator: R²=0.9612
  Form: 1.02*exp(-0.45*x)*cos(0.89*x + 0.2)  # Damped cosine

polynomial: R²=0.8901
  Form: -0.03*x^2 + 0.31*x + 0.12  # Quadratic
```

**Key insight**: The T-wave is better described by a damped cosine (R²=0.96) than a Gaussian (R²=0.92), even though textbooks often assume Gaussian. This has physiological implications - it suggests the repolarization process has oscillatory dynamics.

### Feature Extraction for Cardiac Classification

```python
# Extract interpretable features from each heartbeat
all_features = []
for beat in heartbeats:
    result = regressor.discover(beat.t, beat.amplitude)
    features = extract_features(result)
    all_features.append(features)

# Convert to feature matrix for sklearn
from ppf import feature_matrix
X, feature_names = feature_matrix(all_features)

# Train classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, labels)  # labels: normal, afib, pvc, etc.

# Interpretable feature importance
for name, importance in zip(feature_names, clf.feature_importances_):
    if importance > 0.05:
        print(f"{name}: {importance:.3f}")
```

### Clinical Interpretation

| Feature | Meaning | Clinical Relevance |
|---------|---------|-------------------|
| `damping_k` | Decay rate | Repolarization dynamics |
| `omega` | Oscillation frequency | T-wave duration |
| `amplitude` | Peak height | T-wave amplitude |
| `dominant_family` | Form type | Waveform morphology |

---

## 3. Predictive Maintenance (Vibration Analysis)

### The Problem

Industrial machinery health monitoring typically uses:
1. **Threshold alerts**: Miss early degradation
2. **FFT spectrum analysis**: Requires expert interpretation
3. **Deep learning**: Needs large labeled datasets

PPF discovers the mathematical FORM of vibration signatures, which directly indicates fault type.

### Solution: Form-Based Fault Classification

```python
from ppf import SymbolicRegressor, DiscoveryMode, extract_features

# Collect vibration data from accelerometer
# (e.g., 1 second at 2000 Hz = 2000 samples)
t = np.linspace(0, 1, 2000)
vibration = accelerometer.read()

# Discover the form
regressor = SymbolicRegressor(generations=25)
result = regressor.discover(t, vibration, mode=DiscoveryMode.OSCILLATOR)

expr_str = result.best_tradeoff.expression_string.lower()

# Form-based classification
if "exp(-" in expr_str and ("sin" in expr_str or "cos" in expr_str):
    diagnosis = "BEARING_FAULT"  # Damped oscillations = impact response
elif expr_str.count("sin") > 2:
    diagnosis = "LOOSENESS"  # Many harmonics
elif "sin(2" in expr_str or "cos(2" in expr_str:
    diagnosis = "MISALIGNMENT"  # Strong 2x harmonic
else:
    diagnosis = "NORMAL"

print(f"Diagnosis: {diagnosis}")
print(f"Form: {result.best_tradeoff.expression_string}")
```

### Fault Signature Reference

| Fault Type | Mathematical Form | Physical Cause |
|------------|-------------------|----------------|
| Normal | `A·sin(ωt)` | Clean rotation |
| Imbalance | `A·sin(ωt)` with large A | Mass imbalance |
| Misalignment | `A·sin(ωt) + B·sin(2ωt)` | Shaft misalignment |
| Bearing (outer) | `A·exp(-kt)·sin(ω_r·t)` | Ball impacts on outer race |
| Bearing (inner) | `A·exp(-kt)·sin(ω_r·t)` | Ball impacts on inner race |
| Looseness | `Σ A_n·sin(nωt)` | Mechanical looseness |

**Key insight**: The FORM of the discovered expression IS the diagnosis. No separate classification model needed.

### Edge Deployment for Real-Time Monitoring

```python
# Export to C for embedded monitoring
c_code = export_c(
    reference_healthy_expr,
    fn_name="expected_vibration",
    use_float=True
)

# On the embedded device:
# 1. Evaluate expected_vibration(t)
# 2. Compare with actual sensor reading
# 3. If |actual - expected| > threshold → alert
```

---

## 4. Scientific Data Analysis

### The Problem

Scientific data often follows known physical laws, but with unknown parameters. Traditional approaches:
1. **Curve fitting**: Must guess the form first
2. **Neural networks**: Don't reveal the underlying physics

### Solution: Law Discovery

```python
from ppf import SymbolicRegressor, DiscoveryMode

# Example: Planetary orbit data
# Kepler's 3rd law: T² ∝ a³ (period vs semi-major axis)
orbital_periods = np.array([...])  # Years
semi_major_axes = np.array([...])  # AU

regressor = SymbolicRegressor(generations=50)
result = regressor.discover(
    semi_major_axes,
    orbital_periods**2,  # Hint: analyze T² vs a
    mode=DiscoveryMode.POLYNOMIAL
)

print(result.best_tradeoff.expression_string)
# Output: 1.0*x^1.5  → T² = a^1.5 → T = a^0.75 ≈ a^(3/2)
# Rediscovered Kepler's 3rd law!
```

### Advantages for Science

1. **Hypothesis generation**: PPF can suggest functional forms
2. **Parameter estimation**: Get best-fit coefficients
3. **Model validation**: Compare R² of competing theories
4. **Anomaly detection**: Data that doesn't fit known forms

---

## 5. Feature Engineering for ML Pipelines

### The Problem

Traditional feature engineering is manual and domain-specific. Neural networks learn features automatically but they're not interpretable.

### Solution: PPF as Feature Extractor

```python
from ppf import SymbolicRegressor, extract_features, feature_vector
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

# Example: Classify time series by their underlying form
X_train_raw = [...]  # List of time series
y_train = [...]      # Labels

# Extract PPF features from each time series
X_train_features = []
for ts in X_train_raw:
    t = np.arange(len(ts))
    result = regressor.discover(t, ts, mode=DiscoveryMode.AUTO)
    features = extract_features(result)
    vec, _ = feature_vector(features)
    X_train_features.append(vec)

X_train = np.array(X_train_features)

# Train classifier on PPF features
clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)

# Features are interpretable!
# - dominant_family: "oscillation", "growth", "ratio", etc.
# - damping_k, omega, amplitude: physical parameters
# - r2, complexity: fit quality
```

### Comparison with Raw Features

| Feature Type | Interpretability | Generalization | Dimension |
|--------------|------------------|----------------|-----------|
| Raw time series | None | Poor | High |
| FFT coefficients | Some | Moderate | Medium |
| Statistical moments | Limited | Good | Low |
| **PPF features** | **High** | **Good** | **Low** |

---

## 6. Anomaly Detection

### The Problem

Anomaly detection typically requires training on "normal" data. But what is "normal"?

### Solution: Model-Based Anomaly Detection

```python
from ppf import SymbolicRegressor, export_python

# 1. Learn the "normal" pattern from historical data
regressor = SymbolicRegressor()
result = regressor.discover(historical_t, historical_y)

# 2. Export the model
code = export_python(result.best_tradeoff.expression, fn_name="expected")
exec(code)

# 3. Real-time anomaly detection
def detect_anomaly(t, y_actual, threshold=3.0):
    y_expected = expected(t)
    residual = abs(y_actual - y_expected)

    if residual > threshold * historical_std:
        return True, residual
    return False, residual

# No neural network, no training pipeline, just math!
```

### Advantages

1. **No anomaly examples needed**: Learn from normal data only
2. **Interpretable thresholds**: Residual has physical meaning
3. **Lightweight**: Formula evaluation, not neural inference
4. **Drift detection**: Model drift = parameter drift

---

## Summary: When to Use PPF

| Use Case | PPF Advantage | Example |
|----------|---------------|---------|
| **Edge AI** | 50 bytes vs 50KB | IoT temperature prediction |
| **Biomedical** | Interpretable features | ECG morphology classification |
| **Maintenance** | Form = diagnosis | Vibration fault detection |
| **Science** | Law discovery | Kepler's law from orbital data |
| **Feature eng.** | Physics-informed features | Time series classification |
| **Anomaly det.** | Model-based threshold | Process monitoring |

**Key principle**: If your data has underlying mathematical structure, PPF can find it, export it, and make it useful for downstream tasks.
