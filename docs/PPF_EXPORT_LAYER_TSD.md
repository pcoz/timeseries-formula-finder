# PPF Export Layer - Technical Specification Document

**Version**: 1.0
**Status**: Draft
**Date**: January 2026

---

## 1. Overview

### 1.1 Goal

Enable "discover → deploy" workflows by exporting discovered expressions/models into:

1. **Pure-Python evaluators** (no PPF dependency at runtime)
2. **C evaluators** (for MCU/embedded/edge and high-performance systems)
3. **JSON model bundles** (for MQTT, storage, auditing, remote deployment, interoperability)
4. **Feature vectors** derived from fitted models (for ML classifiers/regressors)

### 1.2 Non-Goals

- Not a full compiler or optimizer
- Not responsible for training downstream ML models
- Not responsible for device OTA update mechanisms (only output artifacts)

### 1.3 Design Principles

| Principle | Rationale |
|-----------|-----------|
| **No lock-in** | Exported artifacts must not depend on PPF |
| **Portable** | Outputs run anywhere (Python 3.8+, C99, JSON) |
| **Safe by default** | Prevent NaN/Inf from division, log, exp edge cases |
| **Deterministic** | Same expression → identical output string |
| **Round-trippable** | JSON export → load reconstructs identical tree |
| **Versioned** | Schemas are versioned for forward compatibility |

---

## 2. Public API

### Module: `ppf.export`

---

### 2.1 `export_python()`

```python
def export_python(
    expr: ExprNode,
    *,
    fn_name: str = "ppf_model",
    signature: tuple[str, ...] = ("t",),
    safe: bool = True
) -> str
```

**Purpose**: Generate a standalone Python function (string) that evaluates the expression.

#### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `expr` | `ExprNode` | Expression tree object |
| `fn_name` | `str` | Function name in output |
| `signature` | `tuple[str]` | Variable names, e.g., `("t",)` or `("x", "y")` |
| `safe` | `bool` | Emit safe wrappers for div/log/exp |

#### Output

A Python code string that:
- Imports `math`
- Defines safe helpers if `safe=True`
- Defines `def <fn_name>(<signature>): return <expression>`

#### Runtime Constraints

- Must **not** depend on `numpy`
- Must **not** depend on PPF
- Must run on CPython 3.8+

#### Safety Rules

If `safe=True`, emit:

```python
def safe_div(a, b, eps=1e-12):
    return a / (b if abs(b) > eps else (eps if b >= 0 else -eps))

def safe_log(x, eps=1e-12):
    return math.log(max(x, eps))

def clamp_exp(x, lo=-60, hi=60):
    return math.exp(min(max(x, lo), hi))
```

#### Determinism

- Output must be stable: same `expr` → identical emitted string (whitespace normalized)

#### Example Output

```python
import math

def safe_div(a, b, eps=1e-12):
    return a / (b if abs(b) > eps else (eps if b >= 0 else -eps))

def safe_log(x, eps=1e-12):
    return math.log(max(x, eps))

def clamp_exp(x, lo=-60, hi=60):
    return math.exp(min(max(x, lo), hi))

def ppf_model(t):
    return 3.004 * clamp_exp(-0.5015 * t) * math.sin(6.008 * t + -1.579)
```

---

### 2.2 `export_c()`

```python
def export_c(
    expr: ExprNode,
    *,
    fn_name: str = "ppf_model",
    signature: tuple[str, ...] = ("double t",),
    safe: bool = True,
    use_float: bool = False,
    macro_style: str = "inline"  # "inline" | "helpers"
) -> str
```

**Purpose**: Generate a standalone C99 function that evaluates the expression.

#### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `expr` | `ExprNode` | Expression tree |
| `fn_name` | `str` | Output function name |
| `signature` | `tuple[str]` | Argument declarations, e.g., `("double t",)` |
| `safe` | `bool` | Emit safe wrappers |
| `use_float` | `bool` | Use `float` math (`sinf`, `expf`) instead of `double` |
| `macro_style` | `str` | `"inline"` expands macros; `"helpers"` emits helper functions |

#### Output

A C code string containing:
- `#include <math.h>`
- Helper functions if `safe=True`
- The model function

#### Portability Requirements

- Must compile as **C99**
- **No dynamic allocation**
- **No external dependencies** beyond `math.h`

#### Safety Rules

If `safe=True`, emit:

```c
static inline double safe_div(double a, double b) {
    const double eps = 1e-12;
    if (fabs(b) > eps) return a / b;
    return a / (b >= 0 ? eps : -eps);
}

static inline double safe_log(double x) {
    const double eps = 1e-12;
    return log(x > eps ? x : eps);
}

static inline double clamp_exp_arg(double x) {
    if (x < -60.0) return -60.0;
    if (x > 60.0) return 60.0;
    return x;
}
```

#### Macro Expansion

**Default (`macro_style="inline"`)**: Expand macros into primitive ops.

`DAMPED_SIN(a, k, t, w, phi)` becomes:
```c
a * exp(clamp_exp_arg(-k * t)) * sin(w * t + phi)
```

**With `macro_style="helpers"`**: Emit helper functions:

```c
static inline double damped_sin(double a, double k, double t, double w, double phi) {
    return a * exp(clamp_exp_arg(-k * t)) * sin(w * t + phi);
}
```

#### Float Variant

If `use_float=True`:
- Use `float` return type and parameters
- Use `sinf`, `cosf`, `expf`, `logf`, `powf`, `fabsf`

#### Notes

- If expression uses `pow`, emit `pow()` / `powf()`
- Emit integer constants with `.0` suffix where needed (e.g., `2.0` not `2`)

#### Example Output

```c
#include <math.h>

static inline double safe_div(double a, double b) {
    const double eps = 1e-12;
    if (fabs(b) > eps) return a / b;
    return a / (b >= 0 ? eps : -eps);
}

static inline double clamp_exp_arg(double x) {
    if (x < -60.0) return -60.0;
    if (x > 60.0) return 60.0;
    return x;
}

static inline double ppf_model(double t) {
    return 3.004 * exp(clamp_exp_arg(-0.5015 * t)) * sin(6.008 * t + -1.579);
}
```

---

### 2.3 `export_json()`

```python
def export_json(
    model: SymbolicFitResult | SymbolicRegressionResult,
    *,
    include_expr_tree: bool = True,
    include_source: str = "ppf",
    version: str | None = None
) -> dict
```

**Purpose**: Create a portable model bundle for storage, MQTT, REST, etc.

#### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `SymbolicFitResult` | Result object from discovery |
| `include_expr_tree` | `bool` | Include full tree serialization |
| `include_source` | `str` | Metadata source tag |
| `version` | `str` | Version string (defaults to PPF version) |

#### Output JSON Schema

```json
{
  "schema": "ppf.export.model.v1",
  "source": "ppf",
  "ppf_version": "0.1.0",
  "created_utc": "2026-01-13T10:00:00Z",

  "metadata": {
    "seed": 42,
    "mode": "OSCILLATOR"
  },

  "metrics": {
    "r2": 0.98,
    "rmse": 0.12,
    "complexity": 5,
    "parsimony": 0.01
  },

  "variables": ["t"],

  "expression": {
    "string": "3.004*exp(-0.5015*t)*sin(6.008*t + -1.579)",
    "latex": "3.004 e^{-0.5015 t} \\sin(6.008 t - 1.579)",
    "tree": { ... }
  },

  "parameters": {
    "a": 3.004,
    "k": 0.5015,
    "w": 6.008,
    "phi": -1.579
  },

  "constraints": {
    "safe": true,
    "div_epsilon": 1e-12,
    "exp_clip": [-60, 60],
    "log_epsilon": 1e-12
  }
}
```

#### Tree Serialization

Recursive node objects with the following types:

| Node Type | Structure |
|-----------|-----------|
| `CONST` | `{"type": "CONST", "value": 3.004}` |
| `VAR` | `{"type": "VAR", "name": "t"}` |
| `UNARY_OP` | `{"type": "UNARY_OP", "op": "EXP", "child": {...}}` |
| `BINARY_OP` | `{"type": "BINARY_OP", "op": "MUL", "children": [{...}, {...}]}` |
| `MACRO_CALL` | `{"type": "MACRO_CALL", "name": "DAMPED_SIN", "args": [...]}` |

#### Macro Serialization

Macros serialize as `MACRO_CALL` nodes with a `name` and `args` array:

```json
{
  "type": "MACRO_CALL",
  "name": "DAMPED_SIN",
  "args": [
    {"type": "CONST", "value": 3.004},
    {"type": "CONST", "value": 0.5015},
    {"type": "VAR", "name": "t"},
    {"type": "CONST", "value": 6.008},
    {"type": "CONST", "value": -1.579}
  ]
}
```

**Contract**: Macros must be round-trippable without expansion. Expansion is optional at export time.

#### Macro Argument Order

Argument order is fixed per macro and must be documented:

| Macro | Arg Order |
|-------|-----------|
| `DAMPED_SIN` | `(a, k, t, w, phi)` |
| `DAMPED_COS` | `(a, k, t, w, phi)` |
| `RC_CHARGE` | `(a, k, t, c)` |
| `EXP_DECAY` | `(a, k, t, c)` |
| `SIGMOID` | `(a, k, x, x0)` |
| `LOGISTIC` | `(a, b, k, x)` |
| `HILL` | `(a, k, n, x)` |
| `RATIO` | `(a, b, c, d, x)` |
| `RATIONAL2` | `(a, b, c, d, e, f, x)` |
| `POWER_LAW` | `(a, b, x, c)` |
| `GAUSSIAN` | `(a, mu, sigma, x)` |
| `TANH_STEP` | `(a, k, x0, x, c)` |

#### Round-Trip Requirement

A model bundle must be loadable back into an `ExprNode` with **no loss of structure**.

---

### 2.4 `load_json()`

```python
def load_json(bundle: dict) -> tuple[ExprNode, dict]
```

**Purpose**: Reconstruct expression + metadata from JSON bundle.

#### Returns

- `expr`: Reconstructed `ExprNode`
- `metadata`: Dict containing metrics, parameters, constraints, etc.

#### Requirements

- Must validate schema version
- Must raise `SchemaVersionError` on unsupported schema
- Must raise `UnsupportedNodeError` on unknown node/op types
- May optionally accept bundles with only `expression.string` (no tree) if a parser is implemented

---

## 3. Feature Extraction API

### Module: `ppf.features`

---

### 3.1 `extract_features()`

```python
def extract_features(
    result: SymbolicRegressionResult | SymbolicFitResult,
    *,
    include_residual_stats: bool = True,
    include_domain_scores: bool = True
) -> dict
```

**Purpose**: Convert a discovery result into a stable, interpretable feature dictionary for downstream ML.

#### Output Keys

**Common keys** (always present):

| Key | Type | Description |
|-----|------|-------------|
| `mode_chosen` | `str` | `"OSCILLATOR"`, `"RATIONAL"`, `"UNIVERSAL"`, etc. |
| `r2` | `float` | R-squared metric |
| `rmse` | `float` | Root mean squared error |
| `complexity` | `int` | Expression tree node count |
| `dominant_family` | `str` | `"oscillation"`, `"growth"`, `"saturation"`, `"ratio"`, `"peaks"` |

**Residual stats** (if `include_residual_stats=True`):

| Key | Type | Description |
|-----|------|-------------|
| `residual_rms` | `float` | RMS of residuals |
| `residual_mad` | `float` | Median absolute deviation |
| `residual_max_abs` | `float` | Maximum absolute residual |

**Domain-specific parameters** (extracted from expression if present):

**Oscillator family**:
| Key | Description |
|-----|-------------|
| `freq_hz` | Frequency in Hz (if time axis known) |
| `omega` | Angular frequency |
| `damping_k` | Decay rate |
| `amplitude` | Signal amplitude |
| `phase` | Phase offset |

**Logistic/Sigmoid family**:
| Key | Description |
|-----|-------------|
| `K` | Carrying capacity / upper asymptote |
| `r` | Growth rate |
| `t0` | Midpoint |

**Rational family**:
| Key | Description |
|-----|-------------|
| `numerator_degree` | Degree of numerator polynomial |
| `denominator_degree` | Degree of denominator polynomial |
| `gain_estimate` | Optional DC gain heuristic |

**Gaussian/Peak family**:
| Key | Description |
|-----|-------------|
| `amplitude` | Peak height |
| `mu` | Peak center |
| `sigma` | Peak width |

#### Stability Requirement

**Scope**: Stable per *result object*, not across independent runs.

GP is stochastic, so you cannot guarantee the same discovered expression across runs. What you **can** guarantee:

Given a specific `result.best_tradeoff.expression` (a fixed expression):
- Same keys every time
- Same ordering (once vectorized)
- Deterministic values (within float tolerance)

**Missing keys**: Must be absent (not `null`). Document this convention.

---

### 3.2 `feature_vector()`

```python
def feature_vector(
    features: dict,
    schema: str = "ppf.features.v1"
) -> tuple[np.ndarray, list[str]]
```

**Purpose**: Convert feature dict to a fixed-order vector for ML.

#### Returns

- `array`: NumPy array of feature values
- `names`: List of feature names in corresponding order

#### Schemas

| Schema | Fields | Use Case |
|--------|--------|----------|
| `ppf.features.v1.edge_min` | 12-20 | Minimal features for edge/TinyML |
| `ppf.features.v1.full` | 30-60 | Rich features for full ML pipelines |

#### Requirements

- Must define and freeze ordering per schema version
- Must return both array and feature names
- Missing features filled with `NaN`

---

## 4. Multi-Variable Support

### Clarification

Export supports expressions over N variables if the expression tree contains `VAR(name)` nodes matching the requested signature.

Discovery is currently **1D** (single time-series variable) unless explicitly enabled by future multi-variable modes.

**Spec statement**:

> "Export supports expressions over N variables if the expression tree contains `VAR(name)` nodes matching the requested signature. Discovery is currently 1D unless explicitly enabled by future multi-var modes."

---

## 5. CLI Interface

### Command: `ppf export`

*(Optional but recommended)*

#### Usage

```bash
# Export from saved model bundle
ppf export model.json --format c --out model.c
ppf export model.json --format python --out model.py

# Discover and export in one step
ppf export data.csv --x t --y y --mode auto --format json --out model.json

# Multiple formats
ppf export model.json --format c,python,json --out-dir ./exports/
```

#### Options

| Option | Description |
|--------|-------------|
| `--format` | Output format: `python`, `c`, `json` |
| `--out` | Output file path |
| `--out-dir` | Output directory (for multiple formats) |
| `--safe / --no-safe` | Enable/disable safety wrappers |
| `--float` | Use float instead of double (C only) |
| `--macro-style` | `inline` or `helpers` (C only) |

---

## 6. Implementation Requirements

### 6.1 Operator Coverage

Exporter must support all operators used by `PrimitiveSet`:

**Binary operators**: `+ - * / pow`

**Unary operators**: `sin cos exp log abs neg sqrt square`

**Macros**: All defined macros (DAMPED_SIN, GAUSSIAN, etc.)

**Error handling**: If an operator is unsupported:
```python
raise UnsupportedOperatorError(op_name, node_repr)
```

### 6.2 Numeric Formatting

- Emit constants with controlled precision (default 12 significant figures)
- Preserve sign explicitly (avoid `+-` artifacts)
- Ensure deterministic string output (sort parameters, stable traversal order)

### 6.3 Safety Defaults

- Default exports should be `safe=True`
- Allow `safe=False` for scientific "pure math" contexts

---

## 7. Testing Plan

### 7.1 Unit Tests

| Test | Description |
|------|-------------|
| **Determinism** | Same expr emits identical code (Python and C) |
| **Correctness** | Emitted Python function matches `expr.evaluate()` within tolerance |
| **Safety behavior** | Division/log/exp edge cases do not crash |
| **Round-trip JSON** | `export_json → load_json` reconstructs identical tree |
| **Macro coverage** | Each macro primitive exports correctly |

### 7.2 Golden Tests

Use existing benchmark canonical forms:

| Benchmark | Test |
|-----------|------|
| Damped oscillator | Exported Python/C produce same R² on test points |
| RC charge | Same |
| Keijzer rational | Same |
| Logistic sigmoid | Same |
| Gaussian | Same |

### 7.3 Integration Test: Edge Simulation

1. Export C code
2. Compile with `gcc -std=c99 -lm`
3. Run evaluation on sample input set
4. Compare outputs with Python reference

---

## 8. Deliverables

### 8.1 Module Structure

```
ppf/
├── export/
│   ├── __init__.py
│   ├── python_export.py    # export_python()
│   ├── c_export.py         # export_c()
│   ├── json_export.py      # export_json()
│   └── load.py             # load_json()
├── features/
│   ├── __init__.py
│   ├── extract.py          # extract_features()
│   └── vectorize.py        # feature_vector()
```

### 8.2 Test Files

```
tests/
├── test_export_python.py
├── test_export_c.py
├── test_export_json.py
├── test_export_roundtrip.py
├── test_features.py
└── golden/
    ├── damped_oscillator.py
    ├── damped_oscillator.c
    └── damped_oscillator.json
```

### 8.3 Example Scripts

```
examples/
├── ecg_export_demo.py      # ECG → C for wearable
├── iot_export_demo.py      # IoT sensor → JSON bundle
└── ml_features_demo.py     # Feature extraction → sklearn
```

### 8.4 Documentation

- README section: "Deploy to Edge"
- API reference for all export functions
- JSON schema documentation

---

## 9. Packaging Readiness Criteria

Before PyPI release:

- [ ] Export modules stable and documented
- [ ] Feature schema versioned (`ppf.features.v1`)
- [ ] JSON schema versioned (`ppf.export.model.v1`)
- [ ] All unit tests passing
- [ ] Golden tests passing
- [ ] Example scripts working
- [ ] README updated with export documentation

---

## Appendix A: JSON Schema Reference

### A.1 Full Schema: `ppf.export.model.v1`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "PPF Model Bundle",
  "type": "object",
  "required": ["schema", "expression", "metrics"],
  "properties": {
    "schema": {"type": "string", "const": "ppf.export.model.v1"},
    "source": {"type": "string"},
    "ppf_version": {"type": "string"},
    "created_utc": {"type": "string", "format": "date-time"},
    "metadata": {
      "type": "object",
      "properties": {
        "seed": {"type": "integer"},
        "mode": {"type": "string"}
      }
    },
    "metrics": {
      "type": "object",
      "required": ["r2", "complexity"],
      "properties": {
        "r2": {"type": "number"},
        "rmse": {"type": "number"},
        "complexity": {"type": "integer"},
        "parsimony": {"type": "number"}
      }
    },
    "variables": {
      "type": "array",
      "items": {"type": "string"}
    },
    "expression": {
      "type": "object",
      "required": ["string"],
      "properties": {
        "string": {"type": "string"},
        "latex": {"type": "string"},
        "tree": {"type": "object"}
      }
    },
    "parameters": {
      "type": "object",
      "additionalProperties": {"type": "number"}
    },
    "constraints": {
      "type": "object",
      "properties": {
        "safe": {"type": "boolean"},
        "div_epsilon": {"type": "number"},
        "exp_clip": {"type": "array", "items": {"type": "number"}},
        "log_epsilon": {"type": "number"}
      }
    }
  }
}
```

---

## Appendix B: Macro Argument Reference

| Macro | Formula | Args (in order) |
|-------|---------|-----------------|
| `DAMPED_SIN` | `a * exp(-k*t) * sin(w*t + phi)` | `a, k, t, w, phi` |
| `DAMPED_COS` | `a * exp(-k*t) * cos(w*t + phi)` | `a, k, t, w, phi` |
| `RC_CHARGE` | `a * (1 - exp(-k*t)) + c` | `a, k, t, c` |
| `EXP_DECAY` | `a * exp(-k*t) + c` | `a, k, t, c` |
| `SIGMOID` | `a / (1 + exp(-k*(x - x0)))` | `a, k, x, x0` |
| `LOGISTIC` | `a / (1 + b*exp(-k*x))` | `a, b, k, x` |
| `HILL` | `a * x^n / (k^n + x^n)` | `a, k, n, x` |
| `RATIO` | `(a*x + b) / (c*x + d)` | `a, b, c, d, x` |
| `RATIONAL2` | `(a*x² + b*x + c) / (d*x² + e*x + f)` | `a, b, c, d, e, f, x` |
| `POWER_LAW` | `a * x^b + c` | `a, b, x, c` |
| `GAUSSIAN` | `a * exp(-((x-mu)/sigma)²)` | `a, mu, sigma, x` |
| `TANH_STEP` | `a * tanh(k*(x-x0)) + c` | `a, k, x0, x, c` |

---

*End of Technical Specification Document*
