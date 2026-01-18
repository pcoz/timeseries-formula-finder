# PPF Command-Line Interface Reference

Complete reference for the PPF command-line interface.

---

## Table of Contents

1. [Installation and Invocation](#installation-and-invocation)
2. [Global Options](#global-options)
3. [Data Input](#data-input)
4. [Commands](#commands)
   - [discover](#discover)
   - [detect](#detect)
   - [stack](#stack)
   - [hierarchy](#hierarchy)
   - [hybrid](#hybrid)
   - [export](#export)
   - [features](#features)
   - [info](#info)
5. [Workflows and Pipelines](#workflows-and-pipelines)
6. [Output Formats](#output-formats)
7. [Error Handling](#error-handling)

---

## Installation and Invocation

### Installation

```bash
pip install timeseries-formula-finder

# With hybrid decomposition support (EMD)
pip install timeseries-formula-finder[hybrid]
```

### Invocation Methods

```bash
# As installed command
ppf --help

# As Python module
python -m ppf --help
```

### Version Check

```bash
ppf --version
```

---

## Global Options

These options can be used with any command:

| Option | Short | Description |
|--------|-------|-------------|
| `--help` | `-h` | Show help message and exit |
| `--version` | | Show version number |
| `--verbose` | `-v` | Show detailed progress to stderr |
| `--quiet` | `-q` | Suppress all output except errors |
| `--json` | | Output results in JSON format |
| `--output FILE` | `-o` | Write output to FILE instead of stdout |

### Usage Notes

- `--json` goes before the subcommand: `ppf --json discover ...`
- `--verbose` output goes to stderr, results to stdout
- Use `-o` to save results while still seeing verbose progress

---

## Data Input

Most commands accept CSV data with these options:

### Positional Argument

```bash
ppf discover data.csv    # First argument is the file path
```

### Column Selection

```bash
# By column name (if headers present)
ppf discover data.csv -x time -y signal

# By column index (0-based)
ppf discover data.csv -x 0 -y 1

# Default: first column = x, second column = y
ppf discover data.csv
```

### Reading from Stdin

```bash
cat data.csv | ppf discover --stdin -x time -y signal
```

### CSV Options

| Option | Default | Description |
|--------|---------|-------------|
| `--delimiter CHAR` | `,` | Field separator character |
| `--skip-header N` | 0 | Skip N rows before parsing |
| `-x, --x-column COL` | first | X (independent) column |
| `-y, --y-column COL` | second | Y (dependent) column |

### Data Format

Expected CSV format:
```csv
time,signal
0.0,1.23
0.1,1.45
0.2,1.67
...
```

Or without headers:
```csv
0.0,1.23
0.1,1.45
0.2,1.67
```

---

## Commands

### discover

**Purpose:** Discover mathematical formulas using symbolic regression.

**Synopsis:**
```bash
ppf discover FILE [options]
ppf discover --stdin [options]
```

**Description:**

Searches the space of mathematical expressions to find formulas that fit your data. Uses multi-objective genetic programming to balance accuracy against complexity, returning a Pareto front of solutions.

The output shows three key solutions:
- **Most accurate:** Best R² regardless of complexity
- **Most parsimonious:** Simplest expression with acceptable fit
- **Best tradeoff:** Recommended balance (knee of Pareto front)

**Algorithm Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-m, --mode MODE` | auto | Discovery mode |
| `-p, --population-size N` | 500 | GP population size |
| `-g, --generations N` | 50 | Number of generations |
| `--max-depth N` | 6 | Maximum tree depth |
| `--parsimony COEF` | 0.001 | Complexity penalty |
| `--optimize-constants` | on | Optimize constants locally |
| `--no-optimize-constants` | | Disable constant optimization |
| `--random-state SEED` | None | Random seed for reproducibility |

**Output Options:**

| Option | Description |
|--------|-------------|
| `--show-pareto` | Show full Pareto front |
| `--max-pareto N` | Maximum Pareto solutions to show (default: 10) |
| `--latex` | Include LaTeX-formatted expressions |
| `--simplify` | Simplify expressions before output |

**Discovery Modes:**

| Mode | Description | Best For |
|------|-------------|----------|
| `auto` | Probes all domains, selects best | Unknown data |
| `identify` | Template matching with macros | Known forms |
| `discover` | Pure GP, no macros | Novel patterns |
| `oscillator` | Damped sinusoids | Vibrations, waves |
| `circuit` | RC charge/discharge | Electronics |
| `growth` | Sigmoid, logistic, Hill | Biology, adoption |
| `rational` | Polynomial ratios | Transfer functions |
| `polynomial` | Pure algebraic | Algebraic relationships |
| `universal` | Power laws, Gaussians | Scaling, peaks |

**Examples:**

```bash
# Basic discovery
ppf discover data.csv -x time -y signal

# Verbose with oscillator mode
ppf discover vibration.csv --mode oscillator -v -g 100

# High-quality search
ppf discover data.csv -g 200 -p 1000 --optimize-constants

# JSON output for piping
ppf --json discover data.csv | ppf export python -f model

# Reproducible results
ppf discover data.csv --random-state 42

# Show Pareto front
ppf discover data.csv --show-pareto --max-pareto 20
```

---

### detect

**Purpose:** Detect mathematical forms in data windows.

**Synopsis:**
```bash
ppf detect FILE [options]
```

**Description:**

Scans your data for windows that match known mathematical forms (constant, linear, quadratic, sine, exponential). Each detected form is validated by testing its extrapolation beyond the fitted region.

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--min-window N` | 20 | Minimum window size for detection |
| `--min-r-squared R2` | 0.7 | Minimum R² for valid fit |
| `--extrapolation-window N` | 10 | Points for extrapolation test |
| `--validation-threshold R2` | 0.6 | R² threshold for validation |

**Output:**

Shows two categories:
- **Validated forms:** Passed extrapolation test (high confidence)
- **Partial forms:** Good fit but not validated

**Examples:**

```bash
# Basic detection
ppf detect sensor.csv -x time -y reading

# Stricter threshold
ppf detect data.csv --min-r-squared 0.9

# Smaller windows for short patterns
ppf detect data.csv --min-window 10

# JSON output
ppf --json detect data.csv > forms.json
```

---

### stack

**Purpose:** Extract forms iteratively until residuals are noise.

**Synopsis:**
```bash
ppf stack FILE [options]
```

**Description:**

Implements residual-based form stacking:
1. Finds the best-fitting form in the data
2. Subtracts it, leaving residuals
3. Repeats until residuals appear noise-like (high entropy)

This "peeling" approach reveals multiple overlapping patterns.

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--entropy-method METHOD` | gzip | Entropy measurement (gzip, spectral) |
| `--noise-threshold T` | 0.85 | Entropy threshold for noise |
| `--min-compression-gain G` | 0.05 | Minimum improvement per layer |
| `--max-iterations N` | 5 | Maximum layers to extract |
| `--min-r-squared R2` | 0.5 | Minimum R² for forms |
| `--save-residuals FILE` | | Save final residuals to CSV |

**Entropy Methods:**

| Method | Description |
|--------|-------------|
| `gzip` | Compression-based (general purpose) |
| `spectral` | Spectral flatness (audio/vibration) |

**Examples:**

```bash
# Basic stacking
ppf stack sensor.csv -x time -y reading

# Use spectral entropy for audio
ppf stack audio.csv --entropy-method spectral

# More aggressive extraction
ppf stack data.csv --noise-threshold 0.95 --max-iterations 10

# Save residuals for inspection
ppf stack data.csv --save-residuals residuals.csv
```

---

### hierarchy

**Purpose:** Find nested patterns at multiple timescales.

**Synopsis:**
```bash
ppf hierarchy FILE [options]
```

**Description:**

Analyzes data hierarchically:
- **Level 0:** Fits forms to windows
- **Level 1:** Finds patterns in how Level 0 parameters evolve
- **Level 2+:** Continues if meta-patterns exist

Reveals structure like: "The frequency increases quadratically over time."

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--window-size N` | auto | Base window size |
| `--window-overlap FRAC` | 0.0 | Window overlap (0-0.9) |
| `--min-r-squared R2` | 0.3 | Minimum R² for fits |
| `--preferred-form FORM` | None | Form to search for |
| `--max-levels N` | 3 | Maximum hierarchy depth |
| `--entropy-method METHOD` | spectral | Entropy method |

**Preferred Forms:**

`constant`, `linear`, `quadratic`, `sine`, `exponential`

**Examples:**

```bash
# Basic hierarchical analysis
ppf hierarchy vibration.csv --window-size 100

# Focus on sine patterns
ppf hierarchy data.csv --preferred-form sine --window-size 50

# Overlapping windows
ppf hierarchy data.csv --window-size 100 --window-overlap 0.5

# Deeper search
ppf hierarchy data.csv --max-levels 5
```

---

### hybrid

**Purpose:** Combine signal decomposition with form interpretation.

**Synopsis:**
```bash
ppf hybrid FILE [options]
```

**Description:**

Uses signal decomposition methods to separate components, then interprets each:
- **EMD/EEMD/CEEMDAN:** Empirical Mode Decomposition variants
- **SSA:** Singular Spectrum Analysis

Each component is classified as signal or noise and given a mathematical interpretation.

**Method Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--method METHOD` | eemd | Decomposition method |
| `--noise-threshold T` | 0.5 | Entropy threshold for noise |
| `--min-r-squared R2` | 0.3 | Minimum R² for forms |
| `--min-variance V` | 0.01 | Minimum variance contribution |

**Decomposition Methods:**

| Method | Description | Dependencies |
|--------|-------------|--------------|
| `emd` | Basic EMD | EMD-signal |
| `eemd` | Ensemble EMD (recommended) | EMD-signal |
| `ceemdan` | Complete EEMD | EMD-signal |
| `ssa` | Singular Spectrum Analysis | None (built-in) |

**EMD-specific Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--max-imfs N` | auto | Maximum IMFs to extract |
| `--noise-width W` | 0.2 | Noise amplitude for EEMD/CEEMDAN |
| `--ensemble-size N` | 100 | Ensemble size |

**SSA-specific Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--ssa-window N` | N/4 | Window length |
| `--ssa-components N` | auto | Components to extract |

**Examples:**

```bash
# SSA analysis (no extra dependencies)
ppf hybrid data.csv --method ssa

# EEMD analysis
ppf hybrid sensor.csv --method eemd

# Custom SSA window
ppf hybrid data.csv --method ssa --ssa-window 50

# Strict noise classification
ppf hybrid data.csv --method eemd --noise-threshold 0.7
```

---

### export

**Purpose:** Export expressions to Python, C, or JSON.

**Synopsis:**
```bash
ppf export python [options]
ppf export c [options]
ppf export json [options]
```

**Description:**

Converts discovered expressions into deployable code. Reads JSON from stdin (piped from `ppf --json discover`).

#### export python

Generate a Python function:

| Option | Default | Description |
|--------|---------|-------------|
| `-f, --function-name NAME` | evaluate | Function name |
| `--variable VAR` | x | Variable name |
| `--safe` | off | Add error handling |
| `--no-safe` | | Skip safety checks |

```bash
ppf --json discover data.csv | ppf export python -f predict > model.py
```

**Generated code:**
```python
import numpy as np

def predict(x):
    """Evaluate the discovered expression."""
    return 2.49*np.sin(3.01*x + 0.51)
```

#### export c

Generate C code for embedded systems:

| Option | Default | Description |
|--------|---------|-------------|
| `-f, --function-name NAME` | evaluate | Function name |
| `--variable VAR` | x | Variable name |
| `--float` | off | Use float instead of double |
| `--macro-style` | off | Generate as #define macro |
| `--safe` | off | Add bounds checking |

```bash
ppf --json discover data.csv | ppf export c -f model --float > model.h
```

**Generated code:**
```c
#include <math.h>

static inline float model(float x) {
    return 2.49f * sinf(3.01f * x + 0.51f);
}
```

#### export json

Transform or annotate JSON:

| Option | Default | Description |
|--------|---------|-------------|
| `--include-tree` | off | Include full expression tree |
| `--source NAME` | | Add source identifier |
| `--variables VARS` | | Comma-separated variable names |
| `--indent N` | 2 | Indentation (0 for compact) |

```bash
ppf --json discover data.csv | ppf export json --source "sensor_v1"
```

---

### features

**Purpose:** Extract ML-ready features from discovery results.

**Synopsis:**
```bash
ppf features [options]
```

**Description:**

Extracts numerical features from discovered expressions for use in machine learning. Reads JSON from stdin.

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--schema` | | Show feature schema and exit |
| `--format FORMAT` | dict | Output format (dict, vector, csv) |
| `--include-residuals` | off | Include residual statistics |
| `--include-domain-scores` | off | Include domain probe scores |

**Example:**

```bash
ppf --json discover data.csv | ppf features
```

**Output:**
```json
{
  "r_squared": 0.9734,
  "mse": 0.068,
  "complexity": 6,
  "has_sin": 1,
  "has_exp": 0,
  "family_oscillator": 1,
  ...
}
```

**Feature Schema:**

```bash
ppf features --schema
```

---

### info

**Purpose:** Show available discovery modes, forms, and macros.

**Synopsis:**
```bash
ppf info [TOPIC]
```

**Topics:**

| Topic | Description |
|-------|-------------|
| `modes` | Discovery modes and use cases |
| `forms` | Mathematical form types |
| `macros` | Macro operators (templates) |
| `primitives` | Primitive operator sets |
| `schemas` | Feature extraction schemas |
| `methods` | Entropy measurement methods |
| `all` | Everything (default) |

**Examples:**

```bash
# Show discovery modes
ppf info modes

# Show macro templates
ppf info macros

# Show everything
ppf info all

# JSON output
ppf --json info modes
```

---

## Workflows and Pipelines

### Basic Discovery and Export

```bash
# 1. Discover
ppf discover data.csv -x time -y signal -v

# 2. Export if satisfied
ppf --json discover data.csv | ppf export python -f model > model.py
```

### Exploration Workflow

```bash
# Try different modes
for mode in oscillator circuit growth universal; do
    echo "=== Mode: $mode ==="
    ppf discover data.csv --mode $mode -g 30
done
```

### Batch Processing

```bash
# Process all CSV files
for f in data/*.csv; do
    echo "Processing $f..."
    ppf --json discover "$f" -g 50 > "${f%.csv}_model.json"
done
```

### Full Analysis Pipeline

```bash
# 1. Decompose signal
ppf hybrid data.csv --method ssa -v

# 2. Stack forms from residuals
ppf stack data.csv -v

# 3. Discover precise formula
ppf --json discover data.csv --mode auto -g 100 | \
    ppf export python -f predict > model.py

# 4. Extract features for ML
ppf --json discover data.csv | ppf features > features.json
```

### Quality Assurance Pipeline

```bash
# Discover with reproducible seed
ppf --json discover data.csv --random-state 42 > model.json

# Extract features
cat model.json | ppf features --format csv > features.csv

# Export for deployment
cat model.json | ppf export c -f sensor_model --float --safe > model.h
```

---

## Output Formats

### Human-Readable (Default)

Structured text with headers and formatted values:

```
============================================================
SYMBOLIC REGRESSION RESULTS
============================================================

Generations: 50
Evaluations: 25,000

BEST TRADEOFF (recommended):
----------------------------------------
  Expression: 2.49*sin(3.01*x + 0.51)
  R-squared:  0.9734
  ...
```

### JSON (`--json`)

Machine-readable format for piping and processing:

```json
{
  "status": "success",
  "generations_run": 50,
  "best_tradeoff": {
    "expression": "2.49*sin(3.01*x + 0.51)",
    "r_squared": 0.9734,
    "mse": 0.068,
    "complexity": 6,
    "depth": 2,
    "is_noise_like": false
  },
  ...
}
```

### Output Redirection

```bash
# To file
ppf discover data.csv -o results.txt

# With JSON
ppf --json discover data.csv > results.json

# Separate verbose and results
ppf discover data.csv -v 2>progress.log >results.txt
```

---

## Error Handling

### Common Errors

**File not found:**
```
Error: File not found: data.csv
Hint: Check file path and working directory
```

**No input:**
```
Error: No input specified. Provide a FILE or use --stdin
```

**Column not found:**
```
Error: y column 'signal' not found. Available columns: time, value, temp
```

**EMD not installed:**
```
Error: EMD methods require the EMD-signal package
Hint: Install with: pip install timeseries-formula-finder[hybrid]
```

### Debugging

```bash
# Verbose mode shows progress
ppf discover data.csv -v

# Check available options
ppf discover --help

# Check data loading
head -5 data.csv
ppf detect data.csv -v
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Argument parsing error |
| 130 | Interrupted (Ctrl+C) |

---

## Quick Reference

### Most Common Commands

```bash
# Discover formula
ppf discover data.csv -x time -y signal

# Discover with mode
ppf discover data.csv --mode oscillator -v

# Export to Python
ppf --json discover data.csv | ppf export python -f model

# Export to C
ppf --json discover data.csv | ppf export c -f model --float

# Show modes
ppf info modes
```

### Data Input Cheat Sheet

```bash
ppf CMD data.csv                    # From file
ppf CMD data.csv -x col1 -y col2    # Named columns
ppf CMD data.csv -x 0 -y 1          # Indexed columns
cat data.csv | ppf CMD --stdin      # From stdin
ppf CMD data.csv --delimiter ";"    # Custom delimiter
```

### Output Cheat Sheet

```bash
ppf CMD ...                         # Human-readable to stdout
ppf CMD ... -o file.txt             # Human-readable to file
ppf --json CMD ...                  # JSON to stdout
ppf --json CMD ... > file.json      # JSON to file
ppf CMD ... -v                      # Verbose progress to stderr
```
