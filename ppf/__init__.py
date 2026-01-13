"""
PPF - Promising Partial Form Detection

A library for detecting mathematical forms in noisy data.

Core Capabilities:
- Form detection: Find mathematical patterns (sine, linear, quadratic, etc.)
- Residual analysis: Extract forms until residuals are noise
- Hierarchical detection: Find patterns in how form parameters evolve
- Hybrid decomposition: Combine EMD/SSA separation with PPF interpretation

Example Usage:

    # Basic form detection
    from ppf import PPFDetector
    detector = PPFDetector(min_r_squared=0.7)
    result = detector.analyze(data)

    # Residual-based analysis
    from ppf import PPFResidualLayer, EntropyMethod
    layer = PPFResidualLayer(entropy_method=EntropyMethod.SPECTRAL)
    result = layer.analyze(data)

    # Hierarchical detection
    from ppf import HierarchicalDetector, FormType
    detector = HierarchicalDetector(
        window_size=100,
        preferred_form=FormType.SINE
    )
    result = detector.analyze(data)

    # Hybrid EMD/SSA + PPF analysis
    from ppf import HybridDecomposer
    decomposer = HybridDecomposer(method="eemd")
    result = decomposer.analyze(signal)
    for comp in result.get_signal_components():
        print(f"{comp.form_type}: {comp.interpretation}")

    # Symbolic regression for discovering new forms
    from ppf import SymbolicRegressor
    regressor = SymbolicRegressor(generations=50)
    result = regressor.discover(x, y, verbose=True)
    print(f"Found: {result.best_tradeoff.expression_string}")
"""

__version__ = "0.1.0"

# Core types
from .types import (
    FormType,
    EntropyMethod,
    FitResult,
    PartialForm,
    PPFResult,
    FormLayer,
    WindowFit,
)

# Detectors
from .detector import (
    PPFDetector,
    fit_form,
    find_best_form,
    evaluate_form,
    check_residuals_are_noise,
)

# Residual layer
from .residual_layer import (
    PPFResidualLayer,
    PPFStackResult,
    measure_entropy,
    measure_entropy_gzip,
    measure_entropy_spectral,
)

# Hierarchical detection
from .hierarchical import (
    HierarchicalDetector,
    HierarchicalResult,
    HierarchyLevel,
    ParameterEvolution,
)

# Utilities
from .utils import (
    print_ppf_result,
    print_stack_result,
    print_hierarchical_result,
)

# Hybrid decomposition (EMD/SSA + PPF)
from .hybrid import (
    HybridDecomposer,
    HybridDecompositionResult,
    InterpretedComponent,
    DecompositionMethod,
    EMDDecomposer,
    SSADecomposer,
    print_hybrid_result,
)

# Symbolic regression
from .symbolic_types import (
    ExprNode,
    NodeType,
    UnaryOp,
    BinaryOp,
    MacroOp,
    PrimitiveSet,
    SymbolicFitResult,
    SymbolicRegressionResult,
    DiscoveryMode,
    TRIG_PRIMITIVES,
    GROWTH_PRIMITIVES,
    POLYNOMIAL_PRIMITIVES,
    OSCILLATOR_PRIMITIVES,
    CIRCUIT_PRIMITIVES,
    PHYSICS_PRIMITIVES,
    MINIMAL_PRIMITIVES,
    RATIONAL_PRIMITIVES,
    SATURATION_PRIMITIVES,
    UNIVERSAL_PRIMITIVES,
    ALL_MACROS_PRIMITIVES,
    MODE_TO_PRIMITIVES,
    DOMAIN_PROBE_ORDER,
)

from .symbolic import (
    SymbolicRegressor,
    GPEngine,
)

from .symbolic_utils import (
    print_symbolic_result,
    simplify_expression,
    format_expression_latex,
)

# Export layer
from .export import (
    export_python,
    export_c,
    export_json,
    load_json,
    load_json_file,
    load_json_string,
    bundle_to_json_string,
    SchemaVersionError,
    UnsupportedNodeError,
    UnsupportedOperatorError,
    SCHEMA_VERSION,
)

# Feature extraction
from .features import (
    extract_features,
    feature_vector,
    feature_matrix,
    get_schema_info,
    FAMILY_CATEGORIES,
)

__all__ = [
    # Version
    "__version__",
    # Types
    "FormType",
    "EntropyMethod",
    "FitResult",
    "PartialForm",
    "PPFResult",
    "FormLayer",
    "WindowFit",
    # Detector
    "PPFDetector",
    "fit_form",
    "find_best_form",
    "evaluate_form",
    "check_residuals_are_noise",
    # Residual Layer
    "PPFResidualLayer",
    "PPFStackResult",
    "measure_entropy",
    "measure_entropy_gzip",
    "measure_entropy_spectral",
    # Hierarchical
    "HierarchicalDetector",
    "HierarchicalResult",
    "HierarchyLevel",
    "ParameterEvolution",
    # Utils
    "print_ppf_result",
    "print_stack_result",
    "print_hierarchical_result",
    # Hybrid
    "HybridDecomposer",
    "HybridDecompositionResult",
    "InterpretedComponent",
    "DecompositionMethod",
    "EMDDecomposer",
    "SSADecomposer",
    "print_hybrid_result",
    # Symbolic regression
    "ExprNode",
    "NodeType",
    "UnaryOp",
    "BinaryOp",
    "MacroOp",
    "PrimitiveSet",
    "SymbolicFitResult",
    "SymbolicRegressionResult",
    "DiscoveryMode",
    "TRIG_PRIMITIVES",
    "GROWTH_PRIMITIVES",
    "POLYNOMIAL_PRIMITIVES",
    "OSCILLATOR_PRIMITIVES",
    "CIRCUIT_PRIMITIVES",
    "PHYSICS_PRIMITIVES",
    "MINIMAL_PRIMITIVES",
    "RATIONAL_PRIMITIVES",
    "SATURATION_PRIMITIVES",
    "UNIVERSAL_PRIMITIVES",
    "ALL_MACROS_PRIMITIVES",
    "MODE_TO_PRIMITIVES",
    "DOMAIN_PROBE_ORDER",
    "SymbolicRegressor",
    "GPEngine",
    "print_symbolic_result",
    "simplify_expression",
    "format_expression_latex",
    # Export layer
    "export_python",
    "export_c",
    "export_json",
    "load_json",
    "load_json_file",
    "load_json_string",
    "bundle_to_json_string",
    "SchemaVersionError",
    "UnsupportedNodeError",
    "UnsupportedOperatorError",
    "SCHEMA_VERSION",
    # Feature extraction
    "extract_features",
    "feature_vector",
    "feature_matrix",
    "get_schema_info",
    "FAMILY_CATEGORIES",
]
