"""
Symbolic regression types for PPF library.

This module defines expression trees and related types for
genetic programming-based symbolic regression.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Callable, Union
import numpy as np


# Constants for protected operations
EPSILON = 1e-10


class NodeType(Enum):
    """Types of nodes in expression trees"""
    CONSTANT = auto()     # Numeric constant (optimizable)
    VARIABLE = auto()     # Input variable (x)
    UNARY_OP = auto()     # Single-argument operator
    BINARY_OP = auto()    # Two-argument operator
    MACRO = auto()        # Multi-parameter template function


class UnaryOp(Enum):
    """Unary operators in the primitive set"""
    SIN = "sin"
    COS = "cos"
    EXP = "exp"
    LOG = "log"          # Protected: log(|x| + epsilon)
    SQRT = "sqrt"        # Protected: sqrt(|x|)
    NEG = "neg"          # Unary negation
    ABS = "abs"
    SQUARE = "square"    # x^2


class BinaryOp(Enum):
    """Binary operators in the primitive set"""
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"          # Protected division
    POW = "pow"          # Protected power


class MacroOp(Enum):
    """
    Macro operators - template functions with multiple learnable constants.

    These capture common physical patterns that are hard for GP to evolve
    from basic primitives (multiplicative compositions across function families).
    """
    # Damped oscillations: a * exp(-k*t) * sin(w*t + phi)
    DAMPED_SIN = "damped_sin"  # 4 params: amplitude, decay, frequency, phase
    DAMPED_COS = "damped_cos"  # 4 params: amplitude, decay, frequency, phase

    # RC circuit charging: a * (1 - exp(-k*t)) + c
    RC_CHARGE = "rc_charge"    # 3 params: amplitude, rate, offset

    # Exponential decay: a * exp(-k*t) + c
    EXP_DECAY = "exp_decay"    # 3 params: amplitude, rate, offset

    # Rational functions: polynomials divided by polynomials
    RATIO = "ratio"            # 4 params: (ax + b) / (cx + d)
    RATIONAL2 = "rational2"    # 6 params: (ax^2 + bx + c) / (dx^2 + ex + f)

    # Sigmoid/saturation functions
    SIGMOID = "sigmoid"        # 3 params: a / (1 + exp(-k*(x - x0)))
    LOGISTIC = "logistic"      # 3 params: a / (1 + b*exp(-k*x))
    HILL = "hill"              # 3 params: a * x^n / (k^n + x^n)

    # Universal forms (common across many domains)
    POWER_LAW = "power_law"    # 3 params: a * x^b + c
    GAUSSIAN = "gaussian"      # 3 params: a * exp(-((x-mu)/sigma)^2)
    TANH_STEP = "tanh_step"    # 4 params: a * tanh(k*(x - x0)) + c


# Number of constants for each macro
MACRO_PARAM_COUNT = {
    MacroOp.DAMPED_SIN: 4,   # a, k, w, phi
    MacroOp.DAMPED_COS: 4,   # a, k, w, phi
    MacroOp.RC_CHARGE: 3,    # a, k, c
    MacroOp.EXP_DECAY: 3,    # a, k, c
    MacroOp.RATIO: 4,        # a, b, c, d
    MacroOp.RATIONAL2: 6,    # a, b, c, d, e, f
    MacroOp.SIGMOID: 3,      # a, k, x0
    MacroOp.LOGISTIC: 3,     # a, b, k
    MacroOp.HILL: 3,         # a, k, n
    MacroOp.POWER_LAW: 3,    # a, b, c
    MacroOp.GAUSSIAN: 3,     # a, mu, sigma
    MacroOp.TANH_STEP: 4,    # a, k, x0, c
}

# Default initial values for macro parameters
MACRO_DEFAULTS = {
    MacroOp.DAMPED_SIN: [1.0, 0.5, 3.0, 0.0],   # a=1, k=0.5, w=3, phi=0
    MacroOp.DAMPED_COS: [1.0, 0.5, 3.0, 0.0],   # a=1, k=0.5, w=3, phi=0
    MacroOp.RC_CHARGE: [1.0, 1.0, 0.0],          # a=1, k=1, c=0
    MacroOp.EXP_DECAY: [1.0, 1.0, 0.0],          # a=1, k=1, c=0
    MacroOp.RATIO: [1.0, 0.0, 1.0, 1.0],         # (x + 0) / (x + 1) = x/(x+1)
    MacroOp.RATIONAL2: [1.0, 0.0, 0.0, 1.0, 0.0, 1.0],  # x^2 / (x^2 + 1)
    MacroOp.SIGMOID: [1.0, 1.0, 0.0],            # 1 / (1 + exp(-x))
    MacroOp.LOGISTIC: [1.0, 1.0, 1.0],           # 1 / (1 + exp(-x))
    MacroOp.HILL: [1.0, 1.0, 1.0],               # x / (1 + x)
    MacroOp.POWER_LAW: [1.0, 1.0, 0.0],          # a=1, b=1 (linear), c=0
    MacroOp.GAUSSIAN: [1.0, 0.0, 1.0],           # a=1, mu=0, sigma=1
    MacroOp.TANH_STEP: [1.0, 1.0, 0.0, 0.0],     # a=1, k=1, x0=0, c=0
}


# Protected operation implementations
def _protected_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Division protected against zero"""
    sign_b = np.sign(b)
    sign_b = np.where(sign_b == 0, 1, sign_b)
    return a / (b + EPSILON * sign_b)


def _protected_log(x: np.ndarray) -> np.ndarray:
    """Log protected against non-positive"""
    return np.log(np.abs(x) + EPSILON)


def _protected_sqrt(x: np.ndarray) -> np.ndarray:
    """Sqrt protected against negative"""
    return np.sqrt(np.abs(x))


def _protected_pow(base: np.ndarray, exp: np.ndarray) -> np.ndarray:
    """Power protected against complex results"""
    abs_base = np.abs(base) + EPSILON
    result = np.power(abs_base, exp)
    # Clip to prevent overflow
    return np.clip(result, -1e100, 1e100)


def _protected_exp(x: np.ndarray) -> np.ndarray:
    """Exp protected against overflow"""
    return np.exp(np.clip(x, -100, 100))


# Operator function lookup tables
UNARY_FUNCS = {
    UnaryOp.SIN: np.sin,
    UnaryOp.COS: np.cos,
    UnaryOp.EXP: _protected_exp,
    UnaryOp.LOG: _protected_log,
    UnaryOp.SQRT: _protected_sqrt,
    UnaryOp.NEG: np.negative,
    UnaryOp.ABS: np.abs,
    UnaryOp.SQUARE: lambda x: x * x,
}

BINARY_FUNCS = {
    BinaryOp.ADD: np.add,
    BinaryOp.SUB: np.subtract,
    BinaryOp.MUL: np.multiply,
    BinaryOp.DIV: _protected_div,
    BinaryOp.POW: _protected_pow,
}


def _eval_damped_sin(x: np.ndarray, params: List[float]) -> np.ndarray:
    """Evaluate a * exp(-k*x) * sin(w*x + phi)"""
    a, k, w, phi = params
    decay = _protected_exp(-np.abs(k) * x)  # Force positive decay rate
    return a * decay * np.sin(w * x + phi)


def _eval_damped_cos(x: np.ndarray, params: List[float]) -> np.ndarray:
    """Evaluate a * exp(-k*x) * cos(w*x + phi)"""
    a, k, w, phi = params
    decay = _protected_exp(-np.abs(k) * x)  # Force positive decay rate
    return a * decay * np.cos(w * x + phi)


def _eval_rc_charge(x: np.ndarray, params: List[float]) -> np.ndarray:
    """Evaluate a * (1 - exp(-k*x)) + c"""
    a, k, c = params
    return a * (1 - _protected_exp(-np.abs(k) * x)) + c


def _eval_exp_decay(x: np.ndarray, params: List[float]) -> np.ndarray:
    """Evaluate a * exp(-k*x) + c"""
    a, k, c = params
    return a * _protected_exp(-np.abs(k) * x) + c


def _eval_ratio(x: np.ndarray, params: List[float]) -> np.ndarray:
    """Evaluate (ax + b) / (cx + d)"""
    a, b, c, d = params
    numerator = a * x + b
    denominator = c * x + d
    return _protected_div(numerator, denominator)


def _eval_rational2(x: np.ndarray, params: List[float]) -> np.ndarray:
    """Evaluate (ax^2 + bx + c) / (dx^2 + ex + f)"""
    a, b, c, d, e, f = params
    numerator = a * x**2 + b * x + c
    denominator = d * x**2 + e * x + f
    return _protected_div(numerator, denominator)


def _eval_sigmoid(x: np.ndarray, params: List[float]) -> np.ndarray:
    """Evaluate a / (1 + exp(-k*(x - x0)))"""
    a, k, x0 = params
    return a / (1 + _protected_exp(-k * (x - x0)))


def _eval_logistic(x: np.ndarray, params: List[float]) -> np.ndarray:
    """Evaluate a / (1 + b*exp(-k*x))"""
    a, b, k = params
    return a / (1 + np.abs(b) * _protected_exp(-k * x))


def _eval_hill(x: np.ndarray, params: List[float]) -> np.ndarray:
    """Evaluate a * x^n / (k^n + x^n) - Hill equation / Michaelis-Menten"""
    a, k, n = params
    k = np.abs(k) + EPSILON  # Ensure positive
    n = np.clip(n, 0.5, 5.0)  # Reasonable Hill coefficient range
    x_abs = np.abs(x) + EPSILON
    return a * x_abs**n / (k**n + x_abs**n)


def _eval_power_law(x: np.ndarray, params: List[float]) -> np.ndarray:
    """Evaluate a * x^b + c - power law with offset"""
    a, b, c = params
    # Use protected power for non-integer exponents
    x_abs = np.abs(x) + EPSILON
    return a * np.power(x_abs, b) + c


def _eval_gaussian(x: np.ndarray, params: List[float]) -> np.ndarray:
    """Evaluate a * exp(-((x-mu)/sigma)^2) - Gaussian/normal curve"""
    a, mu, sigma = params
    sigma = np.abs(sigma) + EPSILON  # Ensure positive width
    z = (x - mu) / sigma
    return a * np.exp(-z * z)


def _eval_tanh_step(x: np.ndarray, params: List[float]) -> np.ndarray:
    """Evaluate a * tanh(k*(x - x0)) + c - smooth step function"""
    a, k, x0, c = params
    return a * np.tanh(k * (x - x0)) + c


MACRO_FUNCS = {
    MacroOp.DAMPED_SIN: _eval_damped_sin,
    MacroOp.DAMPED_COS: _eval_damped_cos,
    MacroOp.RC_CHARGE: _eval_rc_charge,
    MacroOp.EXP_DECAY: _eval_exp_decay,
    MacroOp.RATIO: _eval_ratio,
    MacroOp.RATIONAL2: _eval_rational2,
    MacroOp.SIGMOID: _eval_sigmoid,
    MacroOp.LOGISTIC: _eval_logistic,
    MacroOp.HILL: _eval_hill,
    MacroOp.POWER_LAW: _eval_power_law,
    MacroOp.GAUSSIAN: _eval_gaussian,
    MacroOp.TANH_STEP: _eval_tanh_step,
}


# String representations for operators
UNARY_STR = {
    UnaryOp.SIN: ("sin(", ")"),
    UnaryOp.COS: ("cos(", ")"),
    UnaryOp.EXP: ("exp(", ")"),
    UnaryOp.LOG: ("log(", ")"),
    UnaryOp.SQRT: ("sqrt(", ")"),
    UnaryOp.NEG: ("-(", ")"),
    UnaryOp.ABS: ("abs(", ")"),
    UnaryOp.SQUARE: ("(", ")^2"),
}

BINARY_STR = {
    BinaryOp.ADD: " + ",
    BinaryOp.SUB: " - ",
    BinaryOp.MUL: " * ",
    BinaryOp.DIV: " / ",
    BinaryOp.POW: " ^ ",
}


@dataclass
class ExprNode:
    """
    A node in an expression tree.

    Expression trees represent mathematical formulas that can be
    evaluated, mutated, and converted to human-readable strings.
    """
    node_type: NodeType

    # For CONSTANT nodes
    value: Optional[float] = None

    # For UNARY_OP nodes
    unary_op: Optional[UnaryOp] = None
    child: Optional['ExprNode'] = None

    # For BINARY_OP nodes
    binary_op: Optional[BinaryOp] = None
    left: Optional['ExprNode'] = None
    right: Optional['ExprNode'] = None

    # For MACRO nodes (template functions with learnable parameters)
    macro_op: Optional['MacroOp'] = None
    macro_params: Optional[List[float]] = None  # Learnable constants

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the expression tree at given x values.

        Args:
            x: Input values (1D numpy array)

        Returns:
            Evaluated values (same shape as x)
        """
        x = np.asarray(x, dtype=float)

        if self.node_type == NodeType.CONSTANT:
            return np.full_like(x, self.value, dtype=float)

        elif self.node_type == NodeType.VARIABLE:
            return x.copy()

        elif self.node_type == NodeType.UNARY_OP:
            child_val = self.child.evaluate(x)
            func = UNARY_FUNCS[self.unary_op]
            return func(child_val)

        elif self.node_type == NodeType.BINARY_OP:
            left_val = self.left.evaluate(x)
            right_val = self.right.evaluate(x)
            func = BINARY_FUNCS[self.binary_op]
            return func(left_val, right_val)

        elif self.node_type == NodeType.MACRO:
            func = MACRO_FUNCS[self.macro_op]
            return func(x, self.macro_params)

        raise ValueError(f"Unknown node type: {self.node_type}")

    def to_string(self) -> str:
        """
        Convert to human-readable expression string.

        Returns:
            String representation of the expression
        """
        if self.node_type == NodeType.CONSTANT:
            # Format nicely
            if self.value == int(self.value):
                return str(int(self.value))
            return f"{self.value:.4g}"

        elif self.node_type == NodeType.VARIABLE:
            return "x"

        elif self.node_type == NodeType.UNARY_OP:
            prefix, suffix = UNARY_STR[self.unary_op]
            child_str = self.child.to_string()
            return f"{prefix}{child_str}{suffix}"

        elif self.node_type == NodeType.BINARY_OP:
            left_str = self.left.to_string()
            right_str = self.right.to_string()
            op_str = BINARY_STR[self.binary_op]
            return f"({left_str}{op_str}{right_str})"

        elif self.node_type == NodeType.MACRO:
            return self._macro_to_string()

        return "?"

    def _macro_to_string(self) -> str:
        """Convert macro node to human-readable string"""
        if self.macro_op == MacroOp.DAMPED_SIN:
            a, k, w, phi = self.macro_params
            return f"{a:.4g}*exp(-{abs(k):.4g}*x)*sin({w:.4g}*x + {phi:.4g})"
        elif self.macro_op == MacroOp.DAMPED_COS:
            a, k, w, phi = self.macro_params
            return f"{a:.4g}*exp(-{abs(k):.4g}*x)*cos({w:.4g}*x + {phi:.4g})"
        elif self.macro_op == MacroOp.RC_CHARGE:
            a, k, c = self.macro_params
            return f"{a:.4g}*(1 - exp(-{abs(k):.4g}*x)) + {c:.4g}"
        elif self.macro_op == MacroOp.EXP_DECAY:
            a, k, c = self.macro_params
            return f"{a:.4g}*exp(-{abs(k):.4g}*x) + {c:.4g}"
        elif self.macro_op == MacroOp.RATIO:
            a, b, c, d = self.macro_params
            return f"({a:.4g}*x + {b:.4g})/({c:.4g}*x + {d:.4g})"
        elif self.macro_op == MacroOp.RATIONAL2:
            a, b, c, d, e, f = self.macro_params
            return f"({a:.4g}*x^2 + {b:.4g}*x + {c:.4g})/({d:.4g}*x^2 + {e:.4g}*x + {f:.4g})"
        elif self.macro_op == MacroOp.SIGMOID:
            a, k, x0 = self.macro_params
            return f"{a:.4g}/(1 + exp(-{k:.4g}*(x - {x0:.4g})))"
        elif self.macro_op == MacroOp.LOGISTIC:
            a, b, k = self.macro_params
            return f"{a:.4g}/(1 + {abs(b):.4g}*exp(-{k:.4g}*x))"
        elif self.macro_op == MacroOp.HILL:
            a, k, n = self.macro_params
            return f"{a:.4g}*x^{n:.4g}/({abs(k):.4g}^{n:.4g} + x^{n:.4g})"
        elif self.macro_op == MacroOp.POWER_LAW:
            a, b, c = self.macro_params
            return f"{a:.4g}*x^{b:.4g} + {c:.4g}"
        elif self.macro_op == MacroOp.GAUSSIAN:
            a, mu, sigma = self.macro_params
            return f"{a:.4g}*exp(-((x - {mu:.4g})/{abs(sigma):.4g})^2)"
        elif self.macro_op == MacroOp.TANH_STEP:
            a, k, x0, c = self.macro_params
            return f"{a:.4g}*tanh({k:.4g}*(x - {x0:.4g})) + {c:.4g}"
        return f"{self.macro_op.value}({self.macro_params})"

    def depth(self) -> int:
        """Return the maximum depth of the tree"""
        if self.node_type in (NodeType.CONSTANT, NodeType.VARIABLE, NodeType.MACRO):
            return 1
        elif self.node_type == NodeType.UNARY_OP:
            return 1 + self.child.depth()
        elif self.node_type == NodeType.BINARY_OP:
            return 1 + max(self.left.depth(), self.right.depth())
        return 1

    def size(self) -> int:
        """Return the total number of nodes in the tree"""
        if self.node_type in (NodeType.CONSTANT, NodeType.VARIABLE):
            return 1
        elif self.node_type == NodeType.MACRO:
            # Macro counts as 1 node but has complexity bonus for its parameters
            return 1 + len(self.macro_params)
        elif self.node_type == NodeType.UNARY_OP:
            return 1 + self.child.size()
        elif self.node_type == NodeType.BINARY_OP:
            return 1 + self.left.size() + self.right.size()
        return 1

    def copy(self) -> 'ExprNode':
        """Create a deep copy of the tree"""
        if self.node_type == NodeType.CONSTANT:
            return ExprNode(NodeType.CONSTANT, value=self.value)
        elif self.node_type == NodeType.VARIABLE:
            return ExprNode(NodeType.VARIABLE)
        elif self.node_type == NodeType.UNARY_OP:
            return ExprNode(
                NodeType.UNARY_OP,
                unary_op=self.unary_op,
                child=self.child.copy()
            )
        elif self.node_type == NodeType.BINARY_OP:
            return ExprNode(
                NodeType.BINARY_OP,
                binary_op=self.binary_op,
                left=self.left.copy(),
                right=self.right.copy()
            )
        elif self.node_type == NodeType.MACRO:
            return ExprNode(
                NodeType.MACRO,
                macro_op=self.macro_op,
                macro_params=self.macro_params.copy()
            )
        raise ValueError(f"Unknown node type: {self.node_type}")

    def get_constants(self) -> List[float]:
        """Extract all constant values from the tree (in-order traversal)"""
        constants = []
        self._collect_constants(constants)
        return constants

    def _collect_constants(self, constants: List[float]) -> None:
        """Helper for get_constants"""
        if self.node_type == NodeType.CONSTANT:
            constants.append(self.value)
        elif self.node_type == NodeType.MACRO:
            constants.extend(self.macro_params)
        elif self.node_type == NodeType.UNARY_OP:
            self.child._collect_constants(constants)
        elif self.node_type == NodeType.BINARY_OP:
            self.left._collect_constants(constants)
            self.right._collect_constants(constants)

    def set_constants(self, constants: List[float]) -> None:
        """Set constant values in the tree (in-order traversal)"""
        self._set_constants_iter(iter(constants))

    def _set_constants_iter(self, const_iter) -> None:
        """Helper for set_constants"""
        if self.node_type == NodeType.CONSTANT:
            self.value = next(const_iter)
        elif self.node_type == NodeType.MACRO:
            for i in range(len(self.macro_params)):
                self.macro_params[i] = next(const_iter)
        elif self.node_type == NodeType.UNARY_OP:
            self.child._set_constants_iter(const_iter)
        elif self.node_type == NodeType.BINARY_OP:
            self.left._set_constants_iter(const_iter)
            self.right._set_constants_iter(const_iter)

    def get_all_nodes(self) -> List['ExprNode']:
        """Get a list of all nodes in the tree"""
        nodes = [self]
        if self.node_type == NodeType.UNARY_OP:
            nodes.extend(self.child.get_all_nodes())
        elif self.node_type == NodeType.BINARY_OP:
            nodes.extend(self.left.get_all_nodes())
            nodes.extend(self.right.get_all_nodes())
        # MACRO and CONSTANT/VARIABLE are leaf nodes
        return nodes


@dataclass
class PrimitiveSet:
    """
    Configuration for available operators and terminals.

    Defines which mathematical operations are available during
    symbolic regression evolution.
    """

    # Available unary operators
    unary_ops: List[UnaryOp] = field(default_factory=lambda: [
        UnaryOp.SIN, UnaryOp.COS, UnaryOp.EXP, UnaryOp.LOG,
        UnaryOp.SQRT, UnaryOp.NEG, UnaryOp.SQUARE
    ])

    # Available binary operators
    binary_ops: List[BinaryOp] = field(default_factory=lambda: [
        BinaryOp.ADD, BinaryOp.SUB, BinaryOp.MUL, BinaryOp.DIV
    ])

    # Available macro operators (template functions)
    macro_ops: List[MacroOp] = field(default_factory=list)

    # Probability of generating a constant (vs variable) at terminal
    constant_probability: float = 0.3

    # Probability of generating a macro (when macros available)
    macro_probability: float = 0.2

    # Range for random constant generation
    constant_range: Tuple[float, float] = (-5.0, 5.0)

    # Special constants to include with higher probability
    special_constants: List[float] = field(default_factory=lambda: [
        0.0, 1.0, -1.0, 2.0, 0.5, np.pi, np.e
    ])


# Domain-specific primitive sets
TRIG_PRIMITIVES = PrimitiveSet(
    unary_ops=[UnaryOp.SIN, UnaryOp.COS, UnaryOp.SQUARE, UnaryOp.NEG],
    binary_ops=[BinaryOp.ADD, BinaryOp.MUL],
    special_constants=[0.0, 1.0, np.pi, 2*np.pi]
)

GROWTH_PRIMITIVES = PrimitiveSet(
    unary_ops=[UnaryOp.EXP, UnaryOp.LOG, UnaryOp.SQRT, UnaryOp.SQUARE],
    binary_ops=[BinaryOp.ADD, BinaryOp.SUB, BinaryOp.MUL, BinaryOp.DIV]
)

POLYNOMIAL_PRIMITIVES = PrimitiveSet(
    unary_ops=[UnaryOp.SQUARE, UnaryOp.NEG],
    binary_ops=[BinaryOp.ADD, BinaryOp.SUB, BinaryOp.MUL],
    constant_probability=0.4
)

# Physics-oriented primitive sets with macros for common patterns
OSCILLATOR_PRIMITIVES = PrimitiveSet(
    unary_ops=[UnaryOp.SIN, UnaryOp.COS, UnaryOp.EXP, UnaryOp.NEG],
    binary_ops=[BinaryOp.ADD, BinaryOp.SUB, BinaryOp.MUL],
    macro_ops=[MacroOp.DAMPED_SIN, MacroOp.DAMPED_COS],
    macro_probability=0.3,
    special_constants=[0.0, 1.0, np.pi, 2*np.pi]
)

CIRCUIT_PRIMITIVES = PrimitiveSet(
    unary_ops=[UnaryOp.EXP, UnaryOp.LOG, UnaryOp.NEG],
    binary_ops=[BinaryOp.ADD, BinaryOp.SUB, BinaryOp.MUL, BinaryOp.DIV],
    macro_ops=[MacroOp.RC_CHARGE, MacroOp.EXP_DECAY],
    macro_probability=0.3
)

# Combined physics primitives for general use
PHYSICS_PRIMITIVES = PrimitiveSet(
    unary_ops=[UnaryOp.SIN, UnaryOp.COS, UnaryOp.EXP, UnaryOp.LOG, UnaryOp.NEG],
    binary_ops=[BinaryOp.ADD, BinaryOp.SUB, BinaryOp.MUL],
    macro_ops=[MacroOp.DAMPED_SIN, MacroOp.DAMPED_COS, MacroOp.RC_CHARGE, MacroOp.EXP_DECAY],
    macro_probability=0.25,
    special_constants=[0.0, 1.0, -1.0, np.pi, 2*np.pi]
)

# Minimal primitives for pure discovery (no macros)
MINIMAL_PRIMITIVES = PrimitiveSet(
    unary_ops=[UnaryOp.SIN, UnaryOp.COS, UnaryOp.EXP, UnaryOp.LOG,
               UnaryOp.SQRT, UnaryOp.NEG, UnaryOp.SQUARE],
    binary_ops=[BinaryOp.ADD, BinaryOp.SUB, BinaryOp.MUL, BinaryOp.DIV],
    macro_ops=[],  # No macros - pure discovery
    macro_probability=0.0,
)

# Rational function primitives
RATIONAL_PRIMITIVES = PrimitiveSet(
    unary_ops=[UnaryOp.NEG, UnaryOp.SQUARE],
    binary_ops=[BinaryOp.ADD, BinaryOp.SUB, BinaryOp.MUL, BinaryOp.DIV],
    macro_ops=[MacroOp.RATIO, MacroOp.RATIONAL2],
    macro_probability=0.35,
)

# Saturation/sigmoid primitives (updated GROWTH_PRIMITIVES with macros)
SATURATION_PRIMITIVES = PrimitiveSet(
    unary_ops=[UnaryOp.EXP, UnaryOp.LOG, UnaryOp.NEG],
    binary_ops=[BinaryOp.ADD, BinaryOp.SUB, BinaryOp.MUL, BinaryOp.DIV],
    macro_ops=[MacroOp.SIGMOID, MacroOp.LOGISTIC, MacroOp.HILL],
    macro_probability=0.35,
)

# Universal primitives - common forms across many domains
UNIVERSAL_PRIMITIVES = PrimitiveSet(
    unary_ops=[UnaryOp.EXP, UnaryOp.LOG, UnaryOp.SQRT, UnaryOp.NEG, UnaryOp.SQUARE],
    binary_ops=[BinaryOp.ADD, BinaryOp.SUB, BinaryOp.MUL, BinaryOp.DIV],
    macro_ops=[MacroOp.POWER_LAW, MacroOp.GAUSSIAN, MacroOp.TANH_STEP],
    macro_probability=0.35,
)

# All macros - for IDENTIFY mode (maximum template coverage)
ALL_MACROS_PRIMITIVES = PrimitiveSet(
    unary_ops=[UnaryOp.SIN, UnaryOp.COS, UnaryOp.EXP, UnaryOp.LOG,
               UnaryOp.SQRT, UnaryOp.NEG, UnaryOp.SQUARE],
    binary_ops=[BinaryOp.ADD, BinaryOp.SUB, BinaryOp.MUL, BinaryOp.DIV],
    macro_ops=[
        MacroOp.DAMPED_SIN, MacroOp.DAMPED_COS,
        MacroOp.RC_CHARGE, MacroOp.EXP_DECAY,
        MacroOp.RATIO, MacroOp.RATIONAL2,
        MacroOp.SIGMOID, MacroOp.LOGISTIC, MacroOp.HILL,
        MacroOp.POWER_LAW, MacroOp.GAUSSIAN, MacroOp.TANH_STEP,
    ],
    macro_probability=0.4,
)


class DiscoveryMode(Enum):
    """
    How the symbolic regressor searches for laws.

    Controls the balance between structured template matching
    and open-ended discovery.
    """
    # Automatic hierarchical search - probes domains, uses residual analysis
    AUTO = auto()

    # Law identification - macro-heavy, fast, assumes known form families
    IDENTIFY = auto()

    # Pure discovery - no macros, maximum flexibility for novel forms
    DISCOVER = auto()

    # Domain-specific modes
    OSCILLATOR = auto()   # Vibrations, waves, damped systems
    CIRCUIT = auto()      # RC/RLC, charging, transfer functions
    GROWTH = auto()       # Population, adoption, sigmoids
    RATIONAL = auto()     # Ratios, saturation, feedback
    POLYNOMIAL = auto()   # Pure algebraic relationships
    UNIVERSAL = auto()    # Power laws, Gaussians, smooth steps


# Map discovery modes to primitive sets
MODE_TO_PRIMITIVES = {
    DiscoveryMode.OSCILLATOR: OSCILLATOR_PRIMITIVES,
    DiscoveryMode.CIRCUIT: CIRCUIT_PRIMITIVES,
    DiscoveryMode.GROWTH: SATURATION_PRIMITIVES,
    DiscoveryMode.RATIONAL: RATIONAL_PRIMITIVES,
    DiscoveryMode.POLYNOMIAL: POLYNOMIAL_PRIMITIVES,
    DiscoveryMode.UNIVERSAL: UNIVERSAL_PRIMITIVES,
    DiscoveryMode.DISCOVER: MINIMAL_PRIMITIVES,
    DiscoveryMode.IDENTIFY: ALL_MACROS_PRIMITIVES,
}

# Domain probe order for AUTO mode (most structured -> least structured)
DOMAIN_PROBE_ORDER = [
    ("oscillator", OSCILLATOR_PRIMITIVES),
    ("circuit", CIRCUIT_PRIMITIVES),
    ("growth", SATURATION_PRIMITIVES),
    ("rational", RATIONAL_PRIMITIVES),
    ("universal", UNIVERSAL_PRIMITIVES),
    ("polynomial", POLYNOMIAL_PRIMITIVES),
]


@dataclass
class FitnessResult:
    """Multi-objective fitness evaluation result"""
    mse: float                    # Mean squared error
    r_squared: float              # Coefficient of determination
    complexity: int               # Tree size (number of nodes)
    depth: int                    # Tree depth
    score: float = 0.0            # Combined parsimony-pressure score

    def dominates(self, other: 'FitnessResult') -> bool:
        """
        Check if this solution Pareto-dominates another.

        Returns True if this solution is at least as good in all objectives
        and strictly better in at least one.
        """
        better_or_equal_all = (
            self.mse <= other.mse and
            self.complexity <= other.complexity
        )
        strictly_better_one = (
            self.mse < other.mse or
            self.complexity < other.complexity
        )
        return better_or_equal_all and strictly_better_one


@dataclass
class SymbolicFitResult:
    """
    Result of symbolic regression fitting.

    Mirrors the FitResult structure from detector.py for compatibility,
    but includes the discovered expression tree.
    """
    expression: ExprNode
    expression_string: str
    r_squared: float
    mse: float
    complexity: int
    residuals: np.ndarray
    residual_std: float
    is_noise_like: bool
    params: np.ndarray = field(default_factory=lambda: np.array([]))

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the discovered expression at new x values"""
        return self.expression.evaluate(x)


@dataclass
class SymbolicRegressionResult:
    """
    Complete result of symbolic regression analysis.

    Contains the Pareto front of solutions showing the tradeoff
    between accuracy and complexity.
    """

    # Pareto front of non-dominated solutions
    pareto_front: List[SymbolicFitResult] = field(default_factory=list)

    # Best solutions by different criteria
    most_accurate: Optional[SymbolicFitResult] = None
    most_parsimonious: Optional[SymbolicFitResult] = None
    best_tradeoff: Optional[SymbolicFitResult] = None  # Knee of Pareto front

    # Search statistics
    generations_run: int = 0
    total_evaluations: int = 0

    # Metadata from auto-discovery (domain detected, residual analysis, etc.)
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Generate a human-readable summary"""
        lines = [
            "=" * 60,
            "SYMBOLIC REGRESSION RESULT",
            "=" * 60,
            "",
        ]

        if self.best_tradeoff:
            lines.extend([
                "Best Tradeoff (accuracy vs complexity):",
                f"  Expression: {self.best_tradeoff.expression_string}",
                f"  R-squared:  {self.best_tradeoff.r_squared:.4f}",
                f"  Complexity: {self.best_tradeoff.complexity} nodes",
                "",
            ])

        if self.most_accurate and self.most_accurate != self.best_tradeoff:
            lines.extend([
                "Most Accurate:",
                f"  Expression: {self.most_accurate.expression_string}",
                f"  R-squared:  {self.most_accurate.r_squared:.4f}",
                f"  Complexity: {self.most_accurate.complexity} nodes",
                "",
            ])

        if self.most_parsimonious and self.most_parsimonious != self.best_tradeoff:
            lines.extend([
                "Most Parsimonious (simplest):",
                f"  Expression: {self.most_parsimonious.expression_string}",
                f"  R-squared:  {self.most_parsimonious.r_squared:.4f}",
                f"  Complexity: {self.most_parsimonious.complexity} nodes",
                "",
            ])

        lines.extend([
            f"Pareto front size: {len(self.pareto_front)} solutions",
            f"Generations: {self.generations_run}",
            f"Total evaluations: {self.total_evaluations}",
        ])

        # Include metadata if present
        if self.metadata:
            lines.append("")
            lines.append("Auto-discovery metadata:")
            if "domain_detected" in self.metadata:
                lines.append(f"  Domain detected: {self.metadata['domain_detected']}")
            if "residual_structure" in self.metadata:
                lines.append(f"  Residual structure: {self.metadata['residual_structure']:.4f}")
            if "correction_applied" in self.metadata:
                lines.append(f"  Correction applied: {self.metadata['correction_applied']}")
            if "probe_results" in self.metadata:
                lines.append("  Probe results:")
                for domain, r2 in self.metadata["probe_results"].items():
                    lines.append(f"    {domain}: R²={r2:.4f}")

        return "\n".join(lines)
