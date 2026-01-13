"""
Load PPF model bundles from JSON.

Reconstructs expression trees from serialized format.
"""

from typing import Dict, Any, Tuple, List, Optional
import json

from ..symbolic_types import (
    ExprNode, NodeType, UnaryOp, BinaryOp, MacroOp,
    MACRO_PARAM_COUNT,
)


# Supported schema versions
SUPPORTED_SCHEMAS = ["ppf.export.model.v1"]


class SchemaVersionError(Exception):
    """Raised when JSON bundle has unsupported schema version."""
    pass


class UnsupportedNodeError(Exception):
    """Raised when tree contains unknown node types."""
    pass


class UnsupportedOperatorError(Exception):
    """Raised when tree contains unknown operators."""
    pass


def _parse_unary_op(name: str) -> UnaryOp:
    """Parse unary operator name to enum."""
    try:
        return UnaryOp[name]
    except KeyError:
        raise UnsupportedOperatorError(f"Unknown unary operator: {name}")


def _parse_binary_op(name: str) -> BinaryOp:
    """Parse binary operator name to enum."""
    try:
        return BinaryOp[name]
    except KeyError:
        raise UnsupportedOperatorError(f"Unknown binary operator: {name}")


def _parse_macro_op(name: str) -> MacroOp:
    """Parse macro operator name to enum."""
    try:
        return MacroOp[name]
    except KeyError:
        raise UnsupportedOperatorError(f"Unknown macro operator: {name}")


def _deserialize_tree(tree_dict: Dict[str, Any], variables: List[str] = None) -> ExprNode:
    """
    Deserialize a tree dictionary back to an ExprNode.

    Args:
        tree_dict: Dictionary representation of expression tree
        variables: List of variable names in the expression

    Returns:
        Reconstructed ExprNode
    """
    if variables is None:
        variables = ["t"]

    node_type = tree_dict.get("type")

    if node_type == "CONST":
        return ExprNode(NodeType.CONSTANT, value=float(tree_dict["value"]))

    elif node_type == "VAR":
        # Currently we only support single-variable expressions
        # The variable name is stored but ExprNode uses implicit "x"
        return ExprNode(NodeType.VARIABLE)

    elif node_type == "UNARY_OP":
        op = _parse_unary_op(tree_dict["op"])
        child = _deserialize_tree(tree_dict["child"], variables)
        return ExprNode(NodeType.UNARY_OP, unary_op=op, child=child)

    elif node_type == "BINARY_OP":
        op = _parse_binary_op(tree_dict["op"])
        children = tree_dict["children"]
        left = _deserialize_tree(children[0], variables)
        right = _deserialize_tree(children[1], variables)
        return ExprNode(NodeType.BINARY_OP, binary_op=op, left=left, right=right)

    elif node_type == "MACRO_CALL":
        macro_name = tree_dict["name"]
        macro_op = _parse_macro_op(macro_name)

        # Extract parameters from args (constants only, skip VAR nodes)
        args = tree_dict.get("args", [])
        params = []
        for arg in args:
            if arg.get("type") == "CONST":
                params.append(float(arg["value"]))
            # VAR nodes are skipped - the variable is implicit in evaluation

        # Verify parameter count matches expected
        expected_count = MACRO_PARAM_COUNT.get(macro_op, 0)
        if len(params) != expected_count:
            # Allow flexibility - some serializations might include extra info
            if len(params) > expected_count:
                params = params[:expected_count]
            else:
                # Pad with defaults if needed
                while len(params) < expected_count:
                    params.append(1.0)

        return ExprNode(NodeType.MACRO, macro_op=macro_op, macro_params=params)

    else:
        raise UnsupportedNodeError(f"Unknown node type: {node_type}")


def load_json(bundle: Dict[str, Any]) -> Tuple[ExprNode, Dict[str, Any]]:
    """
    Reconstruct expression and metadata from JSON bundle.

    Args:
        bundle: Dictionary loaded from JSON model bundle

    Returns:
        Tuple of (ExprNode, metadata_dict)
        - expr: Reconstructed expression tree
        - metadata: Dict containing metrics, parameters, constraints, etc.

    Raises:
        SchemaVersionError: If bundle schema is not supported
        UnsupportedNodeError: If tree contains unknown node types
        UnsupportedOperatorError: If tree contains unknown operators

    Example:
        >>> with open("model.json") as f:
        ...     bundle = json.load(f)
        >>> expr, metadata = load_json(bundle)
        >>> y_pred = expr.evaluate(x_new)
    """
    # Validate schema
    schema = bundle.get("schema")
    if schema not in SUPPORTED_SCHEMAS:
        raise SchemaVersionError(
            f"Unsupported schema version: {schema}. "
            f"Supported versions: {SUPPORTED_SCHEMAS}"
        )

    # Get variables list
    variables = bundle.get("variables", ["t"])

    # Reconstruct expression tree
    expression_data = bundle.get("expression", {})
    tree_dict = expression_data.get("tree")

    if tree_dict is None:
        # Tree not included - could try to parse from string in future
        raise ValueError(
            "Bundle does not include expression tree. "
            "String-only bundles are not yet supported."
        )

    expr = _deserialize_tree(tree_dict, variables)

    # Collect metadata
    metadata = {
        "source": bundle.get("source"),
        "ppf_version": bundle.get("ppf_version"),
        "created_utc": bundle.get("created_utc"),
        "metrics": bundle.get("metrics", {}),
        "parameters": bundle.get("parameters", {}),
        "constraints": bundle.get("constraints", {}),
        "variables": variables,
        "expression_string": expression_data.get("string"),
        "expression_latex": expression_data.get("latex"),
        "original_metadata": bundle.get("metadata", {}),
    }

    return expr, metadata


def load_json_file(filepath: str) -> Tuple[ExprNode, Dict[str, Any]]:
    """
    Load expression from a JSON file.

    Args:
        filepath: Path to JSON model bundle file

    Returns:
        Tuple of (ExprNode, metadata_dict)
    """
    with open(filepath, 'r') as f:
        bundle = json.load(f)
    return load_json(bundle)


def load_json_string(json_str: str) -> Tuple[ExprNode, Dict[str, Any]]:
    """
    Load expression from a JSON string.

    Args:
        json_str: JSON model bundle string

    Returns:
        Tuple of (ExprNode, metadata_dict)
    """
    bundle = json.loads(json_str)
    return load_json(bundle)
