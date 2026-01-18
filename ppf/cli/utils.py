"""
Shared utilities for the PPF CLI.

This module provides common functions used across CLI commands:
- Data loading from CSV files and stdin
- Output formatting (JSON and human-readable)
- Error handling and messaging
"""

import csv
import json
import sys
from dataclasses import asdict, is_dataclass
from enum import Enum
from io import StringIO
from typing import Any, Dict, IO, List, Optional, Tuple, Union

import numpy as np


def load_data(
    args,
    require_xy: bool = True,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Load x,y data from a CSV file or stdin.

    Args:
        args: Parsed command-line arguments with:
            - file: Path to CSV file (or None)
            - stdin: Whether to read from stdin
            - x_column: X column name or index (optional)
            - y_column: Y column name or index
            - delimiter: CSV delimiter
            - skip_header: Number of header rows to skip
        require_xy: If True, require both x and y; if False, y only is OK

    Returns:
        Tuple of (x_array, y_array). x_array may be None if not specified
        and require_xy is False.

    Raises:
        ValueError: If data cannot be loaded or parsed
    """
    # Determine input source
    if hasattr(args, "stdin") and args.stdin:
        content = sys.stdin.read()
        if not content.strip():
            raise ValueError("No data received from stdin")
        input_file = StringIO(content)
        filename = "<stdin>"
    elif hasattr(args, "file") and args.file:
        try:
            input_file = open(args.file, "r", encoding="utf-8")
            filename = args.file
        except FileNotFoundError:
            raise ValueError(f"File not found: {args.file}")
        except IOError as e:
            raise ValueError(f"Cannot read file '{args.file}': {e}")
    else:
        raise ValueError("No input specified. Provide a FILE or use --stdin")

    try:
        # Parse CSV options
        delimiter = getattr(args, "delimiter", ",") or ","
        skip_header = getattr(args, "skip_header", 0) or 0

        # Read CSV
        reader = csv.reader(input_file, delimiter=delimiter)
        rows = list(reader)

        if not rows:
            raise ValueError(f"Empty file: {filename}")

        # Skip header rows
        if skip_header > 0:
            rows = rows[skip_header:]

        if not rows:
            raise ValueError(f"No data rows after skipping {skip_header} headers")

        # Check if first row is headers (non-numeric)
        first_row = rows[0]
        has_header = False
        try:
            float(first_row[0])
        except ValueError:
            has_header = True

        if has_header:
            headers = first_row
            data_rows = rows[1:]
        else:
            headers = None
            data_rows = rows

        if not data_rows:
            raise ValueError("No data rows found in file")

        # Parse column specifications
        x_col = getattr(args, "x_column", None)
        y_col = getattr(args, "y_column", None)

        # Resolve column indices
        x_idx = _resolve_column(x_col, headers, "x") if x_col else None
        y_idx = _resolve_column(y_col, headers, "y") if y_col else None

        # Default: if no columns specified, use first column as x, second as y
        # Or if only one column, use index as x
        num_cols = len(data_rows[0]) if data_rows else 0

        if y_idx is None:
            if num_cols >= 2:
                y_idx = 1
            elif num_cols == 1:
                y_idx = 0
            else:
                raise ValueError("Cannot determine y column")

        if x_idx is None and require_xy:
            if num_cols >= 2:
                x_idx = 0
            # else: x will be None, using indices

        # Extract data
        y_values = []
        x_values = [] if x_idx is not None else None

        for i, row in enumerate(data_rows):
            try:
                y_val = float(row[y_idx])
                y_values.append(y_val)

                if x_idx is not None:
                    x_val = float(row[x_idx])
                    x_values.append(x_val)
            except (IndexError, ValueError) as e:
                raise ValueError(f"Error parsing row {i + 1}: {e}")

        y_array = np.array(y_values)

        if x_values is not None:
            x_array = np.array(x_values)
        elif require_xy:
            # Generate index-based x values
            x_array = np.arange(len(y_array), dtype=float)
        else:
            x_array = None

        return x_array, y_array

    finally:
        if hasattr(args, "file") and args.file and input_file:
            input_file.close()


def _resolve_column(
    col_spec: Union[str, int, None],
    headers: Optional[List[str]],
    col_name: str,
) -> int:
    """
    Resolve a column specification to an index.

    Args:
        col_spec: Column name (string) or index (int/string digit)
        headers: List of header names, or None
        col_name: Name for error messages ("x" or "y")

    Returns:
        Column index (0-based)
    """
    if col_spec is None:
        return None

    # Try as integer index first
    try:
        idx = int(col_spec)
        return idx
    except ValueError:
        pass

    # Try as column name
    if headers is not None:
        try:
            return headers.index(col_spec)
        except ValueError:
            raise ValueError(
                f"{col_name} column '{col_spec}' not found. "
                f"Available columns: {', '.join(headers)}"
            )

    raise ValueError(
        f"Cannot resolve {col_name} column '{col_spec}' - "
        "no headers found and it's not a numeric index"
    )


def format_output(
    result: Any,
    args,
    output: IO[str] = None,
) -> str:
    """
    Format a result object for output.

    Args:
        result: Result object (dataclass, dict, or has summary() method)
        args: Parsed arguments with --json flag
        output: Output stream (defaults to returning string)

    Returns:
        Formatted string
    """
    if output is None:
        output = StringIO()
        return_string = True
    else:
        return_string = False

    use_json = getattr(args, "json", False)

    if use_json:
        json_data = _to_json_serializable(result)
        json.dump(json_data, output, indent=2)
        output.write("\n")
    else:
        # Human-readable output
        if hasattr(result, "summary"):
            output.write(result.summary())
        elif is_dataclass(result):
            _format_dataclass(result, output)
        elif isinstance(result, dict):
            _format_dict(result, output)
        else:
            output.write(str(result))
        output.write("\n")

    if return_string:
        return output.getvalue()


def _to_json_serializable(obj: Any) -> Any:
    """Convert an object to a JSON-serializable form."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Enum):
        return obj.name
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    if is_dataclass(obj):
        return _to_json_serializable(asdict(obj))
    if hasattr(obj, "to_dict"):
        return _to_json_serializable(obj.to_dict())
    if hasattr(obj, "__dict__"):
        return _to_json_serializable(obj.__dict__)
    return str(obj)


def _format_dataclass(obj: Any, output: IO[str], indent: int = 0) -> None:
    """Format a dataclass for human-readable output."""
    prefix = "  " * indent
    for field_name in obj.__dataclass_fields__:
        value = getattr(obj, field_name)
        if is_dataclass(value):
            output.write(f"{prefix}{field_name}:\n")
            _format_dataclass(value, output, indent + 1)
        elif isinstance(value, list) and len(value) > 0 and is_dataclass(value[0]):
            output.write(f"{prefix}{field_name}:\n")
            for i, item in enumerate(value):
                output.write(f"{prefix}  [{i}]:\n")
                _format_dataclass(item, output, indent + 2)
        elif isinstance(value, np.ndarray):
            if value.size <= 10:
                output.write(f"{prefix}{field_name}: {value.tolist()}\n")
            else:
                output.write(f"{prefix}{field_name}: array({value.shape})\n")
        else:
            output.write(f"{prefix}{field_name}: {value}\n")


def _format_dict(obj: Dict, output: IO[str], indent: int = 0) -> None:
    """Format a dictionary for human-readable output."""
    prefix = "  " * indent
    for key, value in obj.items():
        if isinstance(value, dict):
            output.write(f"{prefix}{key}:\n")
            _format_dict(value, output, indent + 1)
        elif isinstance(value, np.ndarray):
            if value.size <= 10:
                output.write(f"{prefix}{key}: {value.tolist()}\n")
            else:
                output.write(f"{prefix}{key}: array({value.shape})\n")
        else:
            output.write(f"{prefix}{key}: {value}\n")


def print_error(
    message: str,
    hint: Optional[str] = None,
    file: IO[str] = None,
) -> None:
    """
    Print a formatted error message.

    Args:
        message: The error message
        hint: Optional helpful hint for resolution
        file: Output stream (defaults to stderr)
    """
    if file is None:
        file = sys.stderr

    file.write(f"Error: {message}\n")
    if hint:
        file.write(f"Hint: {hint}\n")


def print_warning(
    message: str,
    file: IO[str] = None,
) -> None:
    """Print a formatted warning message."""
    if file is None:
        file = sys.stderr

    file.write(f"Warning: {message}\n")


def print_progress(
    message: str,
    args,
    file: IO[str] = None,
) -> None:
    """
    Print a progress message if verbose mode is enabled.

    Args:
        message: Progress message
        args: Parsed arguments with --verbose and --quiet flags
        file: Output stream (defaults to stderr)
    """
    if file is None:
        file = sys.stderr

    if getattr(args, "verbose", False) and not getattr(args, "quiet", False):
        file.write(f"{message}\n")


def add_data_input_args(parser) -> None:
    """
    Add common data input arguments to a parser.

    Adds: FILE, --stdin, -x/--x-column, -y/--y-column, --delimiter, --skip-header
    """
    parser.add_argument(
        "file",
        nargs="?",
        metavar="FILE",
        help="Path to CSV data file (use --stdin for piped input)",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read data from stdin instead of a file",
    )
    parser.add_argument(
        "-x", "--x-column",
        metavar="COL",
        help="X (independent variable) column name or index (default: first column)",
    )
    parser.add_argument(
        "-y", "--y-column",
        metavar="COL",
        help="Y (dependent variable) column name or index (default: second column)",
    )
    parser.add_argument(
        "--delimiter",
        metavar="CHAR",
        default=",",
        help="CSV delimiter character (default: comma)",
    )
    parser.add_argument(
        "--skip-header",
        type=int,
        default=0,
        metavar="N",
        help="Skip N header rows before parsing (default: auto-detect)",
    )


def validate_positive_int(value: str, name: str) -> int:
    """Validate that a string is a positive integer."""
    try:
        val = int(value)
        if val <= 0:
            raise ValueError()
        return val
    except ValueError:
        raise ValueError(f"{name} must be a positive integer, got: {value}")


def validate_positive_float(value: str, name: str) -> float:
    """Validate that a string is a positive float."""
    try:
        val = float(value)
        if val <= 0:
            raise ValueError()
        return val
    except ValueError:
        raise ValueError(f"{name} must be a positive number, got: {value}")


def validate_fraction(value: str, name: str) -> float:
    """Validate that a string is a float between 0 and 1."""
    try:
        val = float(value)
        if not (0 <= val <= 1):
            raise ValueError()
        return val
    except ValueError:
        raise ValueError(f"{name} must be between 0 and 1, got: {value}")
