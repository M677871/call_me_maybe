"""Validation helpers for generated function calls."""

from __future__ import annotations

import math
from typing import Any

from src.io_utils import ProjectError
from src.models import FunctionCallResult, FunctionDefinition


def validate_result(
    result: FunctionCallResult,
    functions: list[FunctionDefinition],
) -> FunctionCallResult:
    """Validate one output object against the function definitions."""
    function = _find_function(result.name, functions)
    expected_keys = set(function.parameters.keys())
    actual_keys = set(result.parameters.keys())
    if actual_keys != expected_keys:
        raise ProjectError(
            f"Wrong parameters for {result.name}. "
            f"Expected {expected_keys}, got {actual_keys}."
        )

    for name, spec in function.parameters.items():
        value = result.parameters[name]
        if not matches_type(value, spec.type):
            raise ProjectError(
                f"Parameter {name} must be {spec.type} for {result.name}."
            )
    return result


def matches_type(value: Any, expected_type: str) -> bool:
    """Return True when value matches a supported JSON schema type."""
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
    return False


def schema_default(expected_type: str) -> object:
    """Return a generic safe default based only on the schema type."""
    if expected_type == "string":
        return ""
    if expected_type == "boolean":
        return False
    if expected_type == "integer":
        return 0
    if expected_type == "number":
        return 0.0
    return ""


def _find_function(
    function_name: str,
    functions: list[FunctionDefinition],
) -> FunctionDefinition:
    """Find a function definition by name."""
    for function in functions:
        if function.name == function_name:
            return function
    raise ProjectError(f"Unknown function generated: {function_name}")
