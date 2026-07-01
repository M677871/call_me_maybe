"""Validation for generated function calls."""
from __future__ import annotations
from typing import Any
from src.io_utils import ProjectError
from src.models import FunctionCallResult, FunctionDefinition


def validate_result(
    result: FunctionCallResult,
    functions: list[FunctionDefinition],
) -> FunctionCallResult:
    """Validate output against function definitions."""
    function_map = {function.name: function for function in functions}
    if result.name not in function_map:
        raise ProjectError(f"Unknown function generated: {result.name}")
    function = function_map[result.name]
    expected_parameters = function.parameters
    actual_keys = set(result.parameters.keys())
    expected_keys = set(expected_parameters.keys())
    if actual_keys != expected_keys:
        raise ProjectError(
            f"Wrong parameters for {result.name}. "
            f"Expected {expected_keys}, got {actual_keys}."
        )
    for parameter_name, parameter_spec in expected_parameters.items():
        value = result.parameters[parameter_name]
        _validate_parameter_type(parameter_name, value, parameter_spec.type)
    return result


def _validate_parameter_type(
    parameter_name: str,
    value: Any,
    expected_type: str,
) -> None:
    """Validate one parameter value type."""
    if expected_type == "string" and not isinstance(value, str):
        raise ProjectError(f"Parameter {parameter_name} must be string.")
    if expected_type == "boolean" and not isinstance(value, bool):
        raise ProjectError(f"Parameter {parameter_name} must be boolean.")
    if expected_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ProjectError(f"Parameter {parameter_name} must be integer.")
    if expected_type == "number":
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ProjectError(f"Parameter {parameter_name} must be number.")
