"""Prompt builders kept separate from the pipeline logic."""

from __future__ import annotations

import json

from src.models import FunctionDefinition


def build_function_selection_prompt(
    user_prompt: str,
    functions: list[FunctionDefinition],
) -> str:
    """Build a prompt asking the model to choose one function name."""
    lines = [
        "You are a function router.",
        "Choose the single best function for the user request.",
        "Use only the available function descriptions and schemas.",
        "Return only the function name.",
        "",
        "Available functions:",
    ]
    for function in functions:
        lines.append(f"- name: {function.name}")
        lines.append(f"  description: {function.description}")
        lines.append(f"  parameters: {_format_parameters(function)}")
    lines.extend(
        [
            "",
            f"User request: {user_prompt}",
            "Function name:",
        ]
    )
    return "\n".join(lines)


def build_arguments_prompt(
    user_prompt: str,
    function: FunctionDefinition,
    error: str | None = None,
) -> str:
    """Build a prompt asking the model to produce a JSON arguments object."""
    schema = {
        name: spec.type for name, spec in function.parameters.items()
    }
    lines = [
        "You are extracting function arguments.",
        "Return only one JSON object.",
        "The JSON object must contain exactly the required parameters.",
        "Do not include the function name.",
        "Do not include markdown or explanation.",
        "Use values from the user request.",
        "Do not calculate the final function result.",
        "",
        f"User request: {user_prompt}",
        f"Function name: {function.name}",
        f"Function description: {function.description}",
        f"Required parameter schema: {json.dumps(schema)}",
    ]
    if error is not None:
        lines.extend(
            [
                "",
                f"Previous invalid answer: {error}",
                "Try again with only the JSON arguments object.",
            ]
        )
    lines.extend(["", "JSON arguments:"])
    return "\n".join(lines)


def _format_parameters(function: FunctionDefinition) -> str:
    """Return a compact display of the function parameters."""
    if not function.parameters:
        return "none"
    return ", ".join(
        f"{name}: {spec.type}"
        for name, spec in function.parameters.items()
    )
