"""Prompt construction for the LLM."""
from __future__ import annotations
from src.models import FunctionDefinition
def build_function_selection_prompt(
    user_prompt: str,
    functions: list[FunctionDefinition],
) -> str:
    """Build prompt asking the LLM to choose the correct function."""
    lines: list[str] = []
    lines.append("You are a function-calling AI.")
    lines.append("Choose the best function for the user request.")
    lines.append("Return only the function name.")
    lines.append("")
    lines.append("Available functions:")
    for function in functions:
        lines.append(f"- {function.name}: {function.description}")
        if function.parameters:
            lines.append("  Parameters:")
            for name, spec in function.parameters.items():
                lines.append(f"  - {name}: {spec.type}")
    lines.append("")
    lines.append(f'User request: "{user_prompt}"')
    lines.append("")
    lines.append("Function name:")
    return "\n".join(lines)
def build_parameter_prompt(
    user_prompt: str,
    function: FunctionDefinition,
    parameter_name: str,
    parameter_type: str,
) -> str:
    """Build prompt asking the LLM to extract one parameter value."""
    return "\n".join(
        [
            "You are extracting one function argument.",
            "Return only the JSON value.",
            "",
            f'User request: "{user_prompt}"',
            f"Function name: {function.name}",
            f"Function description: {function.description}",
            f"Parameter name: {parameter_name}",
            f"Parameter type: {parameter_type}",
            "",
            "JSON value:",
        ]
    )
