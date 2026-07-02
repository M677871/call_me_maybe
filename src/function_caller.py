"""High-level function-calling pipeline."""

from __future__ import annotations

import json

from src.constrained_decoder import ConstrainedDecoder
from src.constraints import ArgumentsConstraint, EnumConstraint
from src.llm_adapter import LLMAdapter
from src.models import FunctionCallResult, FunctionDefinition, PromptItem
from src.prompt_builder import (
    build_arguments_prompt,
    build_function_selection_prompt,
)
from src.validator import matches_type, schema_default, validate_result

MAX_RETRIES = 2


class FunctionCallingEngine:
    """Convert natural-language prompts into schema-valid function calls."""

    def __init__(self, functions: list[FunctionDefinition]) -> None:
        """Create the engine with the available function definitions."""
        self.functions = functions
        self.llm = LLMAdapter()
        self.decoder = ConstrainedDecoder(self.llm)

    def process_all(
        self,
        prompts: list[PromptItem],
    ) -> list[FunctionCallResult]:
        """Process all prompts from the input file."""
        return [self.process_one(item.prompt) for item in prompts]

    def process_one(self, user_prompt: str) -> FunctionCallResult:
        """Process one user prompt."""
        function = self._select_function(user_prompt)
        parameters = self._extract_arguments(user_prompt, function)
        result = FunctionCallResult(
            prompt=user_prompt,
            name=function.name,
            parameters=parameters,
        )
        return validate_result(result, self.functions)

    def _select_function(self, user_prompt: str) -> FunctionDefinition:
        """Select the function using the LLM and an enum constraint."""
        names = [function.name for function in self.functions]
        prompt = build_function_selection_prompt(user_prompt, self.functions)
        name = self.decoder.generate(
            prompt=prompt,
            constraint=EnumConstraint(names),
            max_new_tokens=64,
        )
        return self._find_function(name.strip())

    def _extract_arguments(
        self,
        user_prompt: str,
        function: FunctionDefinition,
    ) -> dict[str, object]:
        """Extract all arguments as one constrained JSON object."""
        error: str | None = None
        for _ in range(MAX_RETRIES):
            constraint = ArgumentsConstraint(function.parameters)
            prompt = build_arguments_prompt(user_prompt, function, error)
            try:
                text = self.decoder.generate(
                    prompt=prompt,
                    constraint=constraint,
                    max_new_tokens=160,
                )
                parameters = constraint.parse(text)
                if self._arguments_match_schema(parameters, function):
                    return parameters
                error = f"wrong schema: {json.dumps(parameters)}"
            except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
                error = str(exc)
        return self._default_arguments(function)

    def _arguments_match_schema(
        self,
        parameters: dict[str, object],
        function: FunctionDefinition,
    ) -> bool:
        """Return True when parameters exactly match the function schema."""
        if set(parameters.keys()) != set(function.parameters.keys()):
            return False
        for name, spec in function.parameters.items():
            if not matches_type(parameters[name], spec.type):
                return False
        return True

    def _default_arguments(
        self,
        function: FunctionDefinition,
    ) -> dict[str, object]:
        """Return generic schema defaults as a final safety net."""
        return {
            name: schema_default(spec.type)
            for name, spec in function.parameters.items()
        }

    def _find_function(self, name: str) -> FunctionDefinition:
        """Find a function definition by name."""
        for function in self.functions:
            if function.name == name:
                return function
        raise RuntimeError(f"Unknown function selected: {name}")
