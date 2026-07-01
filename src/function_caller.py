"""Main function calling pipeline."""
from __future__ import annotations
from src.constrained_decoder import ConstrainedDecoder
from src.constraints import EnumConstraint, JsonValueConstraint
from src.llm_adapter import LLMAdapter
from src.models import FunctionCallResult, FunctionDefinition, PromptItem
from src.prompt_builder import (build_function_selection_prompt,
                                build_parameter_prompt)

from src.validator import validate_result


class FunctionCallingEngine:
    """Convert natural-language prompts into structured function calls."""
    def __init__(self, functions: list[FunctionDefinition]) -> None:
        """Initialize engine with available functions."""
        self.functions = functions
        self.llm = LLMAdapter()
        self.decoder = ConstrainedDecoder(self.llm)

    def process_all(self,
                    prompts: list[PromptItem]) -> list[FunctionCallResult]:
        """Process all prompts."""
        results: list[FunctionCallResult] = []
        for item in prompts:
            result = self.process_one(item.prompt)
            results.append(result)
        return results

    def process_one(self, user_prompt: str) -> FunctionCallResult:
        """Process one user prompt."""
        function = self._select_function(user_prompt)
        parameters = self._extract_parameters(user_prompt, function)
        result = FunctionCallResult(
            prompt=user_prompt,
            name=function.name,
            parameters=parameters,
        )
        return validate_result(result, self.functions)

    def _select_function(self, user_prompt: str) -> FunctionDefinition:
        """Use LLM constrained decoding to select one allowed function."""
        function_names = [function.name for function in self.functions]
        prompt = build_function_selection_prompt(
            user_prompt=user_prompt,
            functions=self.functions,
        )
        constraint = EnumConstraint(function_names)
        selected_name = self.decoder.generate(
            prompt=prompt,
            constraint=constraint,
            max_new_tokens=32,
        )
        for function in self.functions:
            if function.name == selected_name:
                return function
        raise RuntimeError(f"Selected unknown function: {selected_name}")

    def _extract_parameters(
        self,
        user_prompt: str,
        function: FunctionDefinition,
    ) -> dict[str, object]:
        """Use LLM constrained decoding to extract function parameters."""
        parameters: dict[str, object] = {}
        for parameter_name, parameter_spec in function.parameters.items():
            prompt = build_parameter_prompt(
                user_prompt=user_prompt,
                function=function,
                parameter_name=parameter_name,
                parameter_type=parameter_spec.type,
            )
            constraint = JsonValueConstraint(parameter_spec.type)
            generated_value = self.decoder.generate(
                prompt=prompt,
                constraint=constraint,
                max_new_tokens=64,
            )
            parameters[parameter_name] = constraint.parse(generated_value)
        return parameters
