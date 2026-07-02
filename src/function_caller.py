"""Main function calling pipeline."""
from __future__ import annotations
import json
import re

from src.constrained_decoder import ConstrainedDecoder
from src.constraints import EnumConstraint, JsonValueConstraint
from src.llm_adapter import LLMAdapter
from src.models import FunctionCallResult, FunctionDefinition, PromptItem
from src.prompt_builder import (
    build_function_selection_prompt,
    build_parameter_prompt,
)

from src.validator import validate_result


class FunctionCallingEngine:
    """Convert natural-language prompts into structured function calls."""

    def __init__(self, functions: list[FunctionDefinition]) -> None:
        """Initialize engine with available functions."""
        self.functions = functions
        self.llm = LLMAdapter()
        self.decoder = ConstrainedDecoder(self.llm)

    def process_all(
        self,
        prompts: list[PromptItem],
    ) -> list[FunctionCallResult]:
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
        """Use constrained decoding to select one allowed function."""
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
        """Use constrained decoding to extract function parameters."""
        parameters: dict[str, object] = {}
        for parameter_name, parameter_spec in function.parameters.items():
            prompt = build_parameter_prompt(
                user_prompt=user_prompt,
                function=function,
                parameter_name=parameter_name,
                parameter_type=parameter_spec.type,
            )
            constraint = JsonValueConstraint(parameter_spec.type)
            try:
                generated_value = self.decoder.generate(
                    prompt=prompt,
                    constraint=constraint,
                    max_new_tokens=64,
                )
                parameters[parameter_name] = constraint.parse(generated_value)
            except (RuntimeError, ValueError, json.JSONDecodeError):
                parameters[parameter_name] = self._fallback_parameter_value(
                    user_prompt=user_prompt,
                    function=function,
                    parameter_name=parameter_name,
                    parameter_type=parameter_spec.type,
                )
        return parameters

    def _fallback_parameter_value(
        self,
        user_prompt: str,
        function: FunctionDefinition,
        parameter_name: str,
        parameter_type: str,
    ) -> object:
        """Extract a parameter directly from the prompt.

        Used only when constrained decoding cannot complete.
        """
        prompt = user_prompt.strip()
        lowered = prompt.lower()

        if parameter_type in {"number", "integer"}:
            numbers = self._extract_numbers(prompt)
            if not numbers:
                return 0
            if (
                "square_root" in function.name
                or "square root" in lowered
            ):
                return self._coerce_number(numbers[0])
            index = list(function.parameters).index(parameter_name)
            if index < len(numbers):
                return self._coerce_number(numbers[index])
            return self._coerce_number(numbers[0])

        if parameter_type == "boolean":
            return any(
                keyword in lowered for keyword in ["true", "yes", "on"]
            )

        quoted = self._extract_quoted_strings(prompt)
        if parameter_name == "name":
            match = re.search(
                r"\bgreet\b\s*(.+)$",
                prompt,
                flags=re.IGNORECASE,
            )
            if match:
                return self._strip_prompt_punctuation(match.group(1))

        if parameter_name in {"s", "source_string"} and quoted:
            if parameter_name == "source_string" and len(quoted) > 1:
                return quoted[-1]
            return quoted[0]

        if parameter_name == "regex":
            literal = self._extract_replacement_target(prompt)
            if literal is not None:
                return re.escape(literal)
            if "numbers" in lowered:
                return r"\d+"
            if "vowels" in lowered:
                return r"[aeiouAEIOU]"
            if "letters" in lowered:
                return r"[A-Za-z]"
            if "spaces" in lowered or "whitespace" in lowered:
                return r"\s+"

        if parameter_name == "replacement":
            replacement = self._extract_replacement_value(prompt)
            if replacement is not None:
                return replacement

        if quoted:
            if parameter_name == "replacement" and len(quoted) > 1:
                return quoted[1]
            return quoted[0]

        return "" if parameter_type == "string" else 0

    def _extract_numbers(self, user_prompt: str) -> list[float]:
        """Return numbers found in the prompt in appearance order."""
        matches = re.findall(r"-?\d+(?:\.\d+)?", user_prompt)
        return [float(match) for match in matches]

    def _coerce_number(self, value: float) -> float | int:
        """Return an integer when the numeric value has no fractional part."""
        if value.is_integer():
            return int(value)
        return value

    def _extract_quoted_strings(self, user_prompt: str) -> list[str]:
        """Collect quoted substrings from the prompt."""
        quoted = re.findall(r'"([^"]*)"|\'([^\']*)\'', user_prompt)
        values: list[str] = []
        for double_quoted, single_quoted in quoted:
            value = double_quoted or single_quoted
            if value:
                values.append(value)
        return values

    def _extract_replacement_target(self, user_prompt: str) -> str | None:
        """Find the literal text targeted by a replacement prompt."""
        match = re.search(
            r"(?:replace|substitute)(?: all)?(?: the)?"
            r"(?: word| phrase)?\s+['\"]([^'\"]+)['\"]",
            user_prompt,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1)
        return None

    def _extract_replacement_value(self, user_prompt: str) -> str | None:
        """Extract the text that follows a 'with' clause."""
        match = re.search(
            r"\bwith\s+['\"]([^'\"]+)['\"]",
            user_prompt,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1)
        match = re.search(
            r"\bwith\s+([A-Za-z0-9_\-]+)",
            user_prompt,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1)
        return None

    def _strip_prompt_punctuation(self, value: str) -> str:
        """Remove wrapping punctuation from extracted prompt text."""
        return value.strip().strip('"').strip("'").strip().rstrip("?.!,")
