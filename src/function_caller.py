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
            results.append(self.process_one(item.prompt))
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
        selected_name = self.decoder.generate(
            prompt=prompt,
            constraint=EnumConstraint(function_names),
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
                value = constraint.parse(generated_value)

                if (
                    parameter_spec.type == "string"
                    and isinstance(value, str)
                    and value.strip() == ""
                ):
                    raise ValueError("empty string argument")

                parameters[parameter_name] = value
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
        """Return a simple schema-valid value when decoding fails."""
        if parameter_type in {"number", "integer"}:
            return self._number_fallback(user_prompt, function, parameter_name)
        if parameter_type == "boolean":
            return self._boolean_fallback(user_prompt)
        if parameter_type == "string":
            return self._string_fallback(user_prompt, function, parameter_name)
        return "unknown"

    def _number_fallback(
        self,
        user_prompt: str,
        function: FunctionDefinition,
        parameter_name: str,
    ) -> int | float:
        """Pick a number from the prompt using numeric parameter order."""
        matches = re.findall(r"-?\d+(?:\.\d+)?", user_prompt)
        numbers = [float(item) for item in matches]
        if not numbers:
            return 0.0

        names = [
            name
            for name, spec in function.parameters.items()
            if spec.type in {"number", "integer"}
        ]
        index = names.index(parameter_name)
        value = numbers[index] if index < len(numbers) else numbers[0]
        return int(value) if value.is_integer() else value

    def _boolean_fallback(self, user_prompt: str) -> bool:
        """Pick a boolean from common words in the prompt."""
        lowered = user_prompt.lower()
        if any(word in lowered for word in ["false", "no", "off"]):
            return False
        return any(word in lowered for word in ["true", "yes", "on"])

    def _string_fallback(
        self,
        user_prompt: str,
        function: FunctionDefinition,
        parameter_name: str,
    ) -> str:
        """Pick a string from quotes, parameter meaning, or final word."""
        quoted = self._quoted_strings(user_prompt)
        lowered_name = parameter_name.lower()

        if "regex" in lowered_name or "pattern" in lowered_name:
            return self._regex_fallback(user_prompt, quoted)
        if "replacement" in lowered_name:
            replacement = self._after_with(user_prompt)
            if replacement:
                return replacement
        if "source" in lowered_name and quoted:
            return quoted[-1] if len(quoted) > 1 else quoted[0]

        names = [
            name
            for name, spec in function.parameters.items()
            if spec.type == "string"
        ]
        index = names.index(parameter_name)
        if index < len(quoted):
            return quoted[index]
        if quoted:
            return quoted[0]
        return self._last_word(user_prompt)

    def _regex_fallback(self, user_prompt: str, quoted: list[str]) -> str:
        lowered = user_prompt.lower()

        if "number" in lowered or "digit" in lowered:
            return r"\d+"
        if "vowel" in lowered:
            return r"[aeiouAEIOU]"
        if "letter" in lowered:
            return r"[A-Za-z]"
        if "space" in lowered or "whitespace" in lowered:
            return r"\s+"

        if quoted:
            text = quoted[0]
            if re.search(r"[.^$*+?{}\\[\\]|()]", text):
                return re.escape(text)
            return text

        return r".*"

    def _quoted_strings(self, user_prompt: str) -> list[str]:
        """Return text found inside single or double quotes."""
        matches = re.findall(r'"([^"]*)"|\'([^\']*)\'', user_prompt)
        return [double or single for double, single in matches]

    def _after_with(self, user_prompt: str) -> str:
        match = re.search(r"\bwith\s+['\"]([^'\"]+)['\"]", user_prompt, flags=re.IGNORECASE)
        if match:
            return match.group(1)

        match = re.search(r"\bwith\s+(.+)$", user_prompt, flags=re.IGNORECASE)
        if match:
            return self._clean_text(match.group(1))

        return ""

    def _last_word(self, user_prompt: str) -> str:
        words = re.findall(r"[A-Za-z0-9_\-]+", user_prompt)
        return words[-1] if words else "unknown"

    def _clean_text(self, value: str) -> str:
        """Remove wrapping quotes and punctuation."""
        return value.strip().strip('"').strip("'").strip().rstrip("?.!,")
