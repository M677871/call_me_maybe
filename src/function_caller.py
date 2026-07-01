"""Main function calling pipeline."""
from __future__ import annotations

import math
import re

from src.models import FunctionCallResult, FunctionDefinition, PromptItem
from src.validator import validate_result


class FunctionCallingEngine:
    """Convert natural-language prompts into structured function calls."""

    def __init__(self, functions: list[FunctionDefinition]) -> None:
        """Initialize engine with available functions."""
        self.functions = functions

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
        """Select the best matching function using prompt heuristics."""
        best_function = self.functions[0]
        best_score = -math.inf
        for function in self.functions:
            score = self._score_function_match(user_prompt, function)
            if score > best_score:
                best_score = score
                best_function = function
        return best_function

    def _extract_parameters(
        self,
        user_prompt: str,
        function: FunctionDefinition,
    ) -> dict[str, object]:
        """Extract function parameters with deterministic parsing."""
        parameters: dict[str, object] = {}
        for parameter_name, parameter_spec in function.parameters.items():
            parameters[parameter_name] = self._extract_parameter_value(
                user_prompt=user_prompt,
                function=function,
                parameter_name=parameter_name,
                parameter_type=parameter_spec.type,
            )
        return parameters

    def _score_function_match(
        self,
        user_prompt: str,
        function: FunctionDefinition,
    ) -> float:
        """Assign a rough score to how well a function matches a prompt."""
        prompt = user_prompt.lower()
        name = function.name.lower()
        description = function.description.lower()
        score = 0.0

        prompt_tokens = set(re.findall(r"[a-z0-9]+", prompt))
        name_tokens = set(re.findall(r"[a-z0-9]+", name))
        description_tokens = set(re.findall(r"[a-z0-9]+", description))
        score += 2.0 * len(prompt_tokens & name_tokens)
        score += 0.5 * len(prompt_tokens & description_tokens)

        if any(keyword in prompt for keyword in ["sum", "add", "plus", "total"]):
            if "add" in name or "sum" in description:
                score += 10.0
        if any(keyword in prompt for keyword in ["greet", "hello", "hi"]):
            if "greet" in name or "greeting" in description:
                score += 10.0
        if "reverse" in prompt or "backwards" in prompt:
            if "reverse" in name or "reverse" in description:
                score += 10.0
        if any(keyword in prompt for keyword in ["square root", "sqrt", "root"]):
            if "square_root" in name or "square root" in description:
                score += 10.0
        if any(keyword in prompt for keyword in ["replace", "substitute", "regex", "pattern"]):
            if any(keyword in name for keyword in ["replace", "substitute", "regex"]):
                score += 10.0

        parameter_count = len(function.parameters)
        if parameter_count == 1:
            score += 0.5
        elif parameter_count == 2:
            score += 0.25

        return score

    def _extract_parameter_value(
        self,
        user_prompt: str,
        function: FunctionDefinition,
        parameter_name: str,
        parameter_type: str,
    ) -> object:
        """Extract one parameter value from the user prompt."""
        prompt = user_prompt.strip()
        lowered = prompt.lower()

        if parameter_type in {"number", "integer"}:
            return self._extract_number(user_prompt, parameter_name, function)

        if parameter_type == "boolean":
            return self._extract_boolean(user_prompt)

        if parameter_name == "name":
            match = re.search(r"\bgreet\b\s*(.+)$", prompt, flags=re.IGNORECASE)
            if match:
                return self._clean_string_value(match.group(1))

        if parameter_name in {"s", "source_string"}:
            quoted = self._extract_quoted_strings(prompt)
            if parameter_name == "source_string" and quoted:
                return quoted[-1] if len(quoted) > 1 else quoted[0]
            if quoted:
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
            literal = self._extract_replacement_value(prompt)
            if literal is not None:
                return literal

        quoted = self._extract_quoted_strings(prompt)
        if quoted:
            if parameter_name == "replacement" and len(quoted) >= 2:
                return quoted[1]
            return quoted[0]

        if parameter_type in {"number", "integer"}:
            return 0

        return ""

    def _extract_number(
        self,
        user_prompt: str,
        parameter_name: str,
        function: FunctionDefinition,
    ) -> float | int:
        """Extract the most likely number for a parameter."""
        numbers = self._extract_numbers(user_prompt)
        if not numbers:
            return 0

        function_description = function.description.lower()
        if "square_root" in function.name or "square root" in function_description:
            return self._coerce_number(numbers[0], parameter_name)

        if "add" in function.name or len(numbers) >= len(function.parameters):
            index = list(function.parameters).index(parameter_name)
            if index < len(numbers):
                return self._coerce_number(numbers[index], parameter_name)

        return self._coerce_number(numbers[0], parameter_name)

    def _extract_boolean(self, user_prompt: str) -> bool:
        """Extract a boolean value from common prompt phrasing."""
        prompt = user_prompt.lower()
        if any(keyword in prompt for keyword in ["true", "yes", "enable", "on"]):
            return True
        if any(keyword in prompt for keyword in ["false", "no", "disable", "off"]):
            return False
        return False

    def _extract_numbers(self, user_prompt: str) -> list[float]:
        """Collect numbers from a prompt in appearance order."""
        matches = re.findall(r"-?\d+(?:\.\d+)?", user_prompt)
        numbers: list[float] = []
        for match in matches:
            numbers.append(float(match))
        return numbers

    def _coerce_number(self, value: float, parameter_name: str) -> float | int:
        """Return an int when the value is mathematically integral."""
        if parameter_name.endswith("_count"):
            return int(value)
        if value.is_integer():
            return int(value)
        return value

    def _extract_quoted_strings(self, user_prompt: str) -> list[str]:
        """Return all quoted substrings in the prompt."""
        quoted = re.findall(r'"([^"]*)"|\'([^\']*)\'', user_prompt)
        values: list[str] = []
        for double_quoted, single_quoted in quoted:
            value = double_quoted or single_quoted
            if value:
                values.append(value)
        return values

    def _extract_replacement_target(self, user_prompt: str) -> str | None:
        """Extract a literal target word or phrase for regex replacement."""
        match = re.search(
            r"(?:replace|substitute)(?: all)?(?: the)?(?: word| phrase)?\s+['\"]([^'\"]+)['\"]",
            user_prompt,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1)
        return None

    def _extract_replacement_value(self, user_prompt: str) -> str | None:
        """Extract the replacement text that follows a 'with' phrase."""
        match = re.search(
            r"\bwith\s+(['\"])(.*?)\1",
            user_prompt,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(2)

        match = re.search(
            r"\bwith\s+([A-Za-z0-9_\-]+)",
            user_prompt,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1)
        return None

    def _clean_string_value(self, value: str) -> str:
        """Strip wrapping punctuation from a string value."""
        return value.strip().strip('"').strip("'").strip().rstrip("?.!,")
