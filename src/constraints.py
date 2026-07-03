"""Simple constraints used by the constrained decoder."""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from json import JSONDecoder
from typing import Any

from src.models import ParameterSpec


class Constraint(ABC):
    """Base class for token-level output constraints."""

    @abstractmethod
    def allows(self, text: str) -> bool:
        """Return True while text can still become a valid output."""

    @abstractmethod
    def is_complete(self, text: str) -> bool:
        """Return True when text is a complete valid output."""


class EnumConstraint(Constraint):
    """Constrain output to one allowed string."""

    def __init__(self, values: list[str]) -> None:
        """Store allowed enum values."""
        if not values:
            raise ValueError("EnumConstraint requires at least one value.")
        self.values = values

    def allows(self, text: str) -> bool:
        """Allow whitespace and prefixes of allowed values."""
        clean_text = text.strip()
        return any(value.startswith(clean_text) for value in self.values)

    def is_complete(self, text: str) -> bool:
        """Complete when text equals one allowed value."""
        return text.strip() in self.values


class ArgumentsConstraint(Constraint):
    """Constrain output to an arguments object matching one function schema."""

    def __init__(self, parameters: dict[str, ParameterSpec]) -> None:
        """Store ordered parameter schema."""
        self.parameters = list(parameters.items())
        self.decoder = JSONDecoder()

    def allows(self, text: str) -> bool:
        """Return True while text can still become the arguments object."""
        return self._check(text, complete=False)

    def is_complete(self, text: str) -> bool:
        """Return True when text is a complete arguments object."""
        return self._check(text, complete=True)

    def parse(self, text: str) -> dict[str, object]:
        """Parse the completed arguments object."""
        value: Any = json.loads(text.strip())
        if not isinstance(value, dict):
            raise ValueError("arguments output is not a JSON object")
        return dict(value)

    def _check(self, text: str, complete: bool) -> bool:
        """Validate complete text or a prefix of the required object."""
        clean_text = text.lstrip()
        if clean_text == "":
            return not complete
        return self._read_object(clean_text, complete)

    def _read_object(self, text: str, complete: bool) -> bool:
        """Read the fixed object shape with dynamic parameter names."""
        index = 0
        ok, index = self._read_static(
            text,
            index,
            "{",
            allow_prefix=not complete,
        )
        if not ok:
            return False

        for position, item in enumerate(self.parameters):
            name, spec = item
            if position > 0:
                ok, index = self._read_static(
                    text,
                    index,
                    ",",
                    allow_prefix=not complete,
                )
                if not ok:
                    return False

            ok, index = self._read_static(
                text,
                index,
                json.dumps(name),
                allow_prefix=not complete,
            )
            if not ok:
                return False

            ok, index = self._read_static(
                text,
                index,
                ":",
                allow_prefix=not complete,
            )
            if not ok:
                return False

            ok, index = self._read_value(text, index, spec.type)
            if not ok:
                return False

        ok, index = self._read_static(
            text,
            index,
            "}",
            allow_prefix=not complete,
        )
        if not ok:
            return False

        if complete:
            return text[index:].strip() == ""
        return True

    def _read_static(
        self,
        text: str,
        index: int,
        expected: str,
        allow_prefix: bool,
    ) -> tuple[bool, int]:
        """Read a fixed JSON syntax/key segment."""
        index = self._skip_spaces(text, index)
        remaining = text[index:]
        if remaining.startswith(expected):
            return True, index + len(expected)
        if allow_prefix and expected.startswith(remaining):
            return True, len(text)
        return False, index

    def _read_value(
        self,
        text: str,
        index: int,
        value_type: str,
    ) -> tuple[bool, int]:
        """Read one JSON value or a valid prefix of it."""
        index = self._skip_spaces(text, index)
        remaining = text[index:]
        if remaining == "":
            return True, len(text)

        try:
            value, end = self.decoder.raw_decode(remaining)
        except json.JSONDecodeError:
            return self._is_value_prefix(remaining, value_type), len(text)

        if not self._matches_type(value, value_type):
            return False, index
        return True, index + end

    def _is_value_prefix(self, text: str, value_type: str) -> bool:
        """Return True when text can still become a JSON value."""
        if value_type == "string":
            return self._is_string_prefix(text)
        if value_type in {"number", "integer"}:
            return self._is_number_prefix(text)
        if value_type == "boolean":
            return "true".startswith(text) or "false".startswith(text)
        return False

    def _matches_type(self, value: Any, value_type: str) -> bool:
        """Return True when value matches the schema primitive type."""
        if value_type == "string":
            return isinstance(value, str)
        if value_type == "boolean":
            return isinstance(value, bool)
        if value_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if value_type == "number":
            return (
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            )
        return False

    def _skip_spaces(self, text: str, index: int) -> int:
        """Skip JSON whitespace outside values."""
        while index < len(text) and text[index] in " \t\r\n":
            index += 1
        return index

    def _is_string_prefix(self, text: str) -> bool:
        """Return True when text can become a JSON string."""
        if text == "":
            return True
        if not text.startswith('"'):
            return False

        escaped = False
        for position, char in enumerate(text[1:], start=1):
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                return position == len(text) - 1
        return True

    def _is_number_prefix(self, text: str) -> bool:
        """Return True when text can become a JSON number."""
        if text in {"", "-"}:
            return True

        index = 1 if text.startswith("-") else 0
        if index == len(text):
            return True

        if text[index] == "0":
            index += 1
            if index < len(text) and text[index].isdigit():
                return False
        elif "1" <= text[index] <= "9":
            index += 1
            while index < len(text) and text[index].isdigit():
                index += 1
        else:
            return False

        if index < len(text) and text[index] == ".":
            index += 1
            while index < len(text) and text[index].isdigit():
                index += 1

        if index < len(text) and text[index] in "eE":
            index += 1
            if index < len(text) and text[index] in "+-":
                index += 1
            while index < len(text) and text[index].isdigit():
                index += 1

        return index == len(text)
