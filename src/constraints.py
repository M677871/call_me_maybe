"""Output constraints used during constrained decoding."""
from __future__ import annotations
import json
import re
from abc import ABC, abstractmethod


class Constraint(ABC):
    """Base class for constrained decoding rules."""
    @abstractmethod
    def allows(self, text: str) -> bool:
        """Return True if text is still a valid prefix."""

    @abstractmethod
    def is_complete(self, text: str) -> bool:
        """Return True if text is a complete valid output."""


class EnumConstraint(Constraint):
    """Constrain output to one exact value from a list."""
    def __init__(self, allowed_values: list[str]) -> None:
        """Store allowed enum values."""
        if not allowed_values:
            raise ValueError("EnumConstraint requires at least one value.")
        self.allowed_values = allowed_values

    def allows(self, text: str) -> bool:
        """Allow only prefixes of allowed values."""
        return any(value.startswith(text) for value in self.allowed_values)

    def is_complete(self, text: str) -> bool:
        """Complete when text exactly matches one value."""
        return text in self.allowed_values


class JsonValueConstraint(Constraint):
    """Constrain output to a JSON primitive followed by a newline."""
    _number_prefix = re.compile(
        r"^-?(0|[1-9][0-9]*)?(\.[0-9]*)?([eE][+-]?[0-9]*)?$"
    )
    _number_full = re.compile(
        r"^-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?$"
    )

    def __init__(self, value_type: str) -> None:
        """Create constraint for a JSON value type."""
        self.value_type = value_type

    def allows(self, text: str) -> bool:
        """Return True if text is still a possible JSON value."""
        if "\n" in text:
            if not text.endswith("\n"):
                return False
            if text.count("\n") > 1:
                return False
            return self._is_full_value(text[:-1])
        return self._is_prefix_value(text)

    def is_complete(self, text: str) -> bool:
        """Complete when valid value is followed by newline."""
        return text.endswith("\n") and self._is_full_value(text[:-1])

    def parse(self, text: str) -> object:
        """Parse completed JSON value."""
        clean_text = text.rstrip("\n")
        return json.loads(clean_text)

    def _is_prefix_value(self, text: str) -> bool:
        """Check if text can become a valid value."""
        if self.value_type == "string":
            return self._is_string_prefix(text)
        if self.value_type in {"number", "integer"}:
            return text == "" or bool(self._number_prefix.match(text))
        if self.value_type == "boolean":
            return "true".startswith(text) or "false".startswith(text)
        return False

    def _is_full_value(self, text: str) -> bool:
        """Check if text is a full valid value of required type."""
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return False
        if self.value_type == "string":
            return isinstance(value, str)
        if self.value_type == "number":
            return (isinstance(value, int | float) and
                    not isinstance(value, bool))
        if self.value_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if self.value_type == "boolean":
            return isinstance(value, bool)
        return False

    def _is_string_prefix(self, text: str) -> bool:
        """Check if text can become a JSON string."""
        if text == "":
            return True
        if not text.startswith('"'):
            return False
        escaped = False
        closed = False
        for index, char in enumerate(text[1:], start=1):
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                closed = True
                return index == len(text) - 1
        return not closed
