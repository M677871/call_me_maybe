"""Pydantic models used by the project."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

JsonType = Literal["string", "number", "integer", "boolean"]


class ParameterSpec(BaseModel):
    """Schema for one function parameter."""

    type: JsonType


class ReturnSpec(BaseModel):
    """Schema for a function return value."""

    type: JsonType


class FunctionDefinition(BaseModel):
    """Definition of one callable function."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, ParameterSpec]
    returns: ReturnSpec | None = None

    @field_validator("parameters")
    @classmethod
    def validate_parameter_names(
        cls,
        parameters: dict[str, ParameterSpec],
    ) -> dict[str, ParameterSpec]:
        """Reject empty parameter names."""
        for name in parameters:
            if not name.strip():
                raise ValueError("parameter name cannot be empty")
        return parameters


class PromptItem(BaseModel):
    """One natural-language prompt from the input file."""

    prompt: str = Field(min_length=1)


class FunctionCallResult(BaseModel):
    """One object written to the final output JSON file."""

    prompt: str
    name: str
    parameters: dict[str, Any]
