from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator

JsonType = Literal["string", "number", "integer", "boolean"]


class ParameterSpec(BaseModel):
    """Specification for one function parameter."""
    type: JsonType


class ReturnSpec(BaseModel):
    """Specification for one function return type"""
    type: JsonType


class FunctionDefinition(BaseModel):
    """One available function that the LLM may call."""
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, ParameterSpec]
    returns: ReturnSpec | None = None

    @field_validator("parameters")
    @classmethod
    def validate_parameters(
        cls,
        value: dict[str, ParameterSpec]
    ) -> dict[str, ParameterSpec]:
        """"Ensure parameter name aren't empty"""
        for parameter_name in value:
            if not parameter_name.strip():
                raise ValueError("parameter name cannot be empty")
        return value


class PromptItem(BaseModel):
    """one user prompt from the test input file."""
    prompt: str = Field(min_length=1)


class FunctionCallResult(BaseModel):
    """final output object"""
    prompt: str
    name: str
    parameters: dict[str, Any]
