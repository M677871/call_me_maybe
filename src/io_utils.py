"""JSON input and output helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from src.models import FunctionCallResult, FunctionDefinition, PromptItem

T = TypeVar("T", bound=BaseModel)


class ProjectError(Exception):
    """Clear error shown to the user instead of a traceback."""


def read_json_file(path: Path) -> Any:
    """Read a JSON file and convert common failures to ProjectError."""
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as exc:
        raise ProjectError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ProjectError(f"Invalid JSON in file: {path}") from exc
    except OSError as exc:
        raise ProjectError(f"Cannot read file: {path}") from exc


def load_model_list(path: Path, model_type: type[T]) -> list[T]:
    """Load a JSON array and validate each item with Pydantic."""
    raw_data = read_json_file(path)
    if not isinstance(raw_data, list):
        raise ProjectError(f"Expected a JSON array in file: {path}")
    try:
        return [model_type.model_validate(item) for item in raw_data]
    except ValidationError as exc:
        raise ProjectError(f"Invalid data in file: {path}\n{exc}") from exc


def load_function_definitions(path: Path) -> list[FunctionDefinition]:
    """Load available function definitions."""
    return load_model_list(path, FunctionDefinition)


def load_prompt_items(path: Path) -> list[PromptItem]:
    """Load the prompts to process."""
    return load_model_list(path, PromptItem)


def write_results(path: Path, results: list[FunctionCallResult]) -> None:
    """Write the final JSON output file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [result.model_dump(mode="json") for result in results]
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise ProjectError(f"Cannot write output file: {path}") from exc
