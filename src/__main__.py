"""Command-line entry point for the project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.function_caller import FunctionCallingEngine
from src.io_utils import (
    ProjectError,
    load_function_definitions,
    load_prompt_items,
    write_results,
)

DEFAULT_FUNCTIONS_PATH = Path("data/input/functions_definition.json")
DEFAULT_INPUT_PATH = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT_PATH = Path("data/output/function_calling_results.json")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert natural language prompts into function calls."
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS_PATH,
        help="Path to the functions definition JSON file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the prompt input JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the output JSON file will be written.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the function-calling pipeline."""
    args = parse_args()
    try:
        functions = load_function_definitions(args.functions_definition)
        prompts = load_prompt_items(args.input)
        engine = FunctionCallingEngine(functions)
        results = engine.process_all(prompts)
        write_results(args.output, results)
        print(f"Successfully wrote results to {args.output}")
        return 0
    except ProjectError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Generation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
