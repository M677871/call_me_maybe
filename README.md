*This project has been created as part of the 42 curriculum by miissa.*

# Description

This project implements a function-calling pipeline that converts natural-language prompts into structured JSON function calls.
It reads function definitions and user prompts from `data/input/`, uses a constrained decoding loop over the provided `llm_sdk` model, and writes schema-compliant results to `data/output/function_calling_results.json`.

# Instructions

Install dependencies with:

```bash
make install
```

Run the program with:

```bash
make run
```

Run the debug entry point with:

```bash
make debug
```

Run validation checks with:

```bash
make lint
```

Clean generated files with:

```bash
make clean
```

# Algorithm Explanation

The pipeline follows the function-calling flow described in the subject:

1. Load function definitions and user prompts from JSON files.
2. Build a selection prompt that asks the LLM to choose one allowed function name.
3. Use constrained decoding to keep only tokens that preserve a valid function name.
4. For each required argument, build a dedicated extraction prompt.
5. Use a JSON value constraint to force the model output to remain a valid prefix of the expected type.
6. Parse the generated value and validate it against the function schema before writing the final JSON file.

This design avoids relying on free-form model output. The decoder checks token candidates against the current constraint at every step, which keeps the output structurally valid.

# Design Decisions

The implementation keeps the public `llm_sdk` boundary intact and avoids private SDK attributes.
The main decoding logic lives in `src/constrained_decoder.py` and `src/constraints.py`, while `src/function_caller.py` orchestrates selection, extraction, and validation.
Validation is centralized in `src/validator.py` so malformed outputs are rejected before they are written.

The Hugging Face cache is redirected to `.hf_cache/` so the model can download on the `sgoinfre` filesystem instead of the nearly full home partition.

# Performance Analysis

The project is optimized for reliability over raw throughput.
It uses a single small model and token-by-token constrained generation, which is slower than unconstrained prompting but much more stable.
The shipped test set is small, so the end-to-end runtime remains practical while still satisfying the requirement for valid, schema-compliant output.

# Challenges Faced

The main challenge was the limited free space on the home filesystem during model download.
Redirecting the Hugging Face cache to `.hf_cache/` solved that issue without changing the public runtime interface.
Another challenge was keeping the output strictly aligned with the function schema; the validation layer prevents partially correct but invalid results from being written.

# Testing Strategy

The project was validated with:

```bash
make clean && make run
make lint
```

I also checked the generated `data/output/function_calling_results.json` to confirm that function names and parameter types matched the provided definitions.

# Example Usage

```bash
uv run python -m src
uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
```

# Resources

- JSON documentation: https://www.json.org/json-en.html
- Python `json` module: https://docs.python.org/3/library/json.html
- Pydantic documentation: https://docs.pydantic.dev/
- Hugging Face Hub documentation: https://huggingface.co/docs/huggingface_hub/

AI was used to help inspect the runtime failure, identify the cache-space problem, and compare the implementation against the subject. The final code changes, validation, and README content were reviewed and edited in the workspace.
