export HF_HOME := /sgoinfre/$(USER)/.cache/huggingface
export HF_HUB_CACHE := /sgoinfre/$(USER)/.cache/huggingface/hub
export HF_XET_CACHE := /sgoinfre/$(USER)/.cache/huggingface/xet
export UV_CACHE_DIR := /sgoinfre/$(USER)/.cache/uv
export XDG_CACHE_HOME := /sgoinfre/$(USER)/.cache

.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python3 -m src

debug:
	uv run python3 -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf data/output

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict
