HF_HOME := $(CURDIR)/.hf_cache
HF_HUB_CACHE := $(HF_HOME)/hub
TRANSFORMERS_CACHE := $(HF_HOME)/transformers

install:
	uv sync
run:
	mkdir -p "$(HF_HOME)"
	HF_HOME="$(HF_HOME)" HF_HUB_CACHE="$(HF_HUB_CACHE)" TRANSFORMERS_CACHE="$(TRANSFORMERS_CACHE)" uv run python3 -m src
debug:
	mkdir -p "$(HF_HOME)"
	HF_HOME="$(HF_HOME)" HF_HUB_CACHE="$(HF_HUB_CACHE)" TRANSFORMERS_CACHE="$(TRANSFORMERS_CACHE)" uv run python3 -m pdb -m src
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf data/output
lint:
	HF_HOME="$(HF_HOME)" HF_HUB_CACHE="$(HF_HUB_CACHE)" TRANSFORMERS_CACHE="$(TRANSFORMERS_CACHE)" uv run flake8
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8
	uv run mypy . --strict
