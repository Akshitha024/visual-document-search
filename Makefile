.PHONY: help install lint typecheck test bench plots clean

help:
	@echo "make install / lint / typecheck / test - quality gates"
	@echo "make bench    - run synthetic ColPali-style benchmark"
	@echo "make plots    - regenerate the 5 chart types"

install: ; uv sync --all-extras
lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
typecheck: ; uv run mypy src
test: ; uv run pytest -m "not slow"
bench: ; uv run cps bench
plots: ; uv run cps plots
clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
