# Contributing to Project Gorgon

Thanks for your interest! Here's how to contribute.

## Setup

```bash
python3 -m venv ~/venvs/gorgon
source ~/venvs/gorgon/bin/activate
pip install -r requirements.txt -r requirements-dev.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

## Development

- **Lint**: `ruff check src/ tests/ scripts/`
- **Test**: `pytest tests/ -v --tb=short`
- **Benchmark (dry)**: `python scripts/benchmark_inference.py --dry-run`

## Code Style

- Use `ruff` for linting
- Type hints on all function signatures
- Docstrings on public functions

## Pull Requests

1. Fork the repo
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit a PR with a clear description
