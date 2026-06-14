# Contributing

Thanks for your interest in NeuroEmbed! A few things to know:

## Setup

```bash
git clone https://github.com/YOUR_ORG/neuroembed-api
cd neuroembed-api
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -q
pytest tests/ --cov=neuroembed --cov-report=term-missing
```

We enforce TDD: write the failing test first, watch it fail, then write the
minimal code to pass.

## Lint / types

```bash
ruff check src tests
mypy src/neuroembed
```

## Pull requests

- One logical change per PR
- Tests must pass (35 currently; coverage >90%)
- Link any related issues
- Update README if you change the API surface
- Use [conventional commit messages](https://www.conventionalcommits.org/)

## What we don't merge

- Pure code reformatting without behavior change
- New core dependencies without justification
- LLM-generated code without the author understanding every line
- Anything that weakens the research-use-only safety labelling
