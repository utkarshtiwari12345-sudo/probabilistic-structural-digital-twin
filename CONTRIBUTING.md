# Contributing

Contributions should preserve the project's distinction between verified
computational completion and supported scientific claims. A new method is
useful only when it answers a declared research question and is evaluated
against an unchanged comparison protocol.

## Local setup

Use Python 3.12 or 3.13 from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

The exact Stage 26 environment is recorded in `requirements-lock.txt`. It may
be used as a constraints file when an exact reconstruction is required.

## Required checks

Before proposing a change, run:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe run_pipeline.py
```

The last command audits locally generated artifacts; it does not execute the
pipeline unless `--run` is supplied. OpenFAST stages require the pinned v5.0.0
executables and inputs described in `docs/release_guide.md`.

## Research changes

For a new experiment:

1. state the research question and predeclare its decision gates;
2. freeze development and validation data before inspecting validation results;
3. record seeds, units, solver versions, and upstream hashes;
4. add focused tests for reusable numerical logic;
5. report failed hypotheses and limitations alongside successful results; and
6. register the script and required artifacts in the ordered pipeline.

Do not weaken a threshold after seeing holdout data and then report the result
as confirmatory. Such an analysis may be added only when clearly labelled
exploratory and followed by a new independent confirmation dataset.

## Generated data

Do not commit `work/`, solver logs, bulk posterior histories, or regenerated
OpenFAST case outputs. Compact summary tables and research figures are retained
according to `.gitignore`; larger reproducibility bundles belong in a versioned
release archive with checksums.
