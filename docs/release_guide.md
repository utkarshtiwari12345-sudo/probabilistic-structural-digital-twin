# Release and reproducibility guide

## What the GitHub repository contains

The compact repository is designed to let a reviewer inspect the methods,
tests, equations, principal figures, concise result tables, and official pinned
OpenFAST reference case without downloading every generated trajectory.

Tracked release evidence includes:

- all configuration, source, scripts, tests, and documentation;
- all research figures;
- compact summaries, validation tables, hypothesis checks, calibration tables,
  detection events, and provenance manifests;
- the official Stage 17 reference input/output and curated Stage 18 deck; and
- the Stage 26 release manifest and SHA-256 checksum list.

Bulk processed time histories, posterior-state matrices, solver logs, temporary
decks, generated TurbSim fields, and Stage 19/23/25 OpenFAST outputs are excluded
from Git. They are deterministic local products whose paths and minimum sizes
are declared in `config/phase1_pipeline.yaml`.

## Reproduce the Python environment

Python 3.12 or 3.13 is supported. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
```

For the exact Stage 26 dependency versions, add
`-c requirements-lock.txt` to the installation command.

## Reproduce the research pipeline

List or audit stages with:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --list
.\.venv\Scripts\python.exe run_pipeline.py
```

Stages 1--16 use the Python environment alone. Stage 17 ingests the included
official OpenFAST regression output. New simulations in Stages 18--25 require
the official OpenFAST v5.0.0 and TurbSim executables at the paths declared by
their configuration files. Executable SHA-256 hashes are verified before use.
The complete workflow can be expensive, so execute a bounded range when needed:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --from-stage 1a --through-stage 16 --run
.\.venv\Scripts\python.exe run_pipeline.py --stage 25 --run
```

The default command audits artifacts without rerunning experiments. A fresh
clone is expected to pass unit tests immediately; a complete artifact audit
passes after the declared pipeline products have been generated or restored
from a checksum-verified research archive.

## Interpretation boundary

A stage `RESULT: PASS` means its implementation, provenance, dimensions, and
predeclared computational checks completed. Scientific hypotheses are reported
separately and may fail. In particular, Stage 25 confirms the distributed-
moment model only under a matched-excitation oracle; the commissioning-reference
feature is rejected as an operational detector.

## Versioned release checklist

1. Run `scripts/verify_release_readiness.py` and inspect every Stage 26 table.
2. Review the candidate paths in `stage26_release_manifest.csv`.
3. Confirm that `git status --short` contains no secrets, private data, `work/`,
   solver logs, or local virtual environments.
4. Confirm that `CITATION.cff`, `pyproject.toml`, and the README name the final
   repository URL and version.
5. Create and inspect the release commit before pushing it.
6. Push a version tag only after the remote CI workflow passes.
7. Create GitHub release notes that state both the main findings and the
   operational limitations.
8. If a DOI is created through Zenodo or another archive, add it to
   `CITATION.cff` and cite the archived release rather than an evolving branch.

Stage 26 validates release content but does not silently mutate external state.
Repository creation, pushes, releases, and DOI deposits are explicit owner-
authorised publication actions.
