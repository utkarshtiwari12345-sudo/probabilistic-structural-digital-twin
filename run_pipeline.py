"""Run or audit ordered project stages from the repository root."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from probdt.diagnostics import (
    PipelineAuditError,
    audit_pipeline_artifacts,
    select_pipeline_stages,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = PROJECT_ROOT / "config" / "phase1_pipeline.yaml"


def _load_stages() -> list[dict[str, object]]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    return list(document["pipeline"]["stages"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit project artifacts by default, or execute selected stages with --run."
        )
    )
    parser.add_argument("--run", action="store_true", help="Execute selected stages.")
    parser.add_argument("--list", action="store_true", help="List stages and exit.")
    parser.add_argument("--stage", action="append", default=[], help="Exact stage id.")
    parser.add_argument("--from-stage", help="First stage id in a contiguous range.")
    parser.add_argument("--through-stage", help="Last stage id in a contiguous range.")
    return parser


def main(arguments: list[str] | None = None) -> int:
    parsed = _parser().parse_args(arguments)
    stages = _load_stages()
    if parsed.list:
        for stage in stages:
            print(f"{stage['id']:>3}  {stage['name']}  {stage['command']}")
        return 0
    try:
        selected = select_pipeline_stages(
            stages,
            selected_ids=parsed.stage,
            from_stage=parsed.from_stage,
            through_stage=parsed.through_stage,
        )
    except (PipelineAuditError, ValueError) as error:
        _parser().error(str(error))
    if parsed.run:
        for stage in selected:
            command = [sys.executable, str(PROJECT_ROOT / str(stage["command"]))]
            print(f"RUNNING STAGE {stage['id']}: {stage['name']}", flush=True)
            completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
            if completed.returncode != 0:
                print(f"PIPELINE RESULT: FAIL at stage {stage['id']}")
                return completed.returncode
    audit = audit_pipeline_artifacts(PROJECT_ROOT, selected)
    for row in audit:
        status = "PASS" if row.passed else "FAIL"
        print(
            f"{status} stage {row.stage_id:>3} | {row.relative_path} | "
            f"{row.size_bytes} bytes"
        )
    passed = all(row.passed for row in audit)
    print(f"PIPELINE RESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
