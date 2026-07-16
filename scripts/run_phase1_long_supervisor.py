#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "phase1"
EVALUATIONS = ARTIFACTS / "evaluations"
LOG_PATH = ARTIFACTS / "phase1_long_supervisor.log"
STATUS_PATH = ARTIFACTS / "phase1_long_supervisor_status.json"
FINAL_PATH = ARTIFACTS / "phase1_long_supervisor_complete.json"
PYTHON = Path(sys.executable)


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    EVALUATIONS.mkdir(parents=True, exist_ok=True)
    _write_status("waiting_for_20k")
    try:
        checkpoint_20k = ARTIFACTS / "temporal_k8_v2_long20000.pt"
        _wait_for_checkpoint(checkpoint_20k, expected_step=20_000)

        trend_path = EVALUATIONS / "k8_long20000_validation_trend.json"
        _run([
            PYTHON,
            ROOT / "scripts" / "analyze_phase1_validation.py",
            "--metrics", ARTIFACTS / "temporal_k8_v2_long20000.metrics.jsonl",
            "--output", trend_path,
        ], stage="analyze_20k_validation")
        trend = json.loads(trend_path.read_text(encoding="utf-8"))
        continue_to_50k = bool(trend["decision"]["continue_to_50k"])

        if continue_to_50k:
            checkpoint_50k = ARTIFACTS / "temporal_k8_v2_long50000.pt"
            if not _checkpoint_matches(checkpoint_50k, expected_step=50_000):
                _run([
                    "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                    "-File", ROOT / "scripts" / "run_phase1_k8_long50k.ps1",
                ], stage="train_20k_to_50k")
            _wait_for_checkpoint(checkpoint_50k, expected_step=50_000)
            terminal_checkpoint = checkpoint_50k
        else:
            terminal_checkpoint = checkpoint_20k

        terminal_step = _checkpoint_step(terminal_checkpoint)
        final_checkpoint, fixed_validation_path, selection_path = (
            _select_best_checkpoint(terminal_checkpoint)
        )
        final_step = _checkpoint_step(final_checkpoint)
        final_tag = f"k8_best_step{final_step}"

        audit_path = EVALUATIONS / f"{final_tag}_checkpoint_audit.json"
        _run([
            PYTHON,
            ROOT / "scripts" / "audit_temporal_checkpoint.py",
            "--checkpoint", final_checkpoint,
            "--expected-step", str(final_step),
            "--expected-chunk-size", "8",
            "--output", audit_path,
        ], stage="checkpoint_audit")

        terminal_audit_path = EVALUATIONS / f"k8_terminal_step{terminal_step}_audit.json"
        _run([
            PYTHON,
            ROOT / "scripts" / "audit_temporal_checkpoint.py",
            "--checkpoint", terminal_checkpoint,
            "--expected-step", str(terminal_step),
            "--expected-chunk-size", "8",
            "--output", terminal_audit_path,
        ], stage="terminal_checkpoint_audit")

        latency_path = EVALUATIONS / f"{final_tag}_latency_cuda.json"
        _run([
            PYTHON,
            ROOT / "scripts" / "benchmark_temporal_latency.py",
            "--checkpoint", final_checkpoint,
            "--output", latency_path,
            "--device", "cuda",
            "--warmup", "100",
            "--samples", "1000",
        ], stage="latency_acceptance")

        matches_dir = EVALUATIONS / f"{final_tag}_match_acceptance"
        _run([
            "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", ROOT / "scripts" / "run_phase1_k8_final_matches.ps1",
            "-Checkpoint", final_checkpoint,
            "-SeedStart", "4000",
            "-SeedCount", "50",
            "-OutputDir", matches_dir,
        ], stage="match_acceptance")

        test_log = EVALUATIONS / f"{final_tag}_tests.log"
        _run([
            PYTHON, "-m", "pytest", "-q",
            "tests/test_training_temporal.py",
            "tests/test_training_bc.py",
            "tests/test_matches.py",
        ], stage="phase1_regression_tests", extra_log=test_log)

        result = {
            "status": "complete",
            "completed_at": _now(),
            "continued_to_50k": continue_to_50k,
            "terminal_checkpoint": str(terminal_checkpoint),
            "terminal_step": terminal_step,
            "selected_checkpoint": str(final_checkpoint),
            "selected_step": final_step,
            "artifacts": {
                "validation_trend": str(trend_path),
                "checkpoint_selection": str(selection_path),
                "fixed_validation": str(fixed_validation_path),
                "checkpoint_audit": str(audit_path),
                "terminal_checkpoint_audit": str(terminal_audit_path),
                "latency": str(latency_path),
                "matches": str(matches_dir / "summary.json"),
                "tests": str(test_log),
            },
        }
        FINAL_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        _write_status("complete", **result)
        _log("Phase 1 long-training supervision completed successfully.")
        return 0
    except BaseException as exc:
        failure = {
            "status": "failed",
            "failed_at": _now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_status("failed", **failure)
        _log(f"FAILED: {type(exc).__name__}: {exc}")
    return 1


def _select_best_checkpoint(
    terminal_checkpoint: Path,
) -> tuple[Path, Path, Path]:
    """Re-evaluate a validation-ranked shortlist and preserve the best model."""
    candidates = [ARTIFACTS / "temporal_k8_v2_screen1000.pt"]
    candidates.extend(sorted(ARTIFACTS.glob("temporal_k8_v2_long20000_step*.pt")))
    candidates.append(ARTIFACTS / "temporal_k8_v2_long20000.pt")
    if _checkpoint_step(terminal_checkpoint, default=0) >= 50_000:
        candidates.extend(sorted(ARTIFACTS.glob("temporal_k8_v2_long50000_step*.pt")))
        candidates.append(ARTIFACTS / "temporal_k8_v2_long50000.pt")
    candidates = list(dict.fromkeys(path for path in candidates if path.is_file()))

    metadata = [_checkpoint_metadata(path) for path in candidates]
    ranked = sorted(
        (item for item in metadata if item["recorded_val_loss"] is not None),
        key=lambda item: float(item["recorded_val_loss"]),
    )
    shortlist_paths = [Path(item["checkpoint"]) for item in ranked[:4]]
    shortlist_paths.extend([
        ARTIFACTS / "temporal_k8_v2_screen1000.pt",
        terminal_checkpoint,
    ])
    shortlist = list(dict.fromkeys(path for path in shortlist_paths if path.is_file()))

    evaluations = []
    selection_dir = EVALUATIONS / "k8_checkpoint_selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    for candidate in shortlist:
        output = selection_dir / f"{candidate.stem}_fixed_validation.json"
        _run([
            PYTHON,
            ROOT / "scripts" / "evaluate_temporal_checkpoint.py",
            "--checkpoint", candidate,
            "--output", output,
            "--split", "val",
            "--limit-replays", "200",
            "--batches", "200",
            "--batch-size", "2",
            "--device", "cuda",
            "--amp",
        ], stage=f"select_validate_{candidate.stem}")
        evaluation = json.loads(output.read_text(encoding="utf-8"))
        evaluations.append({
            "checkpoint": str(candidate),
            "global_step": _checkpoint_step(candidate),
            "loss": float(evaluation["metrics"]["loss"]),
            "evaluation": str(output),
        })

    selected = min(evaluations, key=lambda item: item["loss"])
    selected_source = Path(selected["checkpoint"])
    selected_path = ARTIFACTS / "temporal_k8_phase1_best.pt"
    shutil.copy2(selected_source, selected_path)
    fixed_validation_path = (
        EVALUATIONS / f"k8_best_step{selected['global_step']}_fixed_validation.json"
    )
    shutil.copy2(Path(selected["evaluation"]), fixed_validation_path)
    selection_path = EVALUATIONS / "k8_phase1_checkpoint_selection.json"
    report = {
        "terminal_checkpoint": str(terminal_checkpoint),
        "candidate_metadata": metadata,
        "shortlist_evaluations": evaluations,
        "selected_source": str(selected_source),
        "selected_checkpoint": str(selected_path),
        "selected_step": selected["global_step"],
        "selected_loss": selected["loss"],
    }
    selection_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _log(
        f"Selected {selected_source.name} at step {selected['global_step']} "
        f"with fixed-validation loss={selected['loss']:.6f}."
    )
    return selected_path, fixed_validation_path, selection_path


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    last_val = checkpoint.get("last_val_metrics") or {}
    raw_loss = last_val.get("loss")
    return {
        "checkpoint": str(path),
        "global_step": int(checkpoint.get("global_step", -1)),
        "recorded_val_loss": None if raw_loss is None else float(raw_loss),
    }


def _wait_for_checkpoint(path: Path, *, expected_step: int) -> None:
    polls = 0
    while not _checkpoint_matches(path, expected_step=expected_step):
        if polls % 20 == 0:
            _log(f"Waiting for {path.name} at global_step={expected_step}.")
            _write_status(
                f"waiting_for_{expected_step}",
                checkpoint=str(path),
                observed_step=_checkpoint_step(path, default=None),
            )
        polls += 1
        time.sleep(30)
    _log(f"Observed complete checkpoint {path.name} at step {expected_step}.")


def _checkpoint_matches(path: Path, *, expected_step: int) -> bool:
    return _checkpoint_step(path, default=-1) == expected_step


def _checkpoint_step(path: Path, default: Any = -1) -> Any:
    if not path.is_file():
        return default
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        return int(checkpoint.get("global_step", -1))
    except (OSError, RuntimeError, EOFError, ValueError):
        return default


def _run(command: list[Any], *, stage: str, extra_log: Path | None = None) -> None:
    normalized = [str(part) for part in command]
    _write_status(stage, command=normalized)
    _log(f"Starting stage {stage}.")
    with LOG_PATH.open("a", encoding="utf-8") as shared_log:
        if extra_log is None:
            completed = subprocess.run(
                normalized,
                cwd=ROOT,
                stdout=shared_log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        else:
            extra_log.parent.mkdir(parents=True, exist_ok=True)
            with extra_log.open("w", encoding="utf-8") as stage_log:
                completed = subprocess.run(
                    normalized,
                    cwd=ROOT,
                    stdout=stage_log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
    if completed.returncode != 0:
        raise RuntimeError(
            f"stage {stage} failed with exit code {completed.returncode}"
        )
    _log(f"Completed stage {stage}.")


def _write_status(stage: str, **details: Any) -> None:
    payload = {"stage": stage, "updated_at": _now(), **details}
    STATUS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _log(message: str) -> None:
    line = f"[{_now()}] {message}"
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
