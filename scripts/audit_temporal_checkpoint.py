#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a temporal BC checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-step", type=int, default=None)
    parser.add_argument("--expected-chunk-size", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    path = Path(args.checkpoint)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    errors: list[str] = []

    required = {
        "model_type",
        "model_state",
        "optimizer_state",
        "global_step",
        "samples_seen",
        "train_config",
        "temporal_config",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        errors.append(f"missing checkpoint keys: {missing}")

    model_type = checkpoint.get("model_type")
    if model_type != "temporal_policy_v1":
        errors.append(f"unexpected model_type: {model_type!r}")

    global_step = int(checkpoint.get("global_step", -1))
    if args.expected_step is not None and global_step != args.expected_step:
        errors.append(
            f"global_step={global_step} does not match expected {args.expected_step}"
        )

    temporal_config = dict(checkpoint.get("temporal_config", {}))
    chunk_size = temporal_config.get("chunk_size")
    if (
        args.expected_chunk_size is not None
        and chunk_size != args.expected_chunk_size
    ):
        errors.append(
            f"chunk_size={chunk_size} does not match expected "
            f"{args.expected_chunk_size}"
        )

    model_state = checkpoint.get("model_state", {})
    nonfinite_model_tensors = _nonfinite_tensor_names(model_state)
    if nonfinite_model_tensors:
        errors.append(
            "non-finite model tensors: " + ", ".join(nonfinite_model_tensors[:10])
        )

    optimizer_state = checkpoint.get("optimizer_state", {})
    optimizer_groups = list(optimizer_state.get("param_groups", []))
    if len(optimizer_groups) != 3:
        errors.append(f"expected 3 optimizer groups, found {len(optimizer_groups)}")
    nonfinite_optimizer_tensors = _nonfinite_tensor_names(
        optimizer_state.get("state", {})
    )
    if nonfinite_optimizer_tensors:
        errors.append(
            "non-finite optimizer tensors: "
            + ", ".join(nonfinite_optimizer_tensors[:10])
        )

    tensor_count = sum(1 for value in model_state.values() if torch.is_tensor(value))
    parameter_count = sum(
        int(value.numel()) for value in model_state.values() if torch.is_tensor(value)
    )
    report: dict[str, Any] = {
        "checkpoint": str(path),
        "passed": not errors,
        "errors": errors,
        "sha256": _sha256(path),
        "file_size_bytes": path.stat().st_size,
        "model_type": model_type,
        "global_step": global_step,
        "samples_seen": int(checkpoint.get("samples_seen", -1)),
        "chunk_size": chunk_size,
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "optimizer_group_count": len(optimizer_groups),
        "optimizer_lrs": [group.get("lr") for group in optimizer_groups],
        "amp_enabled": bool(checkpoint.get("train_config", {}).get("use_amp", False)),
        "last_val_metrics": checkpoint.get("last_val_metrics"),
    }

    text = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 1


def _nonfinite_tensor_names(value: Any, prefix: str = "") -> list[str]:
    failures: list[str] = []
    if torch.is_tensor(value):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            failures.append(prefix or "<root>")
        return failures
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            failures.extend(_nonfinite_tensor_names(item, child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]" if prefix else f"[{index}]"
            failures.extend(_nonfinite_tensor_names(item, child))
    return failures


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
