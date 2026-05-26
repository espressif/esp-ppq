"""Run-directory and artifact bookkeeping for AutoQuant."""

import json
import os
import shutil
from typing import Any, Dict, List


def create_run_dir(run_dir: str) -> str:
    """Create ``run_dir`` if needed. Existing contents are preserved."""
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def reset_run_dir(run_dir: str) -> None:
    """Reset index files and remove numbered experiment subdirectories."""
    if os.path.isdir(run_dir):
        for name in os.listdir(run_dir):
            path = os.path.join(run_dir, name)
            if os.path.isdir(path) and name.isdigit():
                shutil.rmtree(path)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump([], f)
    with open(os.path.join(run_dir, "candidates.json"), "w") as f:
        json.dump([], f)


def _make_json_safe(obj: Any) -> Any:
    """Convert ``obj`` into a JSON-serializable structure."""
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    return str(obj)


def update_summary(run_dir: str, record: Dict[str, Any]) -> None:
    """Append ``record`` to ``<run_dir>/summary.json`` (read-modify-write)."""
    summary_path = os.path.join(run_dir, "summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
    else:
        summary = []
    summary.append(_make_json_safe(record))
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)


def load_summary(run_dir: str) -> List[Dict[str, Any]]:
    """Read ``summary.json`` back. Used by ``resume`` mode."""
    path = os.path.join(run_dir, "summary.json")
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return json.load(f)


def save_experiment(
    run_dir: str,
    index: int,
    strategy: Dict[str, dict],
    sampled_param: Dict[str, dict],
    export_path: str,
    runtime_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """Move quantization artifacts into one experiment directory."""
    folder = os.path.join(run_dir, f"{index:04d}")
    os.makedirs(folder, exist_ok=True)

    config = {
        "strategy": {k: v["value"] for k, v in strategy.items()},
        "params": _make_json_safe(sampled_param),
        "runtime": _make_json_safe(runtime_snapshot),
    }
    with open(os.path.join(folder, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    dir_name = os.path.dirname(export_path)
    file_name = os.path.basename(export_path)
    base_name = os.path.splitext(file_name)[0]
    base_path = os.path.join(dir_name, base_name) if dir_name else base_name

    suffixes = [".espdl", ".info", ".json", ".native"]
    moved_files: List[str] = []
    for s in suffixes:
        src = base_path + s
        if not os.path.isfile(src):
            continue
        dst = os.path.join(folder, os.path.basename(src))
        shutil.move(src, dst)
        moved_files.append(os.path.basename(src))

    return {
        "index": index,
        "folder": folder,
        "files": moved_files,
        "strategy": config["strategy"],
        "params": config["params"],
    }
