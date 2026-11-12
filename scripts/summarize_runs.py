"""Summarize every training run in logs/ into one CSV per experiment (one row per run).

Usage (from the repository root, with the LR Project Kernel / venv active):

    python scripts/summarize_runs.py            # all experiments found in logs/
    python scripts/summarize_runs.py flat       # only logs/flat

Output: results/<prefix>_training_runs.csv, e.g. results/nb01_flat_terrain_training_runs.csv

Metric columns are read from the TensorBoard event files and recomputed at every call.
Manual columns (change_vs_baseline, purpose, kept_for_submission, notes) are never
overwritten: fill them in by hand, they are kept across calls (matched by run_id).
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent.parent
LOG_ROOT = ROOT / "logs"
RESULTS = ROOT / "results"

# Experiment name in logs/ -> CSV file prefix (notebook that produces the runs)
PREFIX = {"flat": "nb01_flat_terrain", "stairs": "nb03_stairs"}

# Number of last logged iterations averaged for the "final" value of each curve
LAST_N = 20

# Column -> regex of the TensorBoard tag (first matching tag is used)
METRICS = {
    "mean_reward": r"Train/mean_reward",
    "mean_episode_length": r"Train/mean_episode_length",
    "rew_track_lin_vel": r"Episode_Reward/track_linear_velocity",
    "rew_track_ang_vel": r"Episode_Reward/track_angular_velocity",
    "err_vel_xy": r"Metrics/.*/error_vel_xy",
    "err_vel_yaw": r"Metrics/.*/error_vel_yaw",
    "rew_action_rate": r"Episode_Reward/action_rate",
    "rew_feet_air_time": r"Episode_Reward/feet_air_time",
    "air_time_mean": r"Metrics/air_time_mean",
    "term_base_contact": r"Episode_Termination/base_contact",
    "term_bad_orientation": r"Episode_Termination/bad_orientation",
    "action_std": r"(Policy|Loss)/.*(std|noise).*",
    "terrain_level": r"Curriculum/terrain_levels.*",
}

AUTO_COLUMNS = [
    "run_id", "experiment", "variant", "seed", "created", "status",
    "last_iteration", "checkpoint", *METRICS.keys(), "eval_files",
]
MANUAL_COLUMNS = ["change_vs_baseline", "purpose", "kept_for_submission", "notes"]
COLUMNS = AUTO_COLUMNS + MANUAL_COLUMNS


def read_yaml(path: Path) -> dict:
    try:
        return yaml.unsafe_load(path.read_text()) or {}
    except Exception:
        return {}


def summarize(run_dir: Path) -> dict:
    parts = run_dir.name.split("_")
    agent = read_yaml(run_dir / "params" / "agent.yaml")
    max_it = agent.get("max_iterations")

    ckpts = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    last_ckpt = int(ckpts[-1].stem.split("_")[-1]) if ckpts else None

    row = {
        "run_id": run_dir.name,
        "experiment": run_dir.parent.name,
        "variant": "_".join(parts[2:]),
        "seed": agent.get("seed", ""),
        "created": "_".join(parts[:2]),
        "checkpoint": ckpts[-1].name if ckpts else "",
        "eval_files": ";".join(p.name for p in sorted((run_dir / "eval").glob("*.npz"))),
    }

    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    last_step = None
    for col, pattern in METRICS.items():
        tag = next((t for t in tags if re.fullmatch(pattern, t)), None)
        if tag is None:
            row[col] = ""
            continue
        events = sorted(acc.Scalars(tag), key=lambda e: e.step)
        values = np.array([e.value for e in events[-LAST_N:]])
        row[col] = f"{values.mean():.4f}"
        last_step = max(last_step or 0, events[-1].step)

    row["last_iteration"] = "" if last_step is None else last_step
    finished = max_it is not None and last_ckpt is not None and last_ckpt >= max_it - 1
    row["status"] = "complete" if finished else "interrupted"
    return row


def load_existing(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {r["run_id"]: r for r in csv.DictReader(f)}


def write_experiment(exp_dir: Path) -> Path:
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{PREFIX.get(exp_dir.name, exp_dir.name)}_training_runs.csv"
    existing = load_existing(out)

    rows = {}
    for run_dir in sorted(p for p in exp_dir.iterdir() if (p / "params").exists()):
        row = summarize(run_dir)
        old = existing.get(row["run_id"], {})
        for col in MANUAL_COLUMNS:
            row[col] = old.get(col, "")
        rows[row["run_id"]] = row

    # Keep rows whose log folder was deleted (history), flagged as such
    for run_id, old in existing.items():
        if run_id not in rows:
            old["status"] = "logs_deleted"
            rows[run_id] = old

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for run_id in sorted(rows):
            writer.writerow({c: rows[run_id].get(c, "") for c in COLUMNS})
    return out


def main() -> None:
    wanted = sys.argv[1:]
    if not LOG_ROOT.exists():
        sys.exit(f"No logs folder at {LOG_ROOT}")
    for exp_dir in sorted(p for p in LOG_ROOT.iterdir() if p.is_dir()):
        if wanted and exp_dir.name not in wanted:
            continue
        out = write_experiment(exp_dir)
        print(f"{exp_dir.name}: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
