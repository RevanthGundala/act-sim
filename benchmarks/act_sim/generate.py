from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


TASK_NAME = "sim_transfer_cube_scripted"


def main() -> None:
    args = parse_args()
    act_repo_dir_value = args.act_repo_dir or os.environ.get("ACT_REPO_DIR")
    if not act_repo_dir_value:
        raise SystemExit("Set ACT_REPO_DIR or pass --act-repo-dir pointing to the official tonyzhaozh/act checkout.")
    act_repo_dir = Path(act_repo_dir_value).expanduser()
    record_script = act_repo_dir / "record_sim_episodes.py"
    if not record_script.exists():
        raise SystemExit(f"Missing official ACT record script: {record_script}")

    dataset_dir = Path(args.dataset_dir)
    existing_episodes = sorted(dataset_dir.glob("episode_*.hdf5")) if dataset_dir.exists() else []
    if existing_episodes and not args.overwrite:
        raise SystemExit(
            f"{dataset_dir} already contains {len(existing_episodes)} episode files. "
            "Pass --overwrite to regenerate them."
        )
    dataset_dir.mkdir(parents=True, exist_ok=True)

    command = [
        args.python,
        str(record_script),
        "--task_name",
        TASK_NAME,
        "--dataset_dir",
        str(dataset_dir),
        "--num_episodes",
        str(args.episodes),
    ]
    if args.onscreen_render:
        command.append("--onscreen_render")
    subprocess.run(command, cwd=act_repo_dir, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Generate official ACT {TASK_NAME} HDF5 demonstrations.")
    parser.add_argument(
        "--act-repo-dir",
        default=None,
        help="Path to a checkout of https://github.com/tonyzhaozh/act. Defaults to ACT_REPO_DIR.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="data/benchmarks/act_sim_transfer_cube_scripted",
        help="Output directory for episode_*.hdf5 files.",
    )
    parser.add_argument("--episodes", type=int, default=50, help="Number of scripted episodes to generate.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable from the environment that can run the official ACT simulator.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow regenerating into a non-empty dataset dir.")
    parser.add_argument("--onscreen-render", action="store_true", help="Forward --onscreen_render to official ACT.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
