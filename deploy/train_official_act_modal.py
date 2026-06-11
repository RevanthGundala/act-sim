"""
Train the OFFICIAL tonyzhaozh/act DETRVAE (models/act_official) on Modal.

This mirrors deploy/train_custom_act_modal.py exactly — same image, dataset
staging, and volumes — but launches the genuine ACT DETRVAE harness so you can
A/B it against the custom model on the identical sim dataloader/objective.

Usage:
    modal run --detach deploy/train_official_act_modal.py::train_remote --profile sim_transfer_cube_reference
    modal run deploy/train_official_act_modal.py --profile sim_transfer_cube_reference --preflight-only
    modal run deploy/train_official_act_modal.py --profile sim_transfer_cube_reference \
      --benchmark-dataset-dir data/benchmarks/act_sim_transfer_cube_scripted --upload-benchmark-data
    modal run deploy/train_official_act_modal.py --profile sim_transfer_cube_reference --run-name seed0

Download outputs:
    modal volume get omx-official-act-training-logs / outputs/modal_official_act
"""

from __future__ import annotations

from dataclasses import asdict, replace
import importlib
from pathlib import Path

import modal


REMOTE_WORKSPACE = Path("/workspace")
REMOTE_OUTPUTS = Path("/outputs")
REMOTE_BENCHMARK_DATA = Path("/benchmark-data")
MODAL_GPU = "A10G"
VOLUME_COMMIT_INTERVAL_SECONDS = 300

hf_secret = modal.Secret.from_name("huggingface")

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04", add_python="3.12")
    .apt_install("git", "ffmpeg", "linux-libc-dev", "clang")
    .pip_install(
        "torch",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "huggingface-hub>=1.0,<2.0",
        "lerobot>=0.5.1",
        "av>=15,<16",
        "h5py>=3.15",
        "mlflow>=2.17",
        "pandas",
        "pyarrow",
        "safetensors",
    )
    .add_local_dir("models", remote_path=str(REMOTE_WORKSPACE / "models"))
    .add_local_dir("configs", remote_path=str(REMOTE_WORKSPACE / "configs"))
    .add_local_dir("utils", remote_path=str(REMOTE_WORKSPACE / "utils"))
    .add_local_dir("benchmarks", remote_path=str(REMOTE_WORKSPACE / "benchmarks"))
)

app = modal.App("omx-official-act-training", image=image)
vol = modal.Volume.from_name("omx-official-act-training-logs", create_if_missing=True)
benchmark_vol = modal.Volume.from_name("omx-custom-act-benchmark-data", create_if_missing=True)


def load_profile(profile: str):
    module = importlib.import_module(f"configs.act.{profile}")
    return module.config


def preflight_profile(profile: str) -> None:
    config = load_profile(profile)
    print(f"PROFILE {profile}")
    print(f"  model=official_act_detr_vae")
    print(f"  modal_gpu={MODAL_GPU}")
    print(f"  dataset_format={config.dataset_format}")
    if config.dataset_format == "act_hdf5":
        print(f"  benchmark_task={config.benchmark_task_name}")
        print(f"  benchmark_dataset_dir={config.benchmark_dataset_dir}")
        print(f"  benchmark_num_episodes={config.benchmark_num_episodes}")
        print(f"  benchmark_episode_len={config.benchmark_episode_len}")
    else:
        print(f"  dataset={config.dataset_repo_id}@{config.dataset_revision}")
    print(f"  cameras={config.camera_names}")
    print(f"  batch_size={config.batch_size}")
    print(f"  chunk_size={config.chunk_size}")
    print(f"  d_model={config.d_model}")
    print(f"  encoder_layers={config.num_encoder_layers}")
    print(f"  decoder_layers={config.num_decoder_layers}")
    print(f"  lr={config.learning_rate}")
    print(f"  kl_weight={config.kl_weight}")
    print(f"  steps={config.num_train_steps}")

    if config.dataset_format == "act_hdf5":
        print("  dataset_check=skipped; benchmark HDF5 data is staged in Modal volume")
    else:
        try:
            from huggingface_hub import HfApi
        except ModuleNotFoundError:
            print("  dataset_check=skipped; install/use uv env for local huggingface_hub preflight")
        else:
            HfApi().dataset_info(config.dataset_repo_id, revision=config.dataset_revision)
            print("  dataset_check=ok")


def benchmark_remote_dir(profile: str) -> Path:
    return REMOTE_BENCHMARK_DATA / profile


def upload_benchmark_dataset(profile: str, local_dataset_dir: str) -> None:
    local_path = Path(local_dataset_dir).expanduser()
    if not local_path.is_dir():
        raise FileNotFoundError(f"Benchmark dataset dir not found: {local_path}")
    episode_files = sorted(local_path.glob("episode_*.hdf5"))
    if not episode_files:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in {local_path}")

    remote_dir = f"/{profile}"
    print(f"Uploading {len(episode_files)} benchmark episodes to Modal volume {remote_dir}...")
    with benchmark_vol.batch_upload(force=True) as batch:
        batch.put_directory(local_path, remote_dir)
    print("Benchmark upload complete.")


@app.function(volumes={REMOTE_BENCHMARK_DATA: benchmark_vol})
def list_benchmark_data(profile: str) -> list[str]:
    import os

    benchmark_vol.reload()
    root_dir = benchmark_remote_dir(profile)
    results = []
    if not root_dir.exists():
        return results
    for root, _dirs, files in os.walk(root_dir):
        for filename in files:
            full = os.path.join(root, filename)
            results.append(os.path.relpath(full, root_dir))
    return sorted(results)


def configure_remote_profile(config, profile: str):
    remote_output_root = REMOTE_OUTPUTS / "act_experiments"
    updated = replace(config, output_root=str(remote_output_root))
    if updated.dataset_format == "act_hdf5":
        updated = replace(updated, benchmark_dataset_dir=str(benchmark_remote_dir(profile)))
    return updated


@app.function(
    gpu=MODAL_GPU,
    timeout=86_400,
    memory=65_536,
    volumes={REMOTE_OUTPUTS: vol, REMOTE_BENCHMARK_DATA: benchmark_vol},
    secrets=[hf_secret],
)
def train_remote(profile: str, run_name: str | None = None) -> str:
    import os
    import sys
    import threading

    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ.get("HF_TOKEN", "")

    sys.path.insert(0, str(REMOTE_WORKSPACE))
    os.chdir(REMOTE_WORKSPACE)

    import torch

    print("=== CUDA PREFLIGHT ===")
    print("Torch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Modal container.")
    print("Detected GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))

    config_module = importlib.import_module(f"configs.act.{profile}")
    config_module.config = configure_remote_profile(config_module.config, profile)
    if config_module.config.dataset_format == "act_hdf5":
        benchmark_vol.reload()
        dataset_dir = Path(config_module.config.benchmark_dataset_dir)
        episode_files = sorted(dataset_dir.glob("episode_*.hdf5"))
        expected = config_module.config.benchmark_num_episodes
        print(f"benchmark_dataset_dir={dataset_dir}")
        print(f"benchmark_episode_files={len(episode_files)}")
        if expected is not None and len(episode_files) != expected:
            raise RuntimeError(f"Expected {expected} benchmark episodes in {dataset_dir}, found {len(episode_files)}")
        if not episode_files:
            raise RuntimeError(
                f"No benchmark episodes found in {dataset_dir}. "
                "Run the local entrypoint with --upload-benchmark-data first."
            )
    remote_output_root = Path(config_module.config.output_root)

    from models.act_official.train_official import main as train_main

    print("=== STARTING OFFICIAL ACT (DETRVAE) TRAINING ===")
    print(f"profile={profile}")
    print(f"run_name={run_name}")
    print(f"output_root={remote_output_root}")
    print(f"config={asdict(config_module.config)}")

    stop_commits = threading.Event()

    def commit_loop() -> None:
        while not stop_commits.wait(VOLUME_COMMIT_INTERVAL_SECONDS):
            print("Committing outputs volume for periodic checkpoint download...", flush=True)
            vol.commit()

    commit_thread = threading.Thread(target=commit_loop, daemon=True)
    commit_thread.start()
    try:
        train_main(profile=profile, run_name=run_name, dry_run=False)
    finally:
        stop_commits.set()
        commit_thread.join(timeout=5)
        vol.commit()
    return f"Official ACT training complete for profile={profile}; outputs saved in volume omx-official-act-training-logs."


@app.function(volumes={REMOTE_OUTPUTS: vol})
def list_outputs() -> list[str]:
    import os

    vol.reload()
    results = []
    for root, _dirs, files in os.walk(REMOTE_OUTPUTS):
        for filename in files:
            full = os.path.join(root, filename)
            results.append(os.path.relpath(full, REMOTE_OUTPUTS))
    return sorted(results)


@app.local_entrypoint()
def main(
    profile: str = "sim_transfer_cube_reference",
    run_name: str | None = None,
    preflight_only: bool = False,
    benchmark_dataset_dir: str | None = None,
    upload_benchmark_data: bool = False,
    list_benchmark_only: bool = False,
):
    preflight_profile(profile)
    config = load_profile(profile)
    if upload_benchmark_data:
        if config.dataset_format != "act_hdf5":
            raise ValueError("--upload-benchmark-data is only valid for dataset_format='act_hdf5' profiles")
        upload_benchmark_dataset(profile, benchmark_dataset_dir or config.benchmark_dataset_dir or "")
    if list_benchmark_only:
        files = list_benchmark_data.remote(profile)
        print(f"Benchmark files in Modal volume for {profile}:")
        for filename in files[:20]:
            print(f"  {filename}")
        if len(files) > 20:
            print(f"  ... {len(files) - 20} more")
        return
    if preflight_only:
        print("Preflight only; not launching Modal GPU job.")
        return
    if upload_benchmark_data:
        print("Benchmark upload complete; not launching Modal GPU job.")
        print("Start training with:")
        print(f"  modal run deploy/train_official_act_modal.py --profile {profile} --run-name {run_name or 'seed0'}")
        return

    function_call = train_remote.spawn(profile, run_name)
    print(f"Spawned Modal official ACT training call: {function_call.object_id}")
    print("Outputs volume: omx-official-act-training-logs")
    print("Download later with:")
    print("  modal volume get omx-official-act-training-logs / outputs/modal_official_act")
