"""
Stock LeRobot ACT on the aloha sim transfer-cube task, on Modal.

Trains a fresh ACT policy on `lerobot/aloha_sim_transfer_cube_human` (the World-B
dataset whose released checkpoint we validated at ~80% in gym-aloha), saves the
checkpoint to a Modal volume, and can evaluate it in gym-aloha. Use a small
--steps first to validate the train->save->eval loop, then a full ~80k run.

Usage:
    # short smoke (train + save), then eval the saved checkpoint
    modal run --detach deploy/train_act_aloha_modal.py::train_remote --steps 2000
    modal run deploy/train_act_aloha_modal.py::eval_remote --steps 2000 --episodes 50

    # full run
    modal run --detach deploy/train_act_aloha_modal.py::train_remote --steps 80000

Download checkpoints:
    modal volume get omx-act-aloha-logs /act_aloha/<steps>/checkpoints/last/pretrained_model ./pretrained_model
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import modal


REMOTE_OUTPUTS = Path("/outputs")
MODAL_GPU = "A10G"
DATASET_REPO_ID = "lerobot/aloha_sim_transfer_cube_human"

hf_secret = modal.Secret.from_name("huggingface")

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04", add_python="3.12")
    .apt_install(
        "git", "ffmpeg", "linux-libc-dev", "clang",
        # Headless OpenGL/EGL for MuJoCo rendering during gym-aloha eval.
        "libegl1", "libgl1", "libgles2", "libglib2.0-0", "libosmesa6",
    )
    .pip_install(
        "torch",
        "torchvision",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "lerobot[training]>=0.5.1",
        "gym-aloha>=0.1.4",
        "accelerate",
        "huggingface-hub>=1.0,<2.0",
    )
    # EGL = headless GPU rendering for MuJoCo (no display/X server in container).
    .env({"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"})
)

app = modal.App("omx-act-aloha-training", image=image)
vol = modal.Volume.from_name("omx-act-aloha-logs", create_if_missing=True)


def _run(command: list[str]) -> None:
    print("RUN:", " \\\n    ".join(shlex.quote(p) for p in command), flush=True)
    subprocess.run(command, check=True)


def _run_dir(steps: int) -> Path:
    return REMOTE_OUTPUTS / "act_aloha" / str(steps)


@app.function(
    gpu=MODAL_GPU,
    timeout=86_400,
    memory=65_536,
    volumes={REMOTE_OUTPUTS: vol},
    secrets=[hf_secret],
)
def train_remote(steps: int = 2000, batch_size: int = 8, save_freq: int | None = None) -> str:
    import torch

    print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available in Modal container.")

    output_dir = _run_dir(steps)
    command = [
        "lerobot-train",
        "--policy.type=act",
        "--policy.push_to_hub=false",
        f"--dataset.repo_id={DATASET_REPO_ID}",
        "--policy.device=cuda",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        f"--save_freq={save_freq or steps}",
        "--save_checkpoint=true",
        "--eval_freq=0",
        "--log_freq=200",
        "--wandb.enable=false",
        f"--output_dir={output_dir}",
    ]
    _run(command)
    vol.commit()
    return f"Trained ACT for {steps} steps; checkpoint under {output_dir}/checkpoints."


@app.function(
    gpu=MODAL_GPU,
    timeout=86_400,
    memory=65_536,
    volumes={REMOTE_OUTPUTS: vol},
    secrets=[hf_secret],
)
def eval_remote(steps: int = 2000, episodes: int = 50, checkpoint: str = "last") -> str:
    vol.reload()
    pretrained = _run_dir(steps) / "checkpoints" / checkpoint / "pretrained_model"
    if not pretrained.exists():
        raise FileNotFoundError(f"No checkpoint at {pretrained}. Run train_remote --steps {steps} first.")

    output_dir = _run_dir(steps) / "eval" / checkpoint
    command = [
        "lerobot-eval",
        f"--policy.path={pretrained}",
        "--policy.device=cuda",
        "--env.type=aloha",
        "--env.task=AlohaTransferCube-v0",
        f"--eval.n_episodes={episodes}",
        "--eval.batch_size=10",
        f"--output_dir={output_dir}",
    ]
    _run(command)
    vol.commit()
    return f"Evaluated {pretrained} over {episodes} episodes; results under {output_dir}."


@app.local_entrypoint()
def main(steps: int = 2000, episodes: int = 50, do_eval: bool = True):
    print(f"Spawning ACT aloha training: steps={steps}")
    handle = train_remote.spawn(steps)
    print(f"  train call: {handle.object_id}")
    if do_eval:
        print("After training completes, eval with:")
        print(f"  modal run deploy/train_act_aloha_modal.py::eval_remote --steps {steps} --episodes {episodes}")
    print("Outputs volume: omx-act-aloha-logs")
