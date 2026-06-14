"""
Launch the local act-sim one-batch ACT debug run on a Vast.ai GPU.

Usage:
    export VASTAI_API_KEY="..."
    export HF_TOKEN="..."  # optional for private datasets
    uv run python deploy/train_act_vastai.py --steps 300

For faster, more reliable startup, build/push deploy/vastai.Dockerfile and run
with OMX_VAST_IMAGE set to that image. The launcher still bootstraps missing
pieces over SSH so the generic CUDA image remains usable as a fallback.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from vastai_sdk import VastAI


POLL_INTERVAL = 15
BOOT_TIMEOUT = 1200
SSH_TIMEOUT = 900
SEARCH_LIMIT = 20
SEARCH_ORDER = "dph+"
MAX_LAUNCH_ATTEMPTS = 8
MIN_RELIABILITY = 0.98
REMOTE_WORKSPACE = "/workspace/act-sim"
REMOTE_LOG = "/workspace/act-sim/train.log"
DEFAULT_IMAGE = "nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04"
BAD_STATUSES = {"offline", "exited", "error", "dead"}
BAD_STATUS_MESSAGES = (
    "OCI runtime create failed",
    "failed to create task for container",
    "failed to inject CDI devices",
    "pull access denied",
    "manifest unknown",
    "unauthorized",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch act-sim one-batch ACT debug on Vast.ai.")
    parser.add_argument("--steps", type=int, default=300, help="Number of one-batch smoke training steps.")
    parser.add_argument("--gpu-name", default=os.environ.get("OMX_GPU_NAME", "RTX_3090"))
    parser.add_argument("--min-gpu-ram-mb", type=int, default=20_000)
    parser.add_argument("--disk-gb", type=int, default=80)
    parser.add_argument("--image", default=os.environ.get("OMX_VAST_IMAGE", DEFAULT_IMAGE))
    parser.add_argument("--instance-label", default="act-sim-one-batch-smoke")
    parser.add_argument("--remote-workspace", default=REMOTE_WORKSPACE)
    parser.add_argument("--output-dir", default="outputs/vastai")
    parser.add_argument("--keep-instance", action="store_true", help="Do not destroy the instance after the run.")
    parser.add_argument("--no-download", action="store_true", help="Do not copy remote outputs back after completion.")
    return parser.parse_args()


def coerce_result_data(result):
    if isinstance(result, (dict, list)):
        return result
    if not isinstance(result, str):
        return str(result)
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(result)
        except (ValueError, SyntaxError):
            return result


def parse_instance_id(result) -> int | None:
    result_data = coerce_result_data(result)
    if isinstance(result_data, dict):
        for key in ("new_contract", "instance_id", "contract_id", "id"):
            value = result_data.get(key)
            if isinstance(value, int):
                return value
    return None


def sanitize_error(error: object) -> str:
    api_key = os.environ.get("VASTAI_API_KEY") or os.environ.get("VAST_API_KEY") or ""
    message = re.sub(r"api_key=[^&\s]+", "api_key=***", str(error))
    return message.replace(api_key, "***") if api_key else message


def format_offer(offer: dict) -> str:
    price = offer.get("dph_total", offer.get("dph_base", "?"))
    reliability = offer.get("reliability2", offer.get("reliability", "?"))
    return (
        f"offer={offer.get('id')} host={offer.get('host_id')} machine={offer.get('machine_id')} "
        f"gpu={offer.get('gpu_name')} price=${price}/hr reliability={reliability} "
        f"location={offer.get('geolocation', 'unknown')}"
    )


def search_candidate_offers(vast: VastAI, args: argparse.Namespace) -> list[dict]:
    query = f"gpu_name={args.gpu_name} rentable=True rented=False"
    offers = coerce_result_data(vast.search_offers(query=query, limit=SEARCH_LIMIT, order=SEARCH_ORDER))
    if not isinstance(offers, list):
        raise RuntimeError(f"Unexpected search_offers result: {offers}")

    filtered = []
    for offer in offers:
        if not isinstance(offer, dict):
            continue
        if int(offer.get("num_gpus", 0)) != 1:
            continue
        if float(offer.get("disk_space", 0)) < args.disk_gb:
            continue
        if float(offer.get("gpu_ram", 0)) < args.min_gpu_ram_mb:
            continue
        if offer.get("verification") != "verified":
            continue
        if offer.get("is_vm_deverified"):
            continue
        reliability = float(offer.get("reliability2", offer.get("reliability", 0.0)) or 0.0)
        if reliability < MIN_RELIABILITY:
            continue
        filtered.append(offer)
    return filtered


def setup_onstart_script() -> str:
    return """#!/bin/bash
set -euo pipefail
mkdir -p /workspace
echo "=== ACT-SIM CONTAINER READY ==="
tail -f /dev/null
"""


def launch_offer(vast: VastAI, offer: dict, args: argparse.Namespace):
    print(f"Trying {format_offer(offer)}")
    return vast.create_instance(
        id=offer["id"],
        image=args.image,
        disk=args.disk_gb,
        onstart_cmd=setup_onstart_script(),
        runtype="ssh_proxy",
        label=args.instance_label,
        cancel_unavail=True,
    )


def attach_ssh_key(vast: VastAI, instance_id: int) -> None:
    ssh_key_path = Path.home() / ".ssh" / "id_ed25519.pub"
    if not ssh_key_path.exists():
        ssh_key_path = Path.home() / ".ssh" / "id_rsa.pub"
    if not ssh_key_path.exists():
        raise FileNotFoundError("No SSH public key found at ~/.ssh/id_ed25519.pub or ~/.ssh/id_rsa.pub")
    vast.attach_ssh(instance_id=instance_id, ssh_key=ssh_key_path.read_text().strip())
    print("SSH key attached")


def wait_for_instance_running(vast: VastAI, instance_id: int) -> dict:
    print("Waiting for instance to boot...")
    start = time.time()
    while time.time() - start < BOOT_TIMEOUT:
        data = coerce_result_data(vast.show_instance(id=instance_id))
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected show_instance result: {data}")
        status = data.get("actual_status", data.get("status_msg", "unknown"))
        status_msg = str(data.get("status_msg", ""))
        print(f"  status={status} elapsed={int(time.time() - start)}s")
        if any(message in status_msg for message in BAD_STATUS_MESSAGES):
            raise RuntimeError(f"Instance startup failed: {status_msg}")
        if status == "running":
            if not data.get("ssh_host") or not data.get("ssh_port"):
                raise RuntimeError("Instance is running but does not report ssh_host/ssh_port.")
            return data
        if status in BAD_STATUSES:
            raise RuntimeError(f"Instance entered bad status: {status}")
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Instance not running after {BOOT_TIMEOUT}s")


def ssh_base(instance: dict) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=20",
        "-p",
        str(instance["ssh_port"]),
        f"root@{instance['ssh_host']}",
    ]


def rsync_ssh(instance: dict) -> str:
    return f"ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p {int(instance['ssh_port'])}"


def run_ssh(instance: dict, command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*ssh_base(instance), command],
        cwd=repo_root(),
        text=True,
        check=check,
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def wait_for_ssh(instance: dict) -> None:
    print("Waiting for SSH...")
    start = time.time()
    last_error = None
    while time.time() - start < SSH_TIMEOUT:
        try:
            run_ssh(instance, "echo SSH_OK")
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"SSH not ready after {SSH_TIMEOUT}s: {sanitize_error(last_error)}")


def bootstrap_remote(instance: dict) -> None:
    print("Checking remote Python/CUDA readiness...")
    command = r"""bash -lc 'set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
if ! command -v python3 >/dev/null 2>&1 || ! python3 -m venv --help >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq python3 python3-dev python3-pip python3-venv git ffmpeg linux-libc-dev build-essential clang rsync >/dev/null
fi
if [[ ! -d /opt/venv ]]; then
  python3 -m venv /opt/venv
fi
. /opt/venv/bin/activate
python -m pip install --no-cache-dir --upgrade pip >/dev/null
if ! python - <<PY >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  python -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124
fi
python - <<PY
import torch
print("Torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available.")
print("Detected GPU:", torch.cuda.get_device_name(0))
PY
'"""
    run_ssh(instance, command)


def upload_repo(instance: dict, remote_workspace: str) -> None:
    root = repo_root()
    print(f"Uploading local repo {root} -> {instance['ssh_host']}:{remote_workspace}")
    run_ssh(instance, f"rm -rf {shlex.quote(remote_workspace)} && mkdir -p {shlex.quote(remote_workspace)}")
    subprocess.run(
        [
            "rsync",
            "-az",
            "--delete",
            "--exclude",
            ".git/",
            "--exclude",
            ".venv/",
            "--exclude",
            "__pycache__/",
            "--exclude",
            ".ruff_cache/",
            "--exclude",
            ".pytest_cache/",
            "--exclude",
            "outputs/",
            "-e",
            rsync_ssh(instance),
            f"{root}/",
            f"root@{instance['ssh_host']}:{remote_workspace}/",
        ],
        check=True,
    )


def remote_training_command(args: argparse.Namespace) -> str:
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or ""
    return f"""bash -lc 'set -euo pipefail
cd {shlex.quote(args.remote_workspace)}
. /opt/venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_TOKEN={json.dumps(hf_token)}
export HUGGINGFACE_HUB_TOKEN={json.dumps(hf_token)}
export ACT_DEVICE=cuda
export ACT_SMOKE_STEPS={int(args.steps)}
python -m pip install --no-cache-dir -e .
python -m models.act.one_batch_smoke 2>&1 | tee {shlex.quote(REMOTE_LOG)}
'"""


def run_remote_training(instance: dict, args: argparse.Namespace, local_output_dir: Path) -> None:
    print("Starting remote training...")
    live_log = local_output_dir / "live.log"
    with live_log.open("w") as log_file:
        process = subprocess.Popen(
            [*ssh_base(instance), remote_training_command(args)],
            cwd=repo_root(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[remote] {line}", end="")
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Remote training failed with exit code {return_code}; see {live_log}")


def download_outputs(instance: dict, args: argparse.Namespace, local_output_dir: Path) -> None:
    if args.no_download:
        return
    local_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading remote logs and outputs to {local_output_dir}")
    subprocess.run(
        [
            "rsync",
            "-az",
            "-e",
            rsync_ssh(instance),
            f"root@{instance['ssh_host']}:{REMOTE_LOG}",
            str(local_output_dir / "train.log"),
        ],
        check=False,
    )
    subprocess.run(
        [
            "rsync",
            "-az",
            "-e",
            rsync_ssh(instance),
            f"root@{instance['ssh_host']}:{args.remote_workspace}/outputs/",
            str(local_output_dir / "outputs/"),
        ],
        check=False,
    )


def destroy_instance(vast: VastAI, instance_id: int) -> None:
    print(f"Destroying instance {instance_id}...")
    try:
        vast.destroy_instance(id=instance_id)
    except Exception as exc:
        print(f"WARNING: failed to destroy instance {instance_id}: {sanitize_error(exc)}")
    else:
        print(f"Destroyed instance {instance_id}")


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    args = parse_args()
    api_key = os.environ.get("VASTAI_API_KEY") or os.environ.get("VAST_API_KEY")
    if not api_key:
        print("ERROR: VASTAI_API_KEY is not set.")
        sys.exit(1)

    vast = VastAI(api_key=api_key)
    print(f"Searching for {args.gpu_name} offers...")
    candidates = search_candidate_offers(vast, args)
    if not candidates:
        raise RuntimeError(f"No viable {args.gpu_name} offers found.")

    instance_id = None
    instance = None
    local_output_dir = None
    try:
        last_error = None
        for attempt, offer in enumerate(candidates[:MAX_LAUNCH_ATTEMPTS], start=1):
            try:
                print(f"Launch attempt {attempt}/{min(len(candidates), MAX_LAUNCH_ATTEMPTS)}")
                result = launch_offer(vast, offer, args)
                instance_id = parse_instance_id(result)
                if not instance_id:
                    raise RuntimeError(f"Failed to parse instance ID from: {result}")
                print(f"Instance {instance_id} launched")
                attach_ssh_key(vast, instance_id)
                instance = wait_for_instance_running(vast, instance_id)
                break
            except Exception as exc:
                last_error = exc
                print(f"Launch failed for {format_offer(offer)}: {sanitize_error(exc)}")
                if instance_id is not None:
                    destroy_instance(vast, instance_id)
                    instance_id = None
                    instance = None
        else:
            raise RuntimeError(f"Unable to launch a healthy instance: {sanitize_error(last_error)}")

        assert instance_id is not None and instance is not None
        local_output_dir = repo_root() / args.output_dir / f"instance-{instance_id}"
        local_output_dir.mkdir(parents=True, exist_ok=True)
        wait_for_ssh(instance)
        bootstrap_remote(instance)
        upload_repo(instance, args.remote_workspace)
        run_remote_training(instance, args, local_output_dir)
    finally:
        if instance is not None and local_output_dir is not None:
            download_outputs(instance, args, local_output_dir)
        if instance_id is not None:
            if args.keep_instance:
                print(f"Instance {instance_id} left running because --keep-instance was set.")
            else:
                destroy_instance(vast, instance_id)


if __name__ == "__main__":
    main()
