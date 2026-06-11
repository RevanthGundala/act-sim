import argparse
from dataclasses import asdict, replace
from datetime import datetime
from enum import StrEnum
import importlib
import json
import math
import os
from itertools import cycle
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from .act import ACT
from configs.act.base import ACTConfig


class LeRobotFeatureKey(StrEnum):
    OBSERVATION_STATE = "observation.state"
    ACTION = "action"
    ACTION_IS_PAD = "action_is_pad"
    OBSERVATION_IMAGES_PREFIX = "observation.images."

    @classmethod
    def camera_image(cls, camera_name: str) -> str:
        return f"{cls.OBSERVATION_IMAGES_PREFIX}{camera_name}"


def load_experiment(profile: str) -> ACTConfig:
    module = importlib.import_module(f"configs.act.{profile}")
    config = module.config
    if not isinstance(config, ACTConfig):
        raise TypeError(f"configs.act.{profile}.config must be an ACTConfig")
    return config


def resolve_dataset_root(config: ACTConfig) -> Path:
    return (
        Path.home()
        / ".cache"
        / "huggingface"
        / "lerobot"
        / f"{config.dataset_repo_id.replace('/', '__')}__{config.dataset_revision}"
    )


def describe_dataset(config: ACTConfig) -> str:
    if config.dataset_format == "act_hdf5":
        return f"{config.benchmark_task_name}:{config.benchmark_dataset_dir}"
    return f"{config.dataset_repo_id}@{config.dataset_revision}"


def create_run_dir(config: ACTConfig, run_name: str | None = None) -> Path:
    run_id = run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(config.output_root) / config.job_name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_config(run_dir: Path, profile: str, config: ACTConfig) -> None:
    payload = {"profile": profile, **asdict(config)}
    (run_dir / "config.json").write_text(json.dumps(payload, indent=2) + "\n")


def save_checkpoint(
    run_dir: Path,
    step: int,
    model,
    optimizer,
    train_loss: float,
    val_loss: float | None,
    config: ACTConfig,
    norm_stats: dict[str, dict[str, torch.Tensor]],
) -> Path:
    checkpoint = {
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": asdict(config),
        "norm_stats": {key: {name: value.cpu() for name, value in stats.items()} for key, stats in norm_stats.items()},
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    step_path = run_dir / f"checkpoint_step_{step:06d}.pt"
    torch.save(checkpoint, step_path)
    last_path = run_dir / "checkpoint_last.pt"
    if last_path.exists():
        last_path.unlink()
    try:
        os.link(step_path, last_path)
    except OSError:
        torch.save(checkpoint, last_path)
    return step_path


def append_metrics(run_dir: Path, metrics: dict) -> None:
    with (run_dir / "metrics.jsonl").open("a") as file:
        file.write(json.dumps(metrics, sort_keys=True) + "\n")
    log_mlflow_metrics(metrics)


def default_mlflow_base(config: ACTConfig) -> Path:
    output_root = Path(config.output_root)
    return output_root.parent if output_root.name == "act_experiments" else output_root


def default_mlflow_tracking_uri(config: ACTConfig) -> str:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        return tracking_uri
    return f"sqlite:///{(default_mlflow_base(config) / 'mlflow.db').resolve()}"


def default_mlflow_artifact_uri(config: ACTConfig) -> str:
    artifact_uri = os.environ.get("MLFLOW_ARTIFACT_URI")
    if artifact_uri:
        return artifact_uri
    return (default_mlflow_base(config) / "mlflow_artifacts").resolve().as_uri()


def mlflow_param_value(value) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value)


def start_mlflow_run(profile: str, config: ACTConfig, run_dir: Path):
    if os.environ.get("ACT_DISABLE_MLFLOW") == "1":
        print("MLflow disabled by ACT_DISABLE_MLFLOW=1")
        return None
    import mlflow

    tracking_uri = default_mlflow_tracking_uri(config)
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(config.job_name) is None:
        mlflow.create_experiment(config.job_name, artifact_location=default_mlflow_artifact_uri(config))
    mlflow.set_experiment(config.job_name)
    mlflow.start_run(run_name=run_dir.name)
    params = {"profile": profile, **asdict(config)}
    mlflow.log_params({key: mlflow_param_value(value) for key, value in params.items()})
    mlflow.set_tags(
        {
            "profile": profile,
            "run_dir": str(run_dir),
            "dataset": describe_dataset(config),
        }
    )
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow run ID: {mlflow.active_run().info.run_id}")
    return mlflow


def start_debug_mlflow_run(profile: str, config: ACTConfig, output_dir: Path, diagnostic: str, args=None):
    mlflow_module = start_mlflow_run(profile, config, output_dir)
    if mlflow_module is None:
        return None

    debug_params = {"diagnostic": diagnostic}
    if args is not None:
        debug_params.update({f"debug_{key}": value for key, value in vars(args).items()})
    mlflow_module.log_params({key: mlflow_param_value(value) for key, value in debug_params.items()})
    mlflow_module.set_tags(
        {
            "diagnostic": diagnostic,
            "debug_output_dir": str(output_dir),
        }
    )
    return mlflow_module


def finish_debug_mlflow_run(mlflow_module, output_dir: Path) -> None:
    if mlflow_module is None:
        return
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    if metrics_path.exists():
        mlflow_module.log_artifact(str(metrics_path), artifact_path="metrics")
    if summary_path.exists():
        mlflow_module.log_artifact(str(summary_path), artifact_path="summary")
    mlflow_module.end_run()


def log_mlflow_metrics(metrics: dict) -> None:
    try:
        import mlflow
    except ModuleNotFoundError:
        return
    if mlflow.active_run() is None:
        return

    step_value = metrics.get("step", 0)
    step = int(step_value) if isinstance(step_value, int | float) else 0
    split = metrics.get("split")
    prefix = f"{split}/" if isinstance(split, str) and split else ""
    numeric_metrics = {}
    for key, value in metrics.items():
        if key in {"step", "split"}:
            continue
        if isinstance(value, int | float) and math.isfinite(float(value)):
            numeric_metrics[f"{prefix}{key}"] = float(value)
    if numeric_metrics:
        mlflow.log_metrics(numeric_metrics, step=step)


def current_lr(optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def cuda_memory_metrics(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        "cuda_mem_allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
        "cuda_mem_reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
        "cuda_max_mem_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
    }


def format_metrics(metrics: dict) -> str:
    ordered_keys = [
        "step",
        "split",
        "total_loss",
        "action_l1_loss",
        "kl_loss",
        "weighted_kl_loss",
        "lr",
        "grad_norm",
        "step_seconds",
        "examples_per_second",
        "action_valid_fraction",
        "cuda_mem_allocated_gb",
        "cuda_max_mem_allocated_gb",
    ]
    parts = []
    for key in ordered_keys:
        value = metrics.get(key)
        if value is None:
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:.6g}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def main(profile: str = "pour_water", run_name: str | None = None, dry_run: bool = False):
    config = load_experiment(profile)
    if dry_run and config.num_workers > 0:
        config = replace(config, num_workers=0)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    train_loader, val_loader, norm_stats = make_dataloaders(config, device)
    sample_batch = next(iter(train_loader))
    _, qpos, actions, _ = unpack_lerobot_batch(sample_batch, config.camera_names, norm_stats=None)

    model = ACT(
        d_model=config.d_model,
        d_qpos=actions.shape[-1],
        d_z=config.d_z,
        chunk_size=actions.shape[1],
        device=device,
        num_cameras=len(config.camera_names),
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        num_heads=config.num_heads,
        mlp_dim=config.mlp_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    run_dir = create_run_dir(config, run_name)
    save_config(run_dir, profile, config)
    mlflow_module = start_mlflow_run(profile, config, run_dir)
    print(f"Profile: {profile}")
    print(f"Dataset: {describe_dataset(config)}")
    print(f"Cameras: {config.camera_names}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {0 if val_loader is None else len(val_loader)}")
    print(f"Output: {run_dir}")
    try:
        if mlflow_module is not None:
            mlflow_module.log_artifact(str(run_dir / "config.json"), artifact_path="config")
        if dry_run:
            print("Dry run complete; no training performed.")
            return

        train_iter = cycle(train_loader)
        last_train_metrics = {"total_loss": float("nan")}
        last_val_metrics = None
        for step in range(1, config.num_train_steps + 1):
            set_warmup_lr(optimizer, config.learning_rate, step, config.warmup_steps)
            batch = next(train_iter)
            step_start = time.perf_counter()
            last_train_metrics = train_step(
                model,
                batch,
                optimizer,
                device,
                config.camera_names,
                norm_stats,
                config.kl_weight,
                config.grad_clip,
            )
            step_seconds = time.perf_counter() - step_start
            batch_size = batch[LeRobotFeatureKey.OBSERVATION_STATE].shape[0]
            train_log = {
                "step": step,
                "split": "train",
                **last_train_metrics,
                "lr": current_lr(optimizer),
                "step_seconds": step_seconds,
                "examples_per_second": batch_size / max(step_seconds, 1e-9),
                **cuda_memory_metrics(device),
            }

            should_eval = val_loader is not None and (step % config.eval_freq == 0 or step == config.num_train_steps)
            if should_eval:
                last_val_metrics = evaluate(model, val_loader, device, config.camera_names, norm_stats, config.kl_weight)
                val_log = {"step": step, "split": "val", **last_val_metrics}
                append_metrics(run_dir, train_log)
                append_metrics(run_dir, val_log)
                print(format_metrics(train_log), flush=True)
                print(format_metrics(val_log), flush=True)
            elif step % config.log_freq == 0 or step == 1:
                append_metrics(run_dir, train_log)
                print(format_metrics(train_log), flush=True)

            should_save = step % config.save_freq == 0 or step == config.num_train_steps
            if should_save:
                checkpoint_path = save_checkpoint(
                    run_dir,
                    step,
                    model,
                    optimizer,
                    last_train_metrics["total_loss"],
                    None if last_val_metrics is None else last_val_metrics["total_loss"],
                    config,
                    norm_stats,
                )
                if mlflow_module is not None:
                    mlflow_module.log_param(f"checkpoint_step_{step:06d}", str(checkpoint_path))
                print(f"saved_checkpoint step={step} path={run_dir / f'checkpoint_step_{step:06d}.pt'}", flush=True)
    finally:
        if mlflow_module is not None:
            if (run_dir / "metrics.jsonl").exists():
                mlflow_module.log_artifact(str(run_dir / "metrics.jsonl"), artifact_path="metrics")
            mlflow_module.end_run()


def make_dataloaders(config: ACTConfig, device: torch.device):
    if config.dataset_format == "lerobot":
        return make_lerobot_dataloaders(config, resolve_dataset_root(config), device)
    if config.dataset_format == "act_hdf5":
        from benchmarks.act_sim.dataset import make_act_hdf5_dataloaders

        return make_act_hdf5_dataloaders(config, device)
    raise ValueError(f"Unsupported dataset_format={config.dataset_format!r}")


def make_lerobot_dataloaders(config: ACTConfig, root: Path, device: torch.device):
    from huggingface_hub import snapshot_download
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.act.configuration_act import ACTConfig as LeRobotACTConfig

    snapshot_download(
        repo_id=config.dataset_repo_id,
        repo_type="dataset",
        revision=config.dataset_revision,
        local_dir=root,
    )
    act_config = LeRobotACTConfig(
        output_features={},
        chunk_size=config.chunk_size,
        n_action_steps=config.chunk_size,
    )
    dataset_meta = LeRobotDatasetMetadata(config.dataset_repo_id, root=root)
    delta_timestamps = resolve_delta_timestamps(act_config, dataset_meta)
    dataset = LeRobotDataset(
        config.dataset_repo_id,
        root=root,
        delta_timestamps=delta_timestamps,
        tolerance_s=1e4,
    )
    norm_stats = extract_norm_stats(dataset_meta.stats, device)
    train_indices, val_indices = build_episode_split_indices(root, config.train_split, config.seed)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices) if val_indices else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return train_loader, val_loader, norm_stats


def extract_norm_stats(dataset_stats: dict, device: torch.device) -> dict[str, dict[str, torch.Tensor]]:
    norm_stats = {}
    for key in (LeRobotFeatureKey.OBSERVATION_STATE, LeRobotFeatureKey.ACTION):
        stats = dataset_stats[key]
        mean = torch.as_tensor(stats["mean"], dtype=torch.float32, device=device)
        std = torch.as_tensor(stats["std"], dtype=torch.float32, device=device).clamp_min(1e-6)
        norm_stats[key] = {"mean": mean, "std": std}
    return norm_stats


def build_episode_split_indices(root: Path, train_split: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0.0 < train_split <= 1.0:
        raise ValueError(f"train_split must be in (0, 1], got {train_split}")

    import pandas as pd

    parquet_paths = sorted((root / "data").glob("**/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")

    frame_tables = [pd.read_parquet(path, columns=["index", "episode_index"]) for path in parquet_paths]
    frame_table = pd.concat(frame_tables, ignore_index=True)
    episodes = sorted(int(episode) for episode in frame_table["episode_index"].unique())
    rng = random.Random(seed)
    rng.shuffle(episodes)

    if len(episodes) == 1 or train_split == 1.0:
        train_episodes = set(episodes)
        val_episodes = set()
    else:
        train_count = int(len(episodes) * train_split)
        train_count = min(max(train_count, 1), len(episodes) - 1)
        train_episodes = set(episodes[:train_count])
        val_episodes = set(episodes[train_count:])

    train_indices = frame_table[frame_table["episode_index"].isin(train_episodes)]["index"].astype(int).tolist()
    val_indices = frame_table[frame_table["episode_index"].isin(val_episodes)]["index"].astype(int).tolist()
    return train_indices, val_indices


def normalize_tensor(value: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    return (value - stats["mean"].to(value.device)) / stats["std"].to(value.device)


def unpack_lerobot_batch(batch, camera_names, norm_stats=None, normalize_images=True):
    image_tensors = []
    missing_camera_keys = []
    for camera_name in camera_names:
        key = LeRobotFeatureKey.camera_image(camera_name)
        image = batch.get(key)
        if image is None:
            missing_camera_keys.append(key)
            continue
        if image.ndim != 4:
            raise ValueError(f"Expected {key} to have shape [batch, channels, height, width], got {tuple(image.shape)}")
        if image.shape[1] not in (1, 3, 4) and image.shape[-1] in (1, 3, 4):
            image = image.permute(0, 3, 1, 2).contiguous()
        image = image.float()
        if image.max() > 1.0:
            image = image / 255.0
        # ImageNet normalization expected by the pretrained ResNet backbone
        # (matches official ACT's ACTPolicy image preprocessing). Disable when the
        # consumer normalizes internally (e.g. the official ACTPolicy) to avoid
        # double-normalizing.
        if normalize_images and image.shape[1] == 3:
            mean = torch.tensor([0.485, 0.456, 0.406], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
            image = (image - mean) / std
        image_tensors.append(image)

    if missing_camera_keys:
        available_image_keys = sorted(key for key in batch if key.startswith(LeRobotFeatureKey.OBSERVATION_IMAGES_PREFIX))
        raise KeyError(
            f"Batch is missing required camera keys {missing_camera_keys}. "
            f"Available image keys: {available_image_keys}"
        )

    images = torch.stack(image_tensors, dim=1)
    qpos = batch[LeRobotFeatureKey.OBSERVATION_STATE]
    actions = batch[LeRobotFeatureKey.ACTION]
    if actions.ndim != 3:
        raise ValueError(
            "ACT training needs chunked actions with shape [batch, chunk_size, action_dim]. "
            "Build the LeRobotDataset with ACT delta_timestamps."
        )

    action_is_pad = batch.get(LeRobotFeatureKey.ACTION_IS_PAD)
    action_mask = None if action_is_pad is None else ~action_is_pad.bool()
    if norm_stats is not None:
        qpos = normalize_tensor(qpos.float(), norm_stats[LeRobotFeatureKey.OBSERVATION_STATE])
        actions = normalize_tensor(actions.float(), norm_stats[LeRobotFeatureKey.ACTION])
    return images, qpos, actions, action_mask


def set_warmup_lr(optimizer, base_lr: float, step: int, warmup_steps: int) -> None:
    if warmup_steps <= 0:
        lr = base_lr
    else:
        lr = base_lr * min(1.0, step / warmup_steps)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def compute_loss(model, batch, device, camera_names, norm_stats, kl_weight: float):
    images, qpos, actions, action_mask = unpack_lerobot_batch(batch, camera_names, norm_stats=norm_stats)
    images = images.to(device)
    qpos = qpos.to(device)
    actions = actions.to(device)
    if action_mask is not None:
        action_mask = action_mask.to(device)

    pred_actions, mu, log_var = model(images, qpos, actions, action_mask=action_mask)
    action_error = F.l1_loss(pred_actions, actions, reduction="none")
    if action_mask is not None:
        action_error = action_error * action_mask.unsqueeze(-1)
        action_loss = action_error.sum() / (action_mask.sum() * actions.shape[-1]).clamp_min(1.0)
        action_valid_fraction = action_mask.float().mean()
    else:
        action_loss = action_error.mean()
        action_valid_fraction = torch.ones((), device=device)
    # ACT-style KL: sum over latent dims, mean over batch (matches official
    # tonyzhaozh/act kl_divergence). The previous mean-over-all-dims form was
    # ~d_z x weaker, starving the latent and letting the posterior collapse.
    klds = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    kl_loss = klds.sum(dim=1).mean()
    weighted_kl_loss = kl_weight * kl_loss
    total_loss = action_loss + weighted_kl_loss
    metrics = {
        "total_loss": total_loss.detach().item(),
        "action_l1_loss": action_loss.detach().item(),
        "kl_loss": kl_loss.detach().item(),
        "weighted_kl_loss": weighted_kl_loss.detach().item(),
        "action_valid_fraction": action_valid_fraction.detach().item(),
        "pred_action_mean": pred_actions.detach().mean().item(),
        "pred_action_std": pred_actions.detach().std().item(),
        "target_action_mean": actions.detach().mean().item(),
        "target_action_std": actions.detach().std().item(),
        "posterior_mu_abs_mean": mu.detach().abs().mean().item(),
        "posterior_log_var_mean": log_var.detach().mean().item(),
    }
    return total_loss, metrics


def train_step(model, batch, optimizer, device, camera_names, norm_stats, kl_weight: float, grad_clip: float) -> dict[str, float]:
    model.train()
    optimizer.zero_grad()
    loss, metrics = compute_loss(model, batch, device, camera_names, norm_stats, kl_weight)
    loss.backward()
    grad_norm = None
    if grad_clip > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    if grad_norm is not None:
        metrics["grad_norm"] = float(grad_norm.detach().cpu())
    return metrics


def evaluate(model, dataloader, device, camera_names, norm_stats, kl_weight: float) -> dict[str, float]:
    model.eval()
    totals = {}
    total_examples = 0
    with torch.no_grad():
        for batch in dataloader:
            batch_size = batch[LeRobotFeatureKey.OBSERVATION_STATE].shape[0]
            _, metrics = compute_loss(model, batch, device, camera_names, norm_stats, kl_weight)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * batch_size
            total_examples += batch_size
    return {key: value / max(total_examples, 1) for key, value in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the custom OMX ACT model from a named experiment profile.")
    parser.add_argument("--profile", default="pour_water", help="Experiment profile in configs/act/<profile>.py")
    parser.add_argument("--run-name", default=None, help="Optional fixed run directory name under the profile job.")
    parser.add_argument("--dry-run", action="store_true", help="Load config/data/model and write config without training.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(profile=args.profile, run_name=args.run_name, dry_run=args.dry_run)
