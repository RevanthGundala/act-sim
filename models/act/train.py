from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from configs.act.base import ACTConfig
from models.act.act import ACTPolicy
from models.act.helper import make_dataloaders
from utils.config import RECORD_DATASET_REPO_ID


def compute_loss(model, batch, device, kl_weight):
    images = batch["images"].to(device)
    qpos = batch["qpos"].to(device)
    actions = batch["actions"].to(device)
    action_mask = batch["action_mask"].to(device)
    act_pred, mu, log_var = model(images, qpos, actions, action_mask)
    action_error = F.l1_loss(act_pred, actions, reduction="none")
    valid_values = (action_mask.sum() * actions.shape[-1]).clamp_min(1.0)
    action_loss = (action_error * action_mask.unsqueeze(-1)).sum() / valid_values
    kl_loss = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp())).sum(dim=1).mean()
    loss = action_loss + kl_weight * kl_loss
    return loss, action_loss, kl_loss


def main():
    cfg = ACTConfig(dataset_repo_id=RECORD_DATASET_REPO_ID)
    device_name = os.environ.get("ACT_DEVICE")
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    model = ACTPolicy(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    train_loader, val_loader = make_dataloaders(cfg)
    run_dir = Path(cfg.output_root) / cfg.job_name / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")

    mlflow_module = None
    if os.environ.get("ACT_DISABLE_MLFLOW") == "1":
        print("MLflow disabled by ACT_DISABLE_MLFLOW=1")
    else:
        import mlflow

        output_root = Path(cfg.output_root)
        mlflow_base = output_root.parent if output_root.name == "act_experiments" else output_root
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(mlflow_base / 'mlflow.db').resolve()}")
        artifact_uri = os.environ.get("MLFLOW_ARTIFACT_URI", (mlflow_base / "mlflow_artifacts").resolve().as_uri())
        mlflow.set_tracking_uri(tracking_uri)
        if mlflow.get_experiment_by_name(cfg.job_name) is None:
            mlflow.create_experiment(cfg.job_name, artifact_location=artifact_uri)
        mlflow.set_experiment(cfg.job_name)
        mlflow.start_run(run_name=run_dir.name)
        mlflow.log_params({
            key: value if isinstance(value, str | int | float | bool) else json.dumps(value)
            for key, value in asdict(cfg).items()
        })
        mlflow.set_tags({"run_dir": str(run_dir), "dataset": f"{cfg.dataset_repo_id}@{cfg.dataset_revision}"})
        mlflow_module = mlflow
        print(f"MLflow tracking URI: {tracking_uri}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")

    print(f"Dataset: {cfg.dataset_repo_id}@{cfg.dataset_revision}")
    print(f"Output: {run_dir}")

    model.train()

    step = 0
    try:
        if mlflow_module is not None:
            mlflow_module.log_artifact(str(run_dir / "config.json"), artifact_path="config")
        while step < cfg.num_train_steps:
            for batch in train_loader:
                step_start = time.perf_counter()
                optim.zero_grad(set_to_none=True)
                loss, action_loss, kl_loss = compute_loss(model, batch, device, cfg.kl_weight)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optim.step()

                train_metrics = {
                    "step": step,
                    "split": "train",
                    "loss": loss.item(),
                    "action_l1": action_loss.item(),
                    "kl": kl_loss.item(),
                    "weighted_kl": cfg.kl_weight * kl_loss.item(),
                    "grad_norm": float(grad_norm),
                    "step_seconds": time.perf_counter() - step_start,
                }
                if step % cfg.log_freq == 0:
                    with (run_dir / "metrics.jsonl").open("a") as file:
                        file.write(json.dumps(train_metrics, sort_keys=True) + "\n")
                    if mlflow_module is not None:
                        mlflow_module.log_metrics(
                            {f"train/{key}": float(value) for key, value in train_metrics.items() if isinstance(value, int | float)},
                            step=step,
                        )
                    print(
                        f"step={step} loss={loss.item():.4f} "
                        f"action_l1={action_loss.item():.4f} kl={kl_loss.item():.4f}",
                        flush=True,
                    )
                if val_loader is not None and step % cfg.eval_freq == 0:
                    model.eval()
                    val_loss = 0.0
                    val_action_loss = 0.0
                    val_kl_loss = 0.0
                    val_batches = 0
                    with torch.no_grad():
                        for val_batch in val_loader:
                            loss, action_loss, kl_loss = compute_loss(model, val_batch, device, cfg.kl_weight)
                            val_loss += loss.item()
                            val_action_loss += action_loss.item()
                            val_kl_loss += kl_loss.item()
                            val_batches += 1
                    val_batches = max(val_batches, 1)
                    val_metrics = {
                        "step": step,
                        "split": "val",
                        "loss": val_loss / val_batches,
                        "action_l1": val_action_loss / val_batches,
                        "kl": val_kl_loss / val_batches,
                        "weighted_kl": cfg.kl_weight * val_kl_loss / val_batches,
                    }
                    with (run_dir / "metrics.jsonl").open("a") as file:
                        file.write(json.dumps(val_metrics, sort_keys=True) + "\n")
                    if mlflow_module is not None:
                        mlflow_module.log_metrics(
                            {f"val/{key}": float(value) for key, value in val_metrics.items() if isinstance(value, int | float)},
                            step=step,
                        )
                    print(f"step={step} val_loss={val_metrics['loss']:.4f}", flush=True)
                    model.train()

                step += 1
                if step >= cfg.num_train_steps:
                    break
    finally:
        if mlflow_module is not None:
            if (run_dir / "metrics.jsonl").exists():
                mlflow_module.log_artifact(str(run_dir / "metrics.jsonl"), artifact_path="metrics")
            mlflow_module.end_run()


if __name__ == "__main__":
    main()
