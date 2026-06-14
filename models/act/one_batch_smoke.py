from dataclasses import asdict, replace
from datetime import datetime
import json
import os
from pathlib import Path
import time

import torch

from configs.act.base import ACTConfig
from models.act.act import ACTPolicy
from models.act.helper import make_dataloaders
from models.act.train import compute_loss
from utils.config import RECORD_DATASET_REPO_ID


NUM_STEPS = 2_000
DEFAULT_NUM_THREADS = 4


def main():
    base_cfg = ACTConfig(dataset_repo_id=RECORD_DATASET_REPO_ID)
    cfg = replace(
        base_cfg,
        job_name=f"{base_cfg.job_name}_one_batch_smoke",
        num_train_steps=int(os.environ.get("ACT_SMOKE_STEPS", NUM_STEPS)),
        num_workers=0,
        kl_weight=0.0,
    )
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(int(os.environ.get("ACT_SMOKE_NUM_THREADS", DEFAULT_NUM_THREADS)))

    device_name = os.environ.get("ACT_DEVICE", "cpu")
    device = torch.device(device_name)

    model = ACTPolicy(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    train_loader, _ = make_dataloaders(cfg)
    fixed_batch = next(iter(train_loader))

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
        mlflow.set_tags({
            "run_dir": str(run_dir),
            "dataset": f"{cfg.dataset_repo_id}@{cfg.dataset_revision}",
            "diagnostic": "one_batch_smoke",
        })
        mlflow_module = mlflow
        print(f"MLflow tracking URI: {tracking_uri}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")

    print(f"Dataset: {cfg.dataset_repo_id}@{cfg.dataset_revision}")
    print(f"One-batch smoke steps: {cfg.num_train_steps}")
    print(f"Device: {device}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Num workers: {cfg.num_workers}")
    print(f"Torch threads: {torch.get_num_threads()}")
    print(f"Learning rate: {cfg.learning_rate}")
    print(f"KL weight: {cfg.kl_weight}")
    print(f"Output: {run_dir}")

    model.train()
    first_metrics = None
    last_metrics = None

    try:
        if mlflow_module is not None:
            mlflow_module.log_artifact(str(run_dir / "config.json"), artifact_path="config")
        for step in range(1, cfg.num_train_steps + 1):
            step_start = time.perf_counter()
            optim.zero_grad(set_to_none=True)
            loss, action_loss, kl_loss = compute_loss(model, fixed_batch, device, cfg.kl_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()

            metrics = {
                "step": step,
                "split": "train",
                "loss": loss.item(),
                "action_l1": action_loss.item(),
                "kl": kl_loss.item(),
                "weighted_kl": cfg.kl_weight * kl_loss.item(),
                "grad_norm": float(grad_norm),
                "step_seconds": time.perf_counter() - step_start,
            }
            if first_metrics is None:
                first_metrics = metrics
            last_metrics = metrics

            if step == 1 or step % cfg.log_freq == 0 or step == cfg.num_train_steps:
                with (run_dir / "metrics.jsonl").open("a") as file:
                    file.write(json.dumps(metrics, sort_keys=True) + "\n")
                if mlflow_module is not None:
                    mlflow_module.log_metrics(
                        {f"train/{key}": float(value) for key, value in metrics.items() if isinstance(value, int | float)},
                        step=step,
                    )
                print(
                    f"step={step} total_loss={metrics['loss']:.4f} "
                    f"action_l1={metrics['action_l1']:.4f} kl={metrics['kl']:.4f} "
                    f"weighted_kl={metrics['weighted_kl']:.4f}",
                    flush=True,
                )

        if first_metrics is None or last_metrics is None:
            raise RuntimeError("One-batch smoke did not run any steps")

        summary = {
            "step": cfg.num_train_steps,
            "split": "summary",
            "first_loss": first_metrics["loss"],
            "last_loss": last_metrics["loss"],
            "loss_delta": last_metrics["loss"] - first_metrics["loss"],
            "first_action_l1": first_metrics["action_l1"],
            "last_action_l1": last_metrics["action_l1"],
            "action_l1_delta": last_metrics["action_l1"] - first_metrics["action_l1"],
            "loss_went_down": last_metrics["loss"] < first_metrics["loss"],
            "action_l1_went_down": last_metrics["action_l1"] < first_metrics["action_l1"],
        }
        with (run_dir / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(summary, sort_keys=True) + "\n")
        if mlflow_module is not None:
            mlflow_module.log_metrics(
                {f"summary/{key}": float(value) for key, value in summary.items() if isinstance(value, int | float)},
                step=cfg.num_train_steps,
            )
        print(
            "one_batch_summary "
            f"first_total_loss={summary['first_loss']:.4f} "
            f"last_total_loss={summary['last_loss']:.4f} "
            f"loss_delta={summary['loss_delta']:.4f} "
            f"first_action_l1={summary['first_action_l1']:.4f} "
            f"last_action_l1={summary['last_action_l1']:.4f} "
            f"action_l1_delta={summary['action_l1_delta']:.4f}",
            flush=True,
        )
    finally:
        if mlflow_module is not None:
            if (run_dir / "metrics.jsonl").exists():
                mlflow_module.log_artifact(str(run_dir / "metrics.jsonl"), artifact_path="metrics")
            mlflow_module.end_run()


if __name__ == "__main__":
    main()
