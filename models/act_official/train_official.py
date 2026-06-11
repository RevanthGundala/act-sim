"""train_official.py — run the genuine tonyzhaozh/act DETRVAE on the EXACT
dataloader / normalization / objective used by the custom `models/act` model.

This is the decisive A/B: same data path (reused verbatim from
`models.act.train`), same z-scored qpos/actions, same [0,1] images, same L1+KL
objective and hyperparameters — only the model is swapped for official ACT.

Usage (sim benchmark, the current setting):
    python -m models.act_official.train_official --profile sim_transfer_cube_reference
    python -m models.act_official.train_official --profile sim_transfer_cube_smoke

Watch `action_l1_loss` (== normalized-action L1). If official ACT breaks the
custom model's ~0.08 floor, the bug is in the custom transformer/latent/decoder.
If it also floors ~0.08-0.12, the issue is the data/task/objective, not the
architecture code.
"""
import argparse
import time
from dataclasses import replace
from itertools import cycle

import torch

# Reuse the EXACT data path + helpers from the custom trainer. Nothing about how
# data is loaded, split, or normalized changes between the two experiments.
from models.act.train import (
    LeRobotFeatureKey,
    append_metrics,
    create_run_dir,
    cuda_memory_metrics,
    current_lr,
    describe_dataset,
    format_metrics,
    load_experiment,
    make_dataloaders,
    save_checkpoint,
    save_config,
    set_warmup_lr,
    start_mlflow_run,
    unpack_lerobot_batch,
)
from models.act_official.policy import ACTPolicy


def build_policy_args(config, state_dim: int, action_dim: int, device: torch.device) -> dict:
    return {
        "hidden_dim": config.d_model,
        "dim_feedforward": config.mlp_dim,
        "enc_layers": config.num_encoder_layers,
        "dec_layers": config.num_decoder_layers,
        "nheads": config.num_heads,
        "dropout": config.dropout,
        "num_queries": config.chunk_size,
        "camera_names": list(config.camera_names),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "kl_weight": config.kl_weight,
        "lr": config.learning_rate,
        "lr_backbone": config.learning_rate,
        "weight_decay": config.weight_decay,
        "device": device.type,
    }


def prepare_inputs(batch, camera_names, norm_stats, device):
    # Official ACTPolicy applies ImageNet normalization internally; don't double it.
    images, qpos, actions, action_mask = unpack_lerobot_batch(
        batch, camera_names, norm_stats=norm_stats, normalize_images=False
    )
    images = images.to(device)
    qpos = qpos.to(device)
    actions = actions.to(device)
    if action_mask is not None:
        is_pad = (~action_mask).to(device)
    else:
        is_pad = torch.zeros(actions.shape[0], actions.shape[1], dtype=torch.bool, device=device)
    return images, qpos, actions, is_pad


def loss_dict_to_metrics(loss_dict, kl_weight: float) -> dict:
    l1 = loss_dict["l1"].detach().item()
    kl = loss_dict["kl"].detach().item()
    return {
        "total_loss": loss_dict["loss"].detach().item(),
        "action_l1_loss": l1,
        "kl_loss": kl,
        "weighted_kl_loss": kl * kl_weight,
    }


def evaluate(policy, dataloader, camera_names, norm_stats, device, kl_weight):
    policy.eval()
    totals = {}
    total_examples = 0
    with torch.no_grad():
        for batch in dataloader:
            images, qpos, actions, is_pad = prepare_inputs(batch, camera_names, norm_stats, device)
            batch_size = qpos.shape[0]
            loss_dict = policy(qpos, images, actions, is_pad)
            for key, value in loss_dict_to_metrics(loss_dict, kl_weight).items():
                totals[key] = totals.get(key, 0.0) + value * batch_size
            total_examples += batch_size
    return {key: value / max(total_examples, 1) for key, value in totals.items()}


def main(profile: str, run_name: str | None = None, dry_run: bool = False, steps: int | None = None):
    config = load_experiment(profile)
    if dry_run and config.num_workers > 0:
        config = replace(config, num_workers=0)
    if steps is not None:
        config = replace(config, num_train_steps=steps)
    torch.manual_seed(config.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    train_loader, val_loader, norm_stats = make_dataloaders(config, device)
    sample_batch = next(iter(train_loader))
    _, qpos, actions, _ = unpack_lerobot_batch(sample_batch, config.camera_names, norm_stats=None)
    state_dim = qpos.shape[-1]
    action_dim = actions.shape[-1]

    policy = ACTPolicy(build_policy_args(config, state_dim, action_dim, device))
    policy.to(device)
    optimizer = policy.configure_optimizers()

    run_dir = create_run_dir(config, run_name)
    save_config(run_dir, profile, config)
    mlflow_module = start_mlflow_run(profile, config, run_dir)
    print("MODEL official_act_detr_vae")
    print(f"Profile: {profile}")
    print(f"Dataset: {describe_dataset(config)}")
    print(f"Cameras: {config.camera_names}")
    print(f"state_dim={state_dim} action_dim={action_dim} num_queries={config.chunk_size}")
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
            if config.warmup_steps > 0:
                set_warmup_lr(optimizer, config.learning_rate, step, config.warmup_steps)
            batch = next(train_iter)
            step_start = time.perf_counter()

            policy.train()
            optimizer.zero_grad()
            images, qpos, actions, is_pad = prepare_inputs(batch, config.camera_names, norm_stats, device)
            loss_dict = policy(qpos, images, actions, is_pad)
            loss_dict["loss"].backward()
            grad_norm = None
            if config.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip)
            optimizer.step()

            last_train_metrics = loss_dict_to_metrics(loss_dict, config.kl_weight)
            step_seconds = time.perf_counter() - step_start
            batch_size = qpos.shape[0]
            train_log = {
                "step": step,
                "split": "train",
                **last_train_metrics,
                "lr": current_lr(optimizer),
                "step_seconds": step_seconds,
                "examples_per_second": batch_size / max(step_seconds, 1e-9),
                **cuda_memory_metrics(device),
            }
            if grad_norm is not None:
                train_log["grad_norm"] = float(grad_norm.detach().cpu())

            should_eval = val_loader is not None and (step % config.eval_freq == 0 or step == config.num_train_steps)
            if should_eval:
                last_val_metrics = evaluate(policy, val_loader, config.camera_names, norm_stats, device, config.kl_weight)
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
                    policy,
                    optimizer,
                    last_train_metrics["total_loss"],
                    None if last_val_metrics is None else last_val_metrics["total_loss"],
                    config,
                    norm_stats,
                )
                if mlflow_module is not None:
                    mlflow_module.log_param(f"checkpoint_step_{step:06d}", str(checkpoint_path))
                print(f"saved_checkpoint step={step} path={checkpoint_path}", flush=True)
    finally:
        if mlflow_module is not None:
            if (run_dir / "metrics.jsonl").exists():
                mlflow_module.log_artifact(str(run_dir / "metrics.jsonl"), artifact_path="metrics")
            mlflow_module.end_run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the OFFICIAL tonyzhaozh/act DETRVAE on the exact custom dataloader/objective."
    )
    parser.add_argument("--profile", default="sim_transfer_cube_reference", help="Experiment profile in configs/act/<profile>.py")
    parser.add_argument("--run-name", default=None, help="Optional fixed run directory name under the profile job.")
    parser.add_argument("--dry-run", action="store_true", help="Load config/data/model and write config without training.")
    parser.add_argument("--steps", type=int, default=None, help="Override num_train_steps (e.g. for a quick smoke run).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(profile=args.profile, run_name=args.run_name, dry_run=args.dry_run, steps=args.steps)
