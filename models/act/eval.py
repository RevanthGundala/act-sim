from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from .act import ACT
from .train import LeRobotFeatureKey


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    act_repo_dir = resolve_act_repo_dir(args.act_repo_dir)
    install_ipython_stub()
    sys.path.insert(0, str(act_repo_dir))

    from sim_env import BOX_POSE, make_sim_env  # type: ignore
    from utils import sample_box_pose  # type: ignore

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    norm_stats = checkpoint["norm_stats"]
    device = resolve_device(args.device)

    action_stats = stats_for_key(norm_stats, LeRobotFeatureKey.ACTION)
    state_stats = stats_for_key(norm_stats, LeRobotFeatureKey.OBSERVATION_STATE)
    action_dim = int(action_stats["mean"].numel())
    chunk_size = int(config["chunk_size"])
    camera_names = tuple(config["camera_names"])
    episode_len = args.max_steps or int(config.get("benchmark_episode_len") or config.get("max_steps") or 400)

    model = ACT(
        d_model=int(config["d_model"]),
        d_qpos=action_dim,
        d_z=int(config["d_z"]),
        chunk_size=chunk_size,
        device=device,
        num_cameras=len(camera_names),
        num_encoder_layers=int(config["num_encoder_layers"]),
        num_decoder_layers=int(config["num_decoder_layers"]),
        num_heads=int(config["num_heads"]),
        mlp_dim=int(config["mlp_dim"]),
        dropout=float(config["dropout"]),
        max_steps=episode_len + chunk_size,
        pretrained_backbone=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    env = make_sim_env(args.task)
    env_max_reward = env.task.max_reward
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_returns: list[float] = []
    highest_rewards: list[float] = []
    for rollout_id in range(args.episodes):
        BOX_POSE[0] = sample_box_pose()
        result = rollout_episode(
            env=env,
            model=model,
            device=device,
            camera_names=camera_names,
            state_stats=state_stats,
            action_stats=action_stats,
            episode_len=episode_len,
        )
        episode_returns.append(result["return"])
        highest_rewards.append(result["highest_reward"])
        print(
            "rollout="
            f"{rollout_id} return={result['return']:.6g} highest_reward={result['highest_reward']:.6g} "
            f"success={result['highest_reward'] == env_max_reward}",
            flush=True,
        )

    highest_rewards_array = np.asarray(highest_rewards)
    success_rate = float(np.mean(highest_rewards_array == env_max_reward))
    avg_return = float(np.mean(episode_returns))
    reward_thresholds = {
        f"reward_at_least_{reward}": int((highest_rewards_array >= reward).sum())
        for reward in range(env_max_reward + 1)
    }
    summary = {
        "checkpoint": str(checkpoint_path),
        "task": args.task,
        "episodes": args.episodes,
        "episode_len": episode_len,
        "success_rate": success_rate,
        "avg_return": avg_return,
        "env_max_reward": int(env_max_reward),
        "episode_returns": episode_returns,
        "highest_rewards": highest_rewards,
        **reward_thresholds,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"success_rate={success_rate:.6g} avg_return={avg_return:.6g} "
        f"results={summary_path}",
        flush=True,
    )


def rollout_episode(
    *,
    env,
    model: ACT,
    device: torch.device,
    camera_names: tuple[str, ...],
    state_stats: dict[str, torch.Tensor],
    action_stats: dict[str, torch.Tensor],
    episode_len: int,
) -> dict[str, float]:
    ts = env.reset()
    model.reset_action_selection()
    rewards = []

    with torch.inference_mode():
        for timestep in range(episode_len):
            obs = ts.observation
            qpos = normalize_observation(obs["qpos"], state_stats, device)
            images = stack_images(obs["images"], camera_names, device)
            raw_action = model.select_action(timestep, images, qpos)

            action = denormalize_action(raw_action, action_stats)
            if not np.isfinite(action).all():
                raise FloatingPointError(f"Policy produced non-finite action at timestep {timestep}: {action}")
            ts = env.step(action)
            rewards.append(float(ts.reward or 0.0))

    rewards_array = np.asarray(rewards, dtype=np.float32)
    return {
        "return": float(rewards_array.sum()),
        "highest_reward": float(rewards_array.max(initial=0.0)),
    }


def normalize_observation(qpos: np.ndarray, stats: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    qpos_tensor = torch.as_tensor(qpos, dtype=torch.float32, device=device)
    return ((qpos_tensor - stats["mean"].to(device)) / stats["std"].to(device)).unsqueeze(0)


def stack_images(images: dict[str, np.ndarray], camera_names: tuple[str, ...], device: torch.device) -> torch.Tensor:
    # Must mirror the training-time image preprocessing in
    # models/act/train.py::unpack_lerobot_batch: scale to [0, 1] then apply
    # ImageNet normalization (the pretrained/frozen-BN backbone expects it).
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    tensors = []
    for camera_name in camera_names:
        image = torch.as_tensor(np.ascontiguousarray(images[camera_name]), dtype=torch.float32, device=device)
        image = image.permute(2, 0, 1) / 255.0
        if image.shape[0] == 3:
            image = (image - mean) / std
        tensors.append(image)
    return torch.stack(tensors, dim=0).unsqueeze(0)


def denormalize_action(action: torch.Tensor, stats: dict[str, torch.Tensor]) -> np.ndarray:
    action = action.detach().cpu()
    denormalized = action * stats["std"].cpu() + stats["mean"].cpu()
    return denormalized.numpy()


def stats_for_key(norm_stats: dict, key: LeRobotFeatureKey) -> dict[str, torch.Tensor]:
    stats = norm_stats.get(key) or norm_stats.get(str(key))
    if stats is None:
        raise KeyError(f"Checkpoint norm_stats is missing {key}")
    return {
        "mean": torch.as_tensor(stats["mean"], dtype=torch.float32).flatten(),
        "std": torch.as_tensor(stats["std"], dtype=torch.float32).flatten(),
    }


def resolve_act_repo_dir(value: str | None) -> Path:
    act_repo_dir = Path(value or os.environ.get("ACT_REPO_DIR", "")).expanduser()
    if not act_repo_dir:
        raise SystemExit("Set ACT_REPO_DIR or pass --act-repo-dir pointing to a tonyzhaozh/act checkout.")
    if not (act_repo_dir / "sim_env.py").exists():
        raise SystemExit(f"{act_repo_dir} does not look like a tonyzhaozh/act checkout; missing sim_env.py")
    return act_repo_dir


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def install_ipython_stub() -> None:
    try:
        import IPython  # noqa: F401
    except ModuleNotFoundError:
        import types

        module = types.ModuleType("IPython")
        module.embed = lambda *args, **kwargs: None
        sys.modules["IPython"] = module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a custom ACT checkpoint in the official ACT MuJoCo simulator.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_last.pt or checkpoint_step_*.pt.")
    parser.add_argument("--act-repo-dir", default=None, help="Path to tonyzhaozh/act checkout. Defaults to ACT_REPO_DIR.")
    parser.add_argument("--task", default="sim_transfer_cube_scripted", help="Official ACT simulator task name.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of closed-loop rollouts.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override rollout length. Defaults to checkpoint config.")
    parser.add_argument("--seed", type=int, default=1000, help="Evaluation RNG seed.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, or mps.")
    parser.add_argument("--output-dir", default="outputs/act_eval/sim_transfer_cube_reference/seed0", help="Result directory.")
    return parser.parse_args()


if __name__ == "__main__":
    main()