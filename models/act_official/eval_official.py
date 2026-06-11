"""Evaluate an OFFICIAL ACT (models/act_official) checkpoint in the MuJoCo sim.

Mirror of models/act/eval.py but builds the official ACTPolicy/DETRVAE and loads
a checkpoint saved by models/act_official/train_official.py. Uses the same sim
plumbing, qpos/action normalization, and temporal-ensemble action selection as
the custom eval, so success rates are directly comparable.

Images are passed as raw [0, 1] (CHW); the official ACTPolicy applies ImageNet
normalization internally, so we must NOT pre-normalize here.

Usage:
    python -m models.act_official.eval_official \
        --checkpoint <run_dir>/checkpoint_last.pt \
        --act-repo-dir /path/to/tonyzhaozh/act --episodes 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from models.act.eval import (
    denormalize_action,
    install_ipython_stub,
    normalize_observation,
    resolve_act_repo_dir,
    resolve_device,
    set_seed,
    stats_for_key,
)
from models.act.train import LeRobotFeatureKey
from models.act_official.policy import ACTPolicy


def build_policy_args(config: dict, state_dim: int, action_dim: int, device: torch.device) -> dict:
    return {
        "hidden_dim": int(config["d_model"]),
        "dim_feedforward": int(config["mlp_dim"]),
        "enc_layers": int(config["num_encoder_layers"]),
        "dec_layers": int(config["num_decoder_layers"]),
        "nheads": int(config["num_heads"]),
        "dropout": float(config["dropout"]),
        "num_queries": int(config["chunk_size"]),
        "camera_names": list(config["camera_names"]),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "kl_weight": float(config["kl_weight"]),
        "lr": float(config["learning_rate"]),
        "lr_backbone": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "device": device.type,
    }


def stack_images_raw(images: dict, camera_names: tuple[str, ...], device: torch.device) -> torch.Tensor:
    # Raw [0, 1] CHW per camera; ACTPolicy applies ImageNet normalization itself.
    tensors = []
    for camera_name in camera_names:
        image = torch.as_tensor(np.ascontiguousarray(images[camera_name]), dtype=torch.float32, device=device)
        image = image.permute(2, 0, 1) / 255.0
        tensors.append(image)
    return torch.stack(tensors, dim=0).unsqueeze(0)


class TemporalEnsembler:
    """Same exp-weighted temporal ensemble as models/act/act.py::ACT.select_action."""

    def __init__(self, max_steps: int, chunk_size: int, action_dim: int, device: torch.device):
        self.max_steps = max_steps
        self.chunk_size = chunk_size
        self.buffer = torch.zeros((max_steps, max_steps + chunk_size, action_dim), device=device)
        self.mask = torch.zeros((max_steps, max_steps + chunk_size), dtype=torch.bool, device=device)

    def step(self, timestep: int, action_chunk: torch.Tensor) -> torch.Tensor:
        chunk_size = action_chunk.shape[0]
        self.buffer[timestep, timestep : timestep + chunk_size] = action_chunk
        self.mask[timestep, timestep : timestep + chunk_size] = True
        actions = self.buffer[:, timestep][self.mask[:, timestep]]
        weights = torch.exp(-0.01 * torch.arange(len(actions), device=actions.device, dtype=actions.dtype))
        weights = weights / weights.sum()
        return (actions * weights.unsqueeze(1)).sum(dim=0)


def rollout_episode(*, env, policy, device, camera_names, state_stats, action_stats, episode_len, chunk_size):
    ts = env.reset()
    ensembler = TemporalEnsembler(episode_len, chunk_size, int(action_stats["mean"].numel()), device)
    rewards = []
    with torch.inference_mode():
        for timestep in range(episode_len):
            obs = ts.observation
            qpos = normalize_observation(obs["qpos"], state_stats, device)
            images = stack_images_raw(obs["images"], camera_names, device)
            action_chunk = policy(qpos, images)  # (1, num_queries, action_dim)
            raw_action = ensembler.step(timestep, action_chunk.squeeze(0))
            action = denormalize_action(raw_action, action_stats)
            if not np.isfinite(action).all():
                raise FloatingPointError(f"Policy produced non-finite action at timestep {timestep}: {action}")
            ts = env.step(action)
            rewards.append(float(ts.reward or 0.0))
    rewards_array = np.asarray(rewards, dtype=np.float32)
    return {"return": float(rewards_array.sum()), "highest_reward": float(rewards_array.max(initial=0.0))}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    act_repo_dir = resolve_act_repo_dir(args.act_repo_dir)
    install_ipython_stub()
    sys.path.insert(0, str(act_repo_dir))

    from sim_env import BOX_POSE, make_sim_env  # type: ignore
    from utils import sample_box_pose  # type: ignore

    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    norm_stats = checkpoint["norm_stats"]
    device = resolve_device(args.device)

    action_stats = stats_for_key(norm_stats, LeRobotFeatureKey.ACTION)
    state_stats = stats_for_key(norm_stats, LeRobotFeatureKey.OBSERVATION_STATE)
    action_dim = int(action_stats["mean"].numel())
    state_dim = int(state_stats["mean"].numel())
    chunk_size = int(config["chunk_size"])
    camera_names = tuple(config["camera_names"])
    episode_len = args.max_steps or int(config.get("benchmark_episode_len") or 400)

    policy = ACTPolicy(build_policy_args(config, state_dim, action_dim, device))
    policy.to(device)
    policy.model.load_state_dict({k[len("model."):]: v for k, v in checkpoint["model_state_dict"].items() if k.startswith("model.")})
    policy.eval()

    env = make_sim_env(args.task)
    env_max_reward = env.task.max_reward
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_returns, highest_rewards = [], []
    for rollout_id in range(args.episodes):
        BOX_POSE[0] = sample_box_pose()
        result = rollout_episode(
            env=env, policy=policy, device=device, camera_names=camera_names,
            state_stats=state_stats, action_stats=action_stats,
            episode_len=episode_len, chunk_size=chunk_size,
        )
        episode_returns.append(result["return"])
        highest_rewards.append(result["highest_reward"])
        print(
            f"rollout={rollout_id} return={result['return']:.6g} "
            f"highest_reward={result['highest_reward']:.6g} success={result['highest_reward'] == env_max_reward}",
            flush=True,
        )

    highest = np.asarray(highest_rewards)
    success_rate = float(np.mean(highest == env_max_reward))
    summary = {
        "checkpoint": str(args.checkpoint),
        "task": args.task,
        "episodes": args.episodes,
        "episode_len": episode_len,
        "success_rate": success_rate,
        "avg_return": float(np.mean(episode_returns)),
        "env_max_reward": int(env_max_reward),
        "episode_returns": episode_returns,
        "highest_rewards": highest_rewards,
        **{f"reward_at_least_{r}": int((highest >= r).sum()) for r in range(env_max_reward + 1)},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"success_rate={success_rate:.6g} avg_return={summary['avg_return']:.6g} results={output_dir / 'summary.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an official ACT checkpoint in the ACT MuJoCo simulator.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--act-repo-dir", default=None)
    parser.add_argument("--task", default="sim_transfer_cube_scripted")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="outputs/act_eval/official")
    return parser.parse_args()


if __name__ == "__main__":
    main()
