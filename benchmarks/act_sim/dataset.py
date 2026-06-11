from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from configs.act.base import ACTConfig


OBSERVATION_STATE = "observation.state"
ACTION = "action"
ACTION_IS_PAD = "action_is_pad"
OBSERVATION_IMAGES_PREFIX = "observation.images."
SIM_TRANSFER_CUBE_TASK = "sim_transfer_cube_scripted"
SIM_TRANSFER_CUBE_EPISODE_LEN = 400
SIM_TRANSFER_CUBE_ACTION_DIM = 14
NORM_STD_MIN = 1e-2


@dataclass(frozen=True)
class EpisodeFile:
    episode_id: int
    path: Path


def make_act_hdf5_dataloaders(config: ACTConfig, device: torch.device):
    if config.benchmark_task_name != SIM_TRANSFER_CUBE_TASK:
        raise ValueError(
            "Only sim_transfer_cube_scripted is supported by this benchmark path; "
            f"got {config.benchmark_task_name!r}"
        )
    if tuple(config.camera_names) != ("top",):
        raise ValueError(f"{SIM_TRANSFER_CUBE_TASK} expects camera_names=('top',), got {config.camera_names!r}")
    if config.chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {config.chunk_size}")

    episode_files = discover_episode_files(config)
    norm_stats = compute_norm_stats(episode_files, device)
    train_files, val_files = split_episode_files(episode_files, config.train_split, config.seed)

    train_dataset = ACTHDF5Dataset(
        train_files,
        camera_names=config.camera_names,
        chunk_size=config.chunk_size,
        deterministic_start=False,
        seed=config.seed,
    )
    val_dataset = (
        ACTHDF5Dataset(
            val_files,
            camera_names=config.camera_names,
            chunk_size=config.chunk_size,
            deterministic_start=True,
            seed=config.seed,
        )
        if val_files
        else None
    )

    train_loader = make_loader(train_dataset, config, shuffle=True)
    val_loader = make_loader(val_dataset, config, shuffle=False) if val_dataset is not None else None
    return train_loader, val_loader, norm_stats


class ACTHDF5Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episode_files: list[EpisodeFile],
        camera_names: tuple[str, ...],
        chunk_size: int,
        deterministic_start: bool,
        seed: int,
    ) -> None:
        if not episode_files:
            raise ValueError("ACTHDF5Dataset needs at least one episode file")
        self.episode_files = episode_files
        self.camera_names = camera_names
        self.chunk_size = chunk_size
        self.deterministic_start = deterministic_start
        rng = random.Random(seed)
        self.fixed_start_by_episode = {
            episode.path: rng.randrange(read_episode_len(episode.path)) for episode in episode_files
        }

    def __len__(self) -> int:
        return len(self.episode_files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode = self.episode_files[index]
        with h5py.File(episode.path, "r") as root:
            action_len, action_dim = root["/action"].shape
            if action_dim != SIM_TRANSFER_CUBE_ACTION_DIM:
                raise ValueError(f"Expected action dim 14 in {episode.path}, got {action_dim}")
            start_ts = self.fixed_start_by_episode[episode.path] if self.deterministic_start else random.randrange(action_len)

            qpos = np.asarray(root["/observations/qpos"][start_ts], dtype=np.float32)
            if qpos.shape != (SIM_TRANSFER_CUBE_ACTION_DIM,):
                raise ValueError(f"Expected qpos shape (14,) in {episode.path}, got {qpos.shape}")

            action_chunk = np.zeros((self.chunk_size, action_dim), dtype=np.float32)
            action_is_pad = np.ones((self.chunk_size,), dtype=bool)
            available = min(self.chunk_size, action_len - start_ts)
            if available > 0:
                action_chunk[:available] = root["/action"][start_ts : start_ts + available]
                action_is_pad[:available] = False

            batch = {
                OBSERVATION_STATE: torch.from_numpy(qpos),
                ACTION: torch.from_numpy(action_chunk),
                ACTION_IS_PAD: torch.from_numpy(action_is_pad),
            }
            for camera_name in self.camera_names:
                image = np.asarray(root[f"/observations/images/{camera_name}"][start_ts])
                if image.ndim != 3 or image.shape[-1] != 3:
                    raise ValueError(
                        f"Expected {camera_name} image shape [H, W, 3] in {episode.path}, got {image.shape}"
                    )
                batch[f"{OBSERVATION_IMAGES_PREFIX}{camera_name}"] = torch.from_numpy(image)
        return batch


def discover_episode_files(config: ACTConfig) -> list[EpisodeFile]:
    if config.benchmark_dataset_dir is None:
        raise ValueError("benchmark_dataset_dir must be set for dataset_format='act_hdf5'")

    dataset_dir = Path(config.benchmark_dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Benchmark dataset not found: {dataset_dir}. "
            "Generate sim_transfer_cube episodes with `python -m benchmarks.act_sim.generate` first."
        )

    episode_files = []
    for path in sorted(dataset_dir.glob("episode_*.hdf5")):
        match = re.fullmatch(r"episode_(\d+)\.hdf5", path.name)
        if match is None:
            continue
        episode_files.append(EpisodeFile(int(match.group(1)), path))

    episode_files.sort(key=lambda episode: episode.episode_id)
    if config.benchmark_num_episodes is not None:
        episode_files = episode_files[: config.benchmark_num_episodes]

    expected = config.benchmark_num_episodes
    if expected is not None and len(episode_files) != expected:
        raise FileNotFoundError(f"Expected {expected} HDF5 episodes in {dataset_dir}, found {len(episode_files)}")
    if not episode_files:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in {dataset_dir}")

    for episode in episode_files:
        validate_episode_file(episode.path, config)
    return episode_files


def validate_episode_file(path: Path, config: ACTConfig) -> None:
    with h5py.File(path, "r") as root:
        if "/observations/qpos" not in root or "/action" not in root:
            raise ValueError(f"{path} is missing /observations/qpos or /action")
        episode_len, action_dim = root["/action"].shape
        if action_dim != SIM_TRANSFER_CUBE_ACTION_DIM:
            raise ValueError(f"{path} has action dim {action_dim}; expected {SIM_TRANSFER_CUBE_ACTION_DIM}")
        if root["/observations/qpos"].shape != (episode_len, action_dim):
            raise ValueError(
                f"{path} qpos shape {root['/observations/qpos'].shape} does not match action shape {(episode_len, action_dim)}"
            )
        expected_len = config.benchmark_episode_len
        if expected_len is not None and episode_len != expected_len:
            raise ValueError(f"{path} has episode_len={episode_len}; expected {expected_len}")
        for camera_name in config.camera_names:
            image_key = f"/observations/images/{camera_name}"
            if image_key not in root:
                raise ValueError(f"{path} is missing {image_key}")
            if root[image_key].shape[0] != episode_len:
                raise ValueError(f"{path} {image_key} length does not match action length")


def read_episode_len(path: Path) -> int:
    with h5py.File(path, "r") as root:
        return int(root["/action"].shape[0])


def compute_norm_stats(episode_files: list[EpisodeFile], device: torch.device) -> dict[str, dict[str, torch.Tensor]]:
    qpos_arrays = []
    action_arrays = []
    for episode in episode_files:
        with h5py.File(episode.path, "r") as root:
            qpos_arrays.append(torch.from_numpy(root["/observations/qpos"][()]).float())
            action_arrays.append(torch.from_numpy(root["/action"][()]).float())

    qpos_data = torch.cat(qpos_arrays, dim=0)
    action_data = torch.cat(action_arrays, dim=0)
    return {
        OBSERVATION_STATE: {
            "mean": qpos_data.mean(dim=0).to(device),
            "std": qpos_data.std(dim=0).clamp_min(NORM_STD_MIN).to(device),
        },
        ACTION: {
            "mean": action_data.mean(dim=0).to(device),
            "std": action_data.std(dim=0).clamp_min(NORM_STD_MIN).to(device),
        },
    }


def split_episode_files(
    episode_files: list[EpisodeFile], train_split: float, seed: int
) -> tuple[list[EpisodeFile], list[EpisodeFile]]:
    if not 0.0 < train_split <= 1.0:
        raise ValueError(f"train_split must be in (0, 1], got {train_split}")
    shuffled = episode_files.copy()
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) == 1 or train_split == 1.0:
        return shuffled, []
    train_count = int(len(shuffled) * train_split)
    train_count = min(max(train_count, 1), len(shuffled) - 1)
    return shuffled[:train_count], shuffled[train_count:]


def make_loader(dataset: ACTHDF5Dataset, config: ACTConfig, shuffle: bool) -> torch.utils.data.DataLoader:
    kwargs = {
        "batch_size": config.batch_size,
        "shuffle": shuffle,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if config.num_workers > 0:
        kwargs["prefetch_factor"] = 1
    return torch.utils.data.DataLoader(dataset, **kwargs)
