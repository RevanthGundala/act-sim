from dataclasses import dataclass


@dataclass(frozen=True)
class ACTConfig:
    # Dataset
    dataset_repo_id: str
    dataset_revision: str = "main"
    dataset_format: str = "lerobot"
    camera_names: tuple[str, ...] = ("wrist", "top")
    benchmark_dataset_dir: str | None = None
    benchmark_task_name: str | None = None
    benchmark_num_episodes: int | None = None
    benchmark_episode_len: int | None = None

    # Run identity
    job_name: str = "act_pour_water"
    output_root: str = "outputs/act_experiments"

    # Data loading
    batch_size: int = 8
    chunk_size: int = 100
    num_workers: int = 4
    train_split: float = 0.95

    # Model
    d_model: int = 512
    d_z: int = 32
    num_encoder_layers: int = 4
    num_decoder_layers: int = 7
    num_heads: int = 8
    mlp_dim: int = 3200
    dropout: float = 0.1

    # Optimization
    num_train_steps: int = 100_000
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    kl_weight: float = 10.0
    dropout: float = 0.1
    seed: int = 1000

    # Logging/checkpointing
    log_freq: int = 100
    eval_freq: int = 1000
    save_freq: int = 5000