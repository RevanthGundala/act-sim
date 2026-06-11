from configs.act.base import ACTConfig


config = ACTConfig(
    dataset_repo_id="act/sim_transfer_cube_scripted",
    dataset_format="act_hdf5",
    benchmark_dataset_dir="data/benchmarks/act_sim_transfer_cube_scripted",
    benchmark_task_name="sim_transfer_cube_scripted",
    benchmark_num_episodes=50,
    benchmark_episode_len=400,
    camera_names=("top",),
    job_name="sim_transfer_cube_reference",
    batch_size=8,
    chunk_size=100,
    num_workers=1,
    train_split=0.8,
    d_model=512,
    d_z=32,
    num_encoder_layers=4,
    num_decoder_layers=7,
    num_heads=8,
    mlp_dim=3200,
    # 40 train episodes / batch 8 = 5 steps per epoch; official ACT uses 2000 epochs.
    num_train_steps=10_000,
    learning_rate=1e-5,
    weight_decay=1e-4,
    warmup_steps=0,
    kl_weight=10.0,
    log_freq=100,
    eval_freq=1000,
    save_freq=1000,
)
