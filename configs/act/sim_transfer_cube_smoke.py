from configs.act.base import ACTConfig


config = ACTConfig(
    dataset_repo_id="act/sim_transfer_cube_scripted",
    dataset_format="act_hdf5",
    benchmark_dataset_dir="data/benchmarks/act_sim_transfer_cube_smoke",
    benchmark_task_name="sim_transfer_cube_scripted",
    benchmark_num_episodes=2,
    benchmark_episode_len=400,
    camera_names=("top",),
    job_name="sim_transfer_cube_smoke",
    batch_size=2,
    chunk_size=100,
    num_workers=0,
    train_split=0.5,
    num_train_steps=1,
    log_freq=1,
    eval_freq=1,
    save_freq=1,
)
