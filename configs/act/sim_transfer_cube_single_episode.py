from configs.act.base import ACTConfig


# True single-episode overfit: ONE sim_transfer_cube episode, batch_size 1.
# Decisive sanity check for the official ACT DETRVAE — if it cannot drive
# action_l1_loss far below the custom model's ~0.08 floor on a single episode,
# the problem is not the custom architecture.
config = ACTConfig(
    dataset_repo_id="act/sim_transfer_cube_scripted_single_episode",
    dataset_format="act_hdf5",
    benchmark_dataset_dir="data/benchmarks/act_sim_transfer_cube_single_episode",
    benchmark_task_name="sim_transfer_cube_scripted",
    benchmark_num_episodes=1,
    benchmark_episode_len=400,
    camera_names=("top",),
    job_name="sim_transfer_cube_single_episode",
    batch_size=1,
    chunk_size=100,
    num_workers=0,
    train_split=1.0,
    d_model=512,
    d_z=32,
    num_encoder_layers=4,
    num_decoder_layers=7,
    num_heads=8,
    mlp_dim=3200,
    num_train_steps=3_000,
    learning_rate=1e-5,
    weight_decay=1e-4,
    warmup_steps=0,
    kl_weight=10.0,
    log_freq=25,
    eval_freq=100,
    save_freq=250,
)
