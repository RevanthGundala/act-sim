# act-sim

A focused extraction of the ACT (Action Chunking Transformer) work from
`omx-training` — only the files we actually used while debugging the custom ACT
and benchmarking it against the genuine implementation on the ALOHA
`sim_transfer_cube` task. No robot/teleop/PI0/DAgger clutter, no `debug_*.py`.

## The two "worlds"

Most of the confusion in the original repo came from mixing two setups. They are
kept distinct here:

| | World A — our ACT investigation | World B — known-good reference |
|---|---|---|
| models | `models/act/` (custom) + `models/act_official/` (vendored genuine ACT) | stock LeRobot ACT (pip, no repo files) |
| data | tonyzhaozh **scripted** HDF5 (`benchmarks/act_sim`) | `lerobot/aloha_sim_transfer_cube_human` |
| train | `models/*/train*.py`, `deploy/train_{custom,official}_act_modal.py` | `deploy/train_act_aloha_modal.py` (`lerobot-train`) |
| eval | `models/act/eval.py`, `models/act_official/eval_official.py` (tonyzhaozh `sim_env`, needs `ACT_REPO_DIR`) | `lerobot-eval` + `gym-aloha` |

See `docs/act_debugging_learnings.md` for the full story, results, and the key
lesson (val L1 parity did NOT imply task success; both custom and faithful
official ACT scored ~0–4% in our recipe, while a properly trained LeRobot ACT
hits ~80% in gym-aloha — so the bottleneck is the training recipe, not the
architecture).

## Layout

```
models/act/            custom ACT (the implementation we fixed)
  act.py               model: backbone + CVAE encoder + main encoder + decoder
  attention.py         transformer blocks / attention (incl. mask + per-layer pos)
  train.py             training loop + LeRobot/HDF5 dataloader + L1+KL objective
  eval.py              MuJoCo rollout success-rate eval (tonyzhaozh sim_env)
models/act_official/   faithful vendored tonyzhaozh/act DETRVAE (the A/B baseline)
  detr_vae.py transformer.py backbone.py position_encoding.py misc.py
  policy.py build.py   ACTPolicy wrapper + optimizer
  train_official.py    trains it through the SAME dataloader as the custom model
  eval_official.py     MuJoCo rollout eval for the official model
configs/act/           experiment profiles (base + sim_transfer_cube_* + pour_water)
benchmarks/act_sim/    scripted-sim HDF5 dataset loader (dataset.py) + generator
utils/config.py        shared constants (dataset repo id, etc.)
deploy/                Modal launchers (custom, official, stock-LeRobot-aloha)
docs/                  act_debugging_learnings.md
```

## Setup

```bash
uv sync
```

## Common commands

```bash
# --- World A: custom vs official, same dataloader (val L1 / loss) ---
# local smoke on a single sim episode (needs benchmark HDF5 under data/benchmarks/)
python -m models.act.train --profile sim_transfer_cube_single_episode
python -m models.act_official.train_official --profile sim_transfer_cube_single_episode

# Modal (A10G):
modal run --detach deploy/train_custom_act_modal.py::train_remote   --profile sim_transfer_cube_reference
modal run --detach deploy/train_official_act_modal.py::train_remote --profile sim_transfer_cube_reference

# MuJoCo success-rate eval (needs a tonyzhaozh/act checkout for sim_env):
python -m models.act.eval --checkpoint <run>/checkpoint_last.pt --act-repo-dir /path/to/tonyzhaozh/act --episodes 50
python -m models.act_official.eval_official --checkpoint <run>/checkpoint_last.pt --act-repo-dir /path/to/tonyzhaozh/act --episodes 50

# --- World B: stock LeRobot ACT on the aloha dataset, eval in gym-aloha ---
modal run --detach deploy/train_act_aloha_modal.py::train_remote --steps 80000
modal run deploy/train_act_aloha_modal.py::eval_remote --steps 80000 --episodes 50

# local gym-aloha eval of a released/known-good checkpoint (use sync envs on macOS):
lerobot-eval --policy.path=<dir> --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=50 --eval.use_async_envs=false --output_dir=outputs/eval
```

## Notes

- Headless MuJoCo render (Modal/servers): set `MUJOCO_GL=egl` and install
  `libegl1 libgl1 libgles2 libosmesa6` (already wired in
  `deploy/train_act_aloha_modal.py`).
- macOS gym-aloha eval: pass `--eval.use_async_envs=false` (async vector envs
  hit a `BrokenPipe` on mac multiprocessing).
- Disable MLflow with `ACT_DISABLE_MLFLOW=1`.
