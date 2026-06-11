# Debugging the custom ACT — learnings

Record of the investigation into why the custom `models/act` ACT policy was
training poorly, what we changed, and (importantly) what we did and did not
actually measure.

## TL;DR

- The custom ACT was failing: a hard ~0.08 normalized-L1 floor on a
  single-episode overfit, with a weak/ignored latent.
- We ran the **genuine** tonyzhaozh/act DETRVAE on our **exact** dataloader and
  objective (vendored under `models/act_official/`) as a decisive A/B and fixed
  the custom model piece by piece. On the multi-episode `sim_transfer_cube_reference`
  benchmark its **held-out validation L1 matched** the official implementation
  (~0.196 vs ~0.220).
- **BUT the sim rollout overturns the "we're done" framing.** On the actual
  MuJoCo task, **both** models are catastrophically below the paper's ~90%:
  patched custom **4%** success, faithful official ACT **0%** (50 rollouts each).
  Since even the genuine official implementation fails here, **the bottleneck is
  our recipe/harness, not the custom architecture.** Val L1 ~0.2 is itself an
  underfitting symptom (both policies mostly hold pose; ~14% cube touch).
- Net: the architecture fixes were real and necessary (latent now works, custom
  == official on val L1), but a working policy still requires fixing the
  training/eval **recipe** to reproduce the paper.

## Fixes applied to `models/act` (in order)

1. **Main transformer encoder** (`act.py`). The decoder cross-attended to raw
   projected tokens; there was no DETR/ACT-style encoder. Added a self-attention
   stack over the `[z, joint, image]` memory before the decoder. Highest-impact
   structural fix.
2. **Reparameterization sampling** (`act.py`). Latent uses `z ~ N(mu, sigma)`
   (not `z = mu`).
3. **ACT-style KL** (`train.py`). KL is now sum-over-latent-dims, mean-over-batch
   (matches official). The previous mean-over-all-dims form was ~`d_z`x weaker and
   let the posterior collapse.
4. **Attention mask inversion fix** (`attention.py`). `future_mask` is built with
   `1 = keep / 0 = pad`, but `masked_fill` used `== 1`, masking the **valid**
   tokens (and collapsing to uniform attention when there was no padding). Fixed
   to `== 0`.
5. **2-D sine positional embedding** for image tokens (`act.py`), replacing the
   1-D flatten-index table. Row/col encoded on separate channel halves.
6. **Per-layer positional injection** (`attention.py`, `act.py`). Positions are
   added to q/k at **every** encoder/decoder attention layer (`with_pos_embed`
   style), and the decoder cross-attention adds pos to memory keys + `query_pos`
   to queries, instead of baking position into the residual once. Split
   `cross_kv_proj` into separate `cross_k_proj`/`cross_v_proj` so the value stays
   position-free; decoder starts from zeros with `q_emb` as the learned query pos.
7. **FrozenBatchNorm2d backbone** (`act.py`), like official ACT, instead of
   trainable BatchNorm (unstable at small batch sizes).
8. **ImageNet image normalization** (`train.py`). Images are normalized with
   ImageNet mean/std before the pretrained ResNet (previously raw `[0,1]`).
   Opt-out via `normalize_images=` and disabled for the official harness, whose
   `ACTPolicy` normalizes internally (avoids double-normalizing).

## Decisive A/B harness

- `models/act_official/` is a faithful vendored copy of tonyzhaozh/act
  (detr_vae / transformer / backbone / position_encoding / policy), adapted only
  for: removed `IPython` debug hooks, torchvision `pretrained=` -> `weights=`,
  parametrized the ALOHA-hardcoded dim 14 to `state_dim`/`action_dim`, and a
  device-agnostic builder.
- `models/act_official/train_official.py` reuses our **exact** dataloader, norm
  stats, and L1+KL objective from `models/act/train.py` — only the model is
  swapped. This is what made the comparison trustworthy.
- Modal launchers: `deploy/train_custom_act_modal.py` and
  `deploy/train_official_act_modal.py` (mirrors of each other, same image and
  benchmark volume).

## Results (held-out validation L1, `sim_transfer_cube_reference`, 40 train / 10 val)

| step  | Custom (patched) | Official ACT |
|------:|-----------------:|-------------:|
| 1000  | 0.420            | 0.247        |
| 3000  | 0.219            | 0.237        |
| 5000  | 0.213            | 0.235        |
| 7000  | 0.196            | 0.231        |
| 10000 | **0.196**        | **0.220**    |

Final train L1 was essentially identical (custom 0.030 / official 0.0275).

## Key methodological lesson: overfit L1 was a red herring

On a **single-episode overfit**, the custom model was pinned at ~0.025–0.029
while official reached ~0.014 — and **none** of fixes 4–8 (mask, 2-D pos,
per-layer pos, FrozenBN, ImageNet norm) moved that number. A single fully-valid
episode is memorizable regardless of these details, so the overfit metric was
**blind** to the fixes. The **multi-episode held-out val L1** was the metric that
actually reflected the improvements (padding, spatial generalization, a working
latent). Evaluate ACT changes on multi-episode val, not single-episode overfit.

## Sim eval (actually run, locally) — the decisive result

We ran the MuJoCo rollout eval locally for **both** models, 50 rollouts each,
with identical sim plumbing + temporal-ensemble action selection:

- Patched custom: `models/act/eval.py`
- Official ACT:   `models/act_official/eval_official.py` (mirror; builds
  `ACTPolicy`, feeds raw `[0,1]` images since the policy ImageNet-normalizes
  internally)

| metric (`sim_transfer_cube`) | Patched custom | Official ACT | Paper target |
|---|---|---|---|
| success (reward 4) | **4%** (2/50) | **0%** (0/50) | **~90%** |
| lifted (reward >= 2) | 4% (2/50) | 4% (2/50) | — |
| touched (reward >= 1) | 14% (7/50) | 14% (7/50) | — |
| avg return | 34.0 | 3.6 | — |

(Paper/README: "success rate should be around 90% for transfer cube, ~50% for
insertion".)

### Interpretation (this overturns the "custom == official, done" framing)

- **Both models are catastrophically below ~90%, and the faithful official ACT
  gets 0%.** Since the genuine implementation also fails in our setup, the
  bottleneck is the **recipe/harness, not the custom architecture.**
- Val L1 ~0.2 parity (0.196 vs 0.220) does **not** mean the policy works — it is
  an **underfitting symptom**. Both policies essentially hold pose (~14% touch),
  whereas a converged ACT touches ~100% and reaches far lower L1.
- The architecture fixes were still real and necessary (latent works; custom
  matches official on val L1), but they were never the thing blocking a working
  policy.

### Eval is trustworthy (ruled out artifacts)

- Harness runs end-to-end locally (`mujoco`/`dm_control` in the venv; renders OK).
- **No dimension bug.** `action_spec=(16,)` is raw mujoco actuators; the task's
  `before_step` consumes a **14-dim** policy action. Our 14-dim actions are right.
- **No distribution mismatch.** Training HDF5 `top` image vs eval-rendered `top`
  image are identical (`(480,640,3)`, mean 39.9); reset qpos matches; actions are
  finite/in-range.

## Likely recipe culprits (shared by both models, since both fail)

- **Demos / training length vs the paper:** paper uses 50 demos / 2000 epochs; we
  used 40 train demos / 10k steps with val L1 plateaued ~0.2 (under-converged).
- **Eval protocol:** query frequency / temporal-aggregation vs official
  `imitate_episodes.py`.
- **Normalization:** `NORM_STD_MIN=1e-2` clamp can distort near-constant action
  dims.
- **Objective/data:** whether chunk sampling / loss lets the model collapse to
  "predict current pose".

## Suggested next steps

1. Reproduce the paper recipe first with the **official** model (all 50 demos,
   match epochs/eval protocol) until it hits ~90% — that validates the harness.
   Only then is a custom-vs-official success-rate comparison meaningful.
2. Once the harness reproduces the paper, re-evaluate the patched custom model.
3. Retrain on the real OMX pour-water dataset with these fixes and check the
   latent (KL not collapsing) and val behavior there.

## Reproduce

```bash
# Multi-episode reference comparison (Modal A10G), val L1:
modal run --detach deploy/train_official_act_modal.py --profile sim_transfer_cube_reference --run-name ref_official_seed0
modal run --detach deploy/train_custom_act_modal.py   --profile sim_transfer_cube_reference --run-name ref_patched_seed0

# Metrics land in the volumes (root = /outputs; do NOT prefix /outputs in `modal volume get`):
modal volume get omx-custom-act-training-logs   /act_experiments/sim_transfer_cube_reference/ref_patched_seed0/metrics.jsonl .
modal volume get omx-official-act-training-logs /act_experiments/sim_transfer_cube_reference/ref_official_seed0/metrics.jsonl .

# Real MuJoCo success-rate eval (needs ACT_REPO_DIR + MuJoCo) — NOT yet run:
python -m models.act.eval --checkpoint <run_dir>/checkpoint_last.pt --act-repo-dir /path/to/tonyzhaozh/act
```
