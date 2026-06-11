"""Vendored, faithful copy of the original tonyzhaozh/act DETRVAE.

Source: https://github.com/tonyzhaozh/act (detr/models/*, detr/util/misc.py, policy.py)

Only adaptations vs. upstream:
- Removed `import IPython; e = IPython.embed` debug hooks (IPython not installed).
- ALOHA-hardcoded state/action dim (14) is parametrized as state_dim/action_dim
  so it matches the OMX dataset (6-DOF) inferred from the dataloader.
- backbone.py uses the modern torchvision `weights=` API instead of the removed
  `pretrained=` kwarg (torchvision >= 0.15).
- build/optimizer is device-agnostic instead of hardcoding `.cuda()`.

Everything else (transformer, encoder/decoder, CVAE, KL, L1 objective) is verbatim.
"""
