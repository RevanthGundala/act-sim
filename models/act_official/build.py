"""Faithful port of tonyzhaozh/act detr/main.py::build_ACT_model_and_optimizer.

Builds the genuine DETRVAE and the official AdamW optimizer (separate param
group / lr for the backbone). Device-agnostic instead of hardcoding `.cuda()`.
"""
from types import SimpleNamespace

import torch

from .detr_vae import build as build_vae


# Defaults taken verbatim from the original detr/main.py argument parser.
_DEFAULT_ARGS = {
    "backbone": "resnet18",
    "dilation": False,
    "position_embedding": "sine",
    "dropout": 0.1,
    "pre_norm": False,
    "masks": False,
    "lr": 1e-5,
    "lr_backbone": 1e-5,
    "weight_decay": 1e-4,
    "device": "cuda",
}


def build_ACT_model_and_optimizer(args_override: dict):
    args = SimpleNamespace(**{**_DEFAULT_ARGS, **args_override})

    model = build_vae(args)
    device = torch.device(args.device)
    model.to(device)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer
