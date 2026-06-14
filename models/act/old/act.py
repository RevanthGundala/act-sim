import math

import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet18_Weights, resnet18
from torchvision.ops import FrozenBatchNorm2d

from ..attention import TransformerBlock


def sinusoidal_positional_encoding(length, dim, device):
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0)


def sinusoidal_positional_encoding_2d(height, width, dim, device):
    """2-D sine positional encoding (DETR/ACT-style): row and column are encoded
    on separate halves of the channels. Returns (1, height*width, dim) in the
    same row-major order as a flattened feature map."""
    if dim % 4 != 0:
        raise ValueError(f"2D positional encoding requires dim divisible by 4, got {dim}")
    half_dim = dim // 2
    div_term = torch.exp(
        torch.arange(0, half_dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / half_dim)
    )
    pos_h = torch.arange(height, device=device, dtype=torch.float32).unsqueeze(1)
    pos_w = torch.arange(width, device=device, dtype=torch.float32).unsqueeze(1)

    pe_h = torch.zeros(height, half_dim, device=device)
    pe_h[:, 0::2] = torch.sin(pos_h * div_term)
    pe_h[:, 1::2] = torch.cos(pos_h * div_term)

    pe_w = torch.zeros(width, half_dim, device=device)
    pe_w[:, 0::2] = torch.sin(pos_w * div_term)
    pe_w[:, 1::2] = torch.cos(pos_w * div_term)

    pe = torch.zeros(height, width, dim, device=device)
    pe[:, :, :half_dim] = pe_h.unsqueeze(1)
    pe[:, :, half_dim:] = pe_w.unsqueeze(0)
    return pe.reshape(height * width, dim).unsqueeze(0)


class ACT(nn.Module):
    def __init__(
            self,
            d_model,
            d_qpos, 
            d_z,
            chunk_size,
            device,
            num_cameras=2,
            num_encoder_layers=4,
            num_decoder_layers=4,
            num_heads=8,
            mlp_dim=2048,
            dropout=0.1,
            max_steps=1000,
            pretrained_backbone=True,
    ):
        super(ACT, self).__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
        # FrozenBatchNorm2d (fixed ImageNet stats, like official ACT) instead of
        # trainable BatchNorm, which is unstable at small batch sizes.
        backbone = resnet18(weights=weights, norm_layer=FrozenBatchNorm2d)
        self.image_encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.device = device
        self.d_z = d_z
        self.num_cameras = num_cameras
        self.d_model = d_model
        self.chunk_size = chunk_size
        
        # 512 == resnet final feature dim
        self.image_proj = nn.Linear(512, d_model)
        self.camera_emb = nn.Parameter(torch.zeros(1, num_cameras, d_model))
        self.joint_proj = nn.Linear(d_qpos, d_model)
        self.z_proj = nn.Linear(d_z, d_model)

        self.cvae_encoder = Encoder(
            d_model,
            d_qpos,
            d_z,
            chunk_size,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        self.cvae_decoder = Decoder(
            d_model,
            d_qpos,
            chunk_size,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        self.encoder = nn.ModuleList([
            TransformerBlock(d_model, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])
        self.extra_pos_emb = nn.Parameter(torch.zeros(1, 2, d_model))

        self.max_steps = max_steps
        self.action_buffer = None
        self.action_mask = None

    def forward(self, images, qpos, future_qpos = None, action_mask=None):
        # preprocess images 
        B, N, C, H, W = images.shape
        if N != self.num_cameras:
            raise ValueError(f"Expected {self.num_cameras} cameras, got {N}")

        images = images.reshape(B * N, C, H, W)
        image_features = self.image_encoder(images)
        feat_h, feat_w = image_features.shape[2], image_features.shape[3]
        spatial_tokens = feat_h * feat_w
        image_features = image_features.flatten(2).permute(0, 2, 1)
        image_features = self.image_proj(image_features)
        image_features = image_features.reshape(B, N, spatial_tokens, self.d_model)

        # camera_emb is a per-camera content tag; spatial positions come from a 2-D
        # sine table injected at every attention layer (ACT/DETR-style), not baked in.
        image_features = image_features + self.camera_emb.unsqueeze(2)
        image_features = image_features.reshape(B, N * spatial_tokens, self.d_model)

        image_pos = sinusoidal_positional_encoding_2d(feat_h, feat_w, self.d_model, image_features.device)
        image_pos = image_pos.repeat(1, N, 1)

        mu, log_var = None, None
        if future_qpos is not None: 
            z, mu, log_var = self.cvae_encoder(qpos, future_qpos, action_mask=action_mask)
        else:
            z = torch.zeros(B, self.d_z, device=qpos.device)

        joint_token = self.joint_proj(qpos).unsqueeze(1)
        z_token = self.z_proj(z).unsqueeze(1)

        # ACT-style: main encoder self-attends over [z, joint, image] memory, then
        # the decoder queries cross-attend to it. Positions are injected per layer.
        memory = torch.cat([z_token, joint_token, image_features], dim=1)
        memory_pos = torch.cat([self.extra_pos_emb, image_pos], dim=1)
        for block in self.encoder:
            memory = block(memory, pos=memory_pos)

        act_pred = self.cvae_decoder(memory, memory_pos)
        return act_pred, mu, log_var
    
    @torch.inference_mode()
    def select_action(self, timestep, images, qpos):
        action_pred, _, _ = self.forward(images, qpos)
        if action_pred.shape[0] != 1:
            raise ValueError(f"select_action expects batch size 1, got {action_pred.shape[0]}")
        if timestep < 0 or timestep >= self.max_steps:
            raise ValueError(f"timestep {timestep} is outside max_steps={self.max_steps}")

        action_pred = action_pred.squeeze(0)
        chunk_size, action_dim = action_pred.shape
        if chunk_size != self.chunk_size:
            raise ValueError(f"Expected chunk_size={self.chunk_size}, got {chunk_size}")

        if self.action_buffer is None or self.action_mask is None:
            self.action_buffer = torch.zeros(
                (self.max_steps, self.max_steps + self.chunk_size, action_dim),
                device=action_pred.device,
                dtype=action_pred.dtype,
            )
            self.action_mask = torch.zeros(
                (self.max_steps, self.max_steps + self.chunk_size),
                dtype=torch.bool,
                device=action_pred.device,
            )

        self.action_buffer[timestep, timestep : timestep + chunk_size] = action_pred
        self.action_mask[timestep, timestep : timestep + chunk_size] = True
        actions_for_current_step = self.action_buffer[:, timestep]
        action_masks_for_current_step = self.action_mask[:, timestep]
        actions = actions_for_current_step[action_masks_for_current_step]
        weights = torch.exp(-0.01 * torch.arange(len(actions), device=actions.device, dtype=actions.dtype))
        weights = weights / weights.sum()
        return (actions * weights.unsqueeze(1)).sum(dim=0)

    def reset_action_selection(self):
        self.action_buffer = None
        self.action_mask = None


class Encoder(nn.Module):
    def __init__(
            self, 
            d_model, 
            d_qpos,
            d_z,
            chunk_size,
            num_layers=4,
                num_heads=8,
                mlp_dim=2048,
                dropout=0.1,
    ):
        super(Encoder, self).__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.qpos_proj = nn.Linear(d_qpos, d_model)
        self.action_proj = nn.Linear(d_qpos, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout) for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, d_z * 2)
        self.register_buffer(
            "encoder_pos_table",
            sinusoidal_positional_encoding(1 + 1 + chunk_size, d_model, device=self.cls_token.device),
            persistent=False,
        )

    def encode_cls(self, qpos, future_qpos, action_mask=None):
        B = qpos.shape[0]
        qpos_token = self.qpos_proj(qpos).unsqueeze(1)
        action_tokens = self.action_proj(future_qpos)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, qpos_token, action_tokens], dim=1)
        pos = self.encoder_pos_table[:, : x.shape[1]]

        future_mask = None
        if action_mask is not None:
            # joint token + cls mask + future action mask
            prefix_mask = torch.ones(B, 2, dtype=torch.bool, device=action_mask.device)
            future_mask = torch.cat([prefix_mask, action_mask.bool()], dim=1)
            future_mask = future_mask[:, None, None, :]

        for block in self.blocks:
            x = block(x, future_mask=future_mask, pos=pos)
        return x

    def forward(self, qpos, future_qpos, action_mask=None):
        cls_output = self.encode_cls(qpos, future_qpos, action_mask=action_mask)
        cls = cls_output[:, 0]
        z_params = self.out_proj(cls)
        mu, log_var = z_params.chunk(2, dim=-1)
        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        return z, mu, log_var

class Decoder(nn.Module):
    def __init__(
            self,
            d_model,
            d_qpos,
            chunk_size,
            num_layers=4,
                num_heads=8,
                mlp_dim=2048,
                dropout=0.1,
    ):
        super(Decoder, self).__init__()
        self.q_emb = nn.Parameter(torch.zeros(1, chunk_size, d_model))
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                use_cross_attention=True,
            ) for _ in range(num_layers)
        ])
        self.act_proj = nn.Linear(d_model, d_qpos)
        self.last_hidden = None

    def forward(self, x, memory_pos=None):
        B = x.size(0)
        query_pos = self.q_emb.expand(B, -1, -1)
        tgt = torch.zeros_like(query_pos)
        for block in self.blocks:
            tgt = block(tgt, encoder_out=x, pos=memory_pos, query_pos=query_pos)
        self.last_hidden = tgt.detach()
        act_pred = self.act_proj(tgt)
        return act_pred
        