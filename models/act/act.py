import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet18_Weights, resnet18
from torchvision.ops import FrozenBatchNorm2d

from configs.act.base import ACTConfig
from .attention import TransformerBlock
from .pos_emb import ACTSinusoidalPositionEmbedding2d, create_sinusoidal_pos_embedding

class ACTPolicy(nn.Module):
    def __init__(self, config: ACTConfig, pretrained_backbone: bool = False):
        super().__init__()
        self.d_z = config.d_z
        self.cvae_encoder = CVAE_Encoder(
            joint_dim=config.joint_dim,
            d_model=config.d_model,
            d_z=config.d_z,
            num_encoder_layers=config.num_vae_encoder_layers,
            chunk_size=config.chunk_size,
            num_heads=config.num_heads,
            mlp_dim=config.mlp_dim,
            dropout=config.dropout,
        )
        self.decoder = Decoder(
            d_model=config.d_model,
            joint_dim=config.joint_dim,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            d_z=config.d_z,
            chunk_size=config.chunk_size,
            num_cameras=len(config.camera_names),
            num_heads=config.num_heads,
            mlp_dim=config.mlp_dim,
            dropout=config.dropout,
            pretrained_backbone=pretrained_backbone,
        )

    def forward(self, images, qpos, future_actions=None, action_mask=None):
        if future_actions is None:
            z = torch.zeros(qpos.shape[0], self.d_z, dtype=qpos.dtype, device=qpos.device)
            mu, log_var = None, None
        else:
            z, mu, log_var = self.cvae_encoder(qpos, future_actions, action_mask)
        act_pred = self.decoder(images, qpos, z)
        return act_pred, mu, log_var

    @torch.no_grad()
    def select_action(self, images, qpos):
        act_pred, _, _ = self.forward(images, qpos)
        return act_pred

class CVAE_Encoder(nn.Module):
    def __init__(
        self,
        joint_dim,
        d_model,
        d_z,
        num_encoder_layers,
        chunk_size,
        num_heads=8,
        mlp_dim=2048,
        dropout=0.1,
    ):
        super().__init__()
        # 2 separate projects for future actions and current joint pos
        self.joint_proj = nn.Linear(joint_dim, d_model)
        self.act_proj = nn.Linear(joint_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn((1, 1, d_model), )) 
        self.out_proj = nn.Linear(d_model, d_z * 2)
        self.transformer_encoder = nn.ModuleList([
            TransformerBlock(
                dim=d_model,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                use_cross_attention=False,
            ) for _ in range(num_encoder_layers)
        ])
        self.chunk_size = chunk_size
        self.register_buffer("encoder_pos_table", create_sinusoidal_pos_embedding(1 + 1 + chunk_size, d_model), persistent=False)

    def forward(self, qpos, future_actions, action_mask=None):
        bs = future_actions.shape[0]
        qpos_token = self.joint_proj(qpos).unsqueeze(1)
        cls_token = self.cls_token.expand(bs, -1, -1)
        act_tokens = self.act_proj(future_actions) 
        pos = self.encoder_pos_table.unsqueeze(0)
        x = torch.cat([cls_token, qpos_token, act_tokens], dim=1)
        key_padding_mask = None
        if action_mask is not None:
            cls_joint_is_pad = torch.ones((bs, 2), dtype=torch.bool, device=action_mask.device)
            key_padding_mask = torch.cat([cls_joint_is_pad, action_mask.bool()], dim=1)
            key_padding_mask = key_padding_mask[:, None, None, :]
        for block in self.transformer_encoder: 
            x = block(x, key_padding_mask=key_padding_mask, pos=pos)
        # reparameterize
        cls = x[:, 0]
        mu, log_var = self.out_proj(cls).chunk(2, dim=-1)
        std = torch.exp(0.5 * log_var)
        z = mu + std * torch.randn_like(mu)
        return z, mu, log_var


class Decoder(nn.Module):
    def __init__(
        self,
        d_model,
        joint_dim,
        num_encoder_layers,
        num_decoder_layers,
        d_z,
        chunk_size,
        num_cameras=1,
        num_heads=8,
        mlp_dim=2048,
        dropout=0.1,
        pretrained_backbone=False,
    ): 
        super().__init__() 
        self.d_model = d_model
        self.num_cameras = num_cameras
        weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
        backbone = resnet18(weights=weights, norm_layer=FrozenBatchNorm2d)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.image_proj = nn.Linear(512, d_model)
        self.image_pos_emb = ACTSinusoidalPositionEmbedding2d(d_model // 2)
        self.camera_emb = nn.Parameter(torch.zeros(1, num_cameras, d_model))
        self.z_proj = nn.Linear(d_z, d_model)
        self.joint_proj = nn.Linear(joint_dim, d_model)
        self.transformer_encoder = nn.ModuleList([
            TransformerBlock(
                dim=d_model,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                use_cross_attention=False,
            ) for _ in range(num_encoder_layers)
        ]) 
        self.transformer_decoder = nn.ModuleList([
            TransformerBlock(
                dim=d_model,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                use_cross_attention=True,
            ) for _ in range(num_decoder_layers)
        ]) 
        self.joint_z_pos_emb = nn.Embedding(2, d_model)
        self.q_emb = nn.Parameter(torch.randn(1, chunk_size, d_model))
        self.act_proj = nn.Linear(d_model, joint_dim)
    
    def forward(self, images, joints, z): 
        images = self._prepare_images(images)
        bs, num_cameras, channels, height, width = images.shape
        if num_cameras != self.num_cameras:
            raise ValueError(f"Expected {self.num_cameras} cameras, got {num_cameras}")

        image_features = self.backbone(images.reshape(bs * num_cameras, channels, height, width))
        spatial_tokens = image_features.shape[2] * image_features.shape[3]
        image_pos = self.image_pos_emb(image_features).flatten(2).permute(0, 2, 1)
        image_pos = image_pos.to(dtype=image_features.dtype).repeat(1, num_cameras, 1)

        image_tokens = image_features.flatten(2).permute(0, 2, 1)
        image_tokens = self.image_proj(image_tokens)
        image_tokens = image_tokens.reshape(bs, num_cameras, spatial_tokens, self.d_model)
        image_tokens = image_tokens + self.camera_emb.unsqueeze(2)
        image_tokens = image_tokens.reshape(bs, num_cameras * spatial_tokens, self.d_model)

        qpos_token = self.joint_proj(joints).unsqueeze(1)
        z_token = self.z_proj(z).unsqueeze(1)
        memory = torch.cat([z_token, qpos_token, image_tokens], dim=1)
        joint_z_pos = self.joint_z_pos_emb.weight.unsqueeze(0)
        pos = torch.cat([joint_z_pos, image_pos], dim=1)
        for block in self.transformer_encoder:
            memory = block(memory, pos=pos)
        q_tok = self.q_emb.expand(bs, -1, -1)
        out = torch.zeros_like(q_tok)
        for block in self.transformer_decoder:
            out = block(out, encoder_out=memory, pos=pos, query_pos=q_tok)
        act_pred = self.act_proj(out)
        return act_pred

    def _prepare_images(self, images):
        if images.dim() == 4:
            if images.shape[1] in (1, 3):
                images = images.unsqueeze(1)
            elif images.shape[-1] in (1, 3):
                images = images.permute(0, 3, 1, 2).unsqueeze(1)
            else:
                raise ValueError(f"Expected image shape (B,C,H,W) or (B,H,W,C), got {tuple(images.shape)}")
        elif images.dim() == 5:
            if images.shape[2] not in (1, 3) and images.shape[-1] in (1, 3):
                images = images.permute(0, 1, 4, 2, 3)
        else:
            raise ValueError(f"Expected images with 4 or 5 dimensions, got {tuple(images.shape)}")
        # run real normalization in preprocessor
        if not images.is_floating_point():
            images = images.float() / 255.0
        return images
        

        



