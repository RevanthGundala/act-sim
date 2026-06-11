import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_dim=2048, dropout=0.1, use_cross_attention=False):
        super(TransformerBlock, self).__init__()
        self.self_attn = Attention(dim, num_heads, dropout)
        self.cross_attn = Attention(dim, num_heads, dropout, use_cross_attention=True) if use_cross_attention else None
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        self.self_norm = nn.LayerNorm(dim)
        self.cross_norm = nn.LayerNorm(dim) if use_cross_attention else None
        self.mlp_norm = nn.LayerNorm(dim)

    def forward(self, x, encoder_out=None, future_mask=None, pos=None, query_pos=None):
        # Decoder blocks position their self-attention with the query positions;
        # encoder blocks position theirs with the token positions (pos).
        self_pos = query_pos if self.cross_attn is not None else pos
        self_norm = self.self_norm(x)
        attn_output = self.self_attn(self_norm, future_mask=future_mask, pos=self_pos)
        x = x + attn_output

        if self.cross_attn is not None and encoder_out is not None:
            cross_norm = self.cross_norm(x)
            attn_output = self.cross_attn(cross_norm, encoder_out=encoder_out, pos=pos, query_pos=query_pos)
            x = x + attn_output

        mlp_norm = self.mlp_norm(x)
        mlp_output = self.mlp(mlp_norm)
        output = x + mlp_output
        return output

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1, use_cross_attention=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert (
            self.head_dim * num_heads == dim
        ), "Embedding dimension must be divisible by number of heads"

        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.use_cross_attention = use_cross_attention
        self.last_attn_weights = None
        self._debug_print_count = 0
        if use_cross_attention:
            self.cross_q_proj = nn.Linear(dim, dim)
            self.cross_k_proj = nn.Linear(dim, dim)
            self.cross_v_proj = nn.Linear(dim, dim)

    def forward(self, x, encoder_out=None, future_mask=None, pos=None, query_pos=None):
        B, T, C = x.size()
        if self.use_cross_attention and encoder_out is not None:
            S = encoder_out.size(1)
            q_in = x if query_pos is None else x + query_pos
            k_in = encoder_out if pos is None else encoder_out + pos
            cross_q = self.cross_q_proj(q_in).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            cross_k = self.cross_k_proj(k_in).reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            cross_v = self.cross_v_proj(encoder_out).reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            attn_weights = (cross_q @ cross_k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            if future_mask is not None:
                attn_weights = attn_weights.masked_fill(future_mask == 0, torch.finfo(attn_weights.dtype).min)
            attn_weights = F.softmax(attn_weights, dim=-1)
            self.last_attn_weights = attn_weights.detach()

            attn_weights = self.dropout(attn_weights)
            attn_output = (attn_weights @ cross_v).transpose(1, 2).reshape(B, T, C)


            output = self.out_proj(attn_output)
            return output

        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        if pos is None:
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qk = self.qkv_proj(x + pos).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qk[0], qk[1], qkv[2]

        attn_weights = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if future_mask is not None:
            attn_weights = attn_weights.masked_fill(future_mask == 0, torch.finfo(attn_weights.dtype).min)
        attn_weights = F.softmax(attn_weights, dim=-1)
        self.last_attn_weights = attn_weights.detach()

        cls_attn = attn_weights[:, :, 0, :]
        cls_act_attn = cls_attn[..., 2:]
        max_cls_act_attn = cls_act_attn.max(dim=-1).values
        topk_vals, topk_idxs = cls_act_attn.topk(min(10, cls_act_attn.shape[-1]), dim=-1)
        if self._debug_print_count % 100 == 0:
            print("Max attention:", max_cls_act_attn.mean().item(), flush=True)
            print("Topk vals:", topk_vals[0, 0].detach().cpu().tolist(), flush=True)
            print("Topk idxs:", topk_idxs[0, 0].detach().cpu().tolist(), flush=True)
        self._debug_print_count += 1

        attn_weights = self.dropout(attn_weights)
        attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, T, C)


        output = self.out_proj(attn_output)
        return output
