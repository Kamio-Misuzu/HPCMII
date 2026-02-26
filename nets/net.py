import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .encoder.pvtv2_encoder import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2
from .decoder.mmformer_decoder import Decoder_fuse

Encoder = pvt_v2_b0
Decoder = Decoder_fuse

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x_q, x_kv):
        """
        x_q: query tensor (B, N_q, C)
        x_kv: key-value tensor (B, N_kv, C)
        """
        B, N_q, C = x_q.shape
        N_kv = x_kv.shape[1]
        
        q = self.q(x_q).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (B, heads, N_q, head_dim)
        k = self.k(x_kv).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (B, heads, N_kv, head_dim)
        v = self.v(x_kv).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (B, heads, N_kv, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N_q, N_kv)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)  # (B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads=8, qkv_bias=False, qk_scale=None,
                 dropout_rate=0.0, ffn_ratio=4.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = CrossAttention(dim, heads, qkv_bias, qk_scale, dropout_rate)

        hidden_dim = int(dim * ffn_ratio)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x_q, x_kv):
        """
        x_q: (B, N_q, C)
        x_kv: (B, N_kv, C)
        """
        q = self.norm_q(x_q)
        kv = self.norm_kv(x_kv)
        attn_out = self.attn(q, kv)          # (B, N_q, C)
        x = x_q + attn_out                   # residual

        x = x + self.ffn(self.norm_ffn(x))   # (B, N_q, C)
        return x



class MaskModal(nn.Module):
    def __init__(self):
        super(MaskModal, self).__init__()
    
    def forward(self, x, mask):
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        x = y.view(B, -1, H, W, Z)
        return x

class Model_3D(nn.Module):
    def __init__(self, num_cls=4, transformer_dim=128, num_heads=8, dropout_rate=0.1):
        super(Model_3D, self).__init__()
        self.flair_encoder = Encoder()
        self.t1ce_encoder = Encoder()
        self.t1_encoder = Encoder()
        self.t2_encoder = Encoder()

        self.masker = MaskModal()
        
        self.t1_to_t1ce_cross_attn = CrossAttentionBlock(dim=transformer_dim, heads=num_heads, dropout_rate=dropout_rate)
        self.t1ce_to_t1_cross_attn = CrossAttentionBlock(dim=transformer_dim, heads=num_heads, dropout_rate=dropout_rate)
        
        self.flair_to_t2_cross_attn = CrossAttentionBlock(dim=transformer_dim, heads=num_heads, dropout_rate=dropout_rate)
        self.t2_to_flair_cross_attn = CrossAttentionBlock(dim=transformer_dim, heads=num_heads, dropout_rate=dropout_rate)
        
        self.flair_t2_to_t1_t1ce_cross_attn = CrossAttentionBlock(dim=transformer_dim*2, heads=num_heads, dropout_rate=dropout_rate)
        self.t1_t1ce_to_flair_t2_cross_attn = CrossAttentionBlock(dim=transformer_dim*2, heads=num_heads, dropout_rate=dropout_rate)
        
        self.x5_channel = nn.Conv3d(1024, 512, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.decoder = Decoder(num_cls=num_cls)
        
        self.is_training = False

    def forward(self, x, mask):
        flair_x1, flair_x2, flair_x3, flair_x4, flair_x5 = self.flair_encoder(x[:, 0:1, :, :, :])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5 = self.t1ce_encoder(x[:, 1:2, :, :, :])
        t1_x1, t1_x2, t1_x3, t1_x4, t1_x5 = self.t1_encoder(x[:, 2:3, :, :, :])
        t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.t2_encoder(x[:, 3:4, :, :, :])
        
        x1 = self.masker(torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1), mask)
        x2 = self.masker(torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1), mask)
        x3 = self.masker(torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1), mask)
        x4 = self.masker(torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1), mask)
        
        x5 = self.masker(torch.stack((flair_x5, t1ce_x5, t1_x5, t2_x5), dim=1), mask)  # (B, 512, 10, 12, 8)
        
        B, C, H, W, Z = x5.shape
        num_modalities = 4
        assert C % num_modalities == 0, \
            f"Expected channel dimension {C} to be divisible by {num_modalities} modalities."
        C_per_modal = C // num_modalities

        flair_x5_masked, t1ce_x5_masked, t1_x5_masked, t2_x5_masked = torch.chunk(x5, num_modalities, dim=1)

        def to_tokens(feat: torch.Tensor) -> torch.Tensor:
            return feat.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, C_per_modal)

        flair_token = to_tokens(flair_x5_masked)   # (B, N_tokens, 128)
        t1ce_token = to_tokens(t1ce_x5_masked)     # (B, N_tokens, 128)
        t1_token = to_tokens(t1_x5_masked)         # (B, N_tokens, 128)
        t2_token = to_tokens(t2_x5_masked)         # (B, N_tokens, 128)

        t1_fused = self.t1_to_t1ce_cross_attn(t1_token, t1ce_token)      # (B, N_tokens, 128)
        t1ce_fused = self.t1ce_to_t1_cross_attn(t1ce_token, t1_token)      # (B, N_tokens, 128)

        flair_fused = self.flair_to_t2_cross_attn(flair_token, t2_token)   # (B, N_tokens, 128)
        t2_fused = self.t2_to_flair_cross_attn(t2_token, flair_token)   # (B, N_tokens, 128)


        flair_t2_combined = torch.cat([flair_fused, t2_fused], dim=-1)   # (B, N_tokens, 256)
        t1_t1ce_combined = torch.cat([t1_fused, t1ce_fused], dim=-1)     # (B, N_tokens, 256)

        flair_t2_fuse = self.flair_t2_to_t1_t1ce_cross_attn(
            flair_t2_combined, t1_t1ce_combined
        )  # (B, N_tokens, 256)
        t1_t1ce_fuse = self.t1_t1ce_to_flair_t2_cross_attn(
            t1_t1ce_combined, flair_t2_combined
        )  # (B, N_tokens, 256)

        x5_token = torch.cat(
            [flair_t2_fuse, t1_t1ce_fuse],
            dim=-1,
        )  # (B, N_tokens, 1024)

        x5 = x5_token.view(B, H, W, Z, -1).permute(0, 4, 1, 2, 3).contiguous()  # (B, 1024, H, W, Z)

        
        pred, preds = self.decoder(x1, x2, x3, x4, x5)
        
        if self.is_training:
            return pred, preds
        return pred
        
    

def build_model(cfg=None, load_path=None, **kwargs):
    return Model_3D(**kwargs)
