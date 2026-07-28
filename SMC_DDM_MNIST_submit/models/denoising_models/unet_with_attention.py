import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from .utils.time_embeddings import SinusoidalPosEmb
from typing import Callable


class NonLinearity(nn.Module):
    def forward(self, x):
        # Swish
        return x * torch.sigmoid(x)

class Normalizaton(nn.Module):
    def __init__(self, num_channels, num_groups=32):
        super(Normalizaton, self).__init__()
        self.norm = nn.GroupNorm(num_channels=num_channels, num_groups=num_groups)

    def forward(self, x):
        return self.norm(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super(UpSample, self).__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        assert x.shape == (B, C, H * 2, W * 2)
        if self.with_conv:
            x = self.conv(x)
            assert x.shape == (B, C, H * 2, W * 2)
        return x
    
class DownSample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super(DownSample, self).__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0
        if self.with_conv:
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
        assert x.shape == (B, C, H // 2, W // 2)
        return x
    
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, temb_dim=64):
        super(ResNetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalizaton(in_channels)
        self.non_lin1 = NonLinearity()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.norm2 = Normalizaton(out_channels)
        self.non_lin2 = NonLinearity()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.temb_projection = nn.Linear(temb_dim, out_channels)

    def forward(self, input):
        x, temb = input
        B, C, H, W = x.shape
        h = x
        h = self.norm1(h)
        h = self.non_lin1(h)
        h = self.conv1(h)
        
        h = h + self.temb_projection(temb)[:, :, None, None]

        h = self.norm2(h)
        h = self.non_lin2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        assert x.shape == (B, self.out_channels, H, W)
        return (x + h, temb)
    

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.in_channels = in_channels

        self.norm = Normalizaton(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        x, temb = input
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        w = torch.einsum("bchw,bcij->bhwij", q, k) * (int(C) ** (-0.5)) # Shape (B, H, W, H, W)
        w = F.softmax(w.view(B, H, W, -1), dim=-1).view(B, H, W, H, W)
        w = torch.einsum("bhwij,bcij->bchw", w, v)

        h = self.proj_out(w)
        assert h.shape == x.shape
        return (x + h, temb)


class UNetWithAttention(nn.Module):
    def __init__(
        self, 
        num_categories,
        embedding_dim=64,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2, 
        attention_resolutions=(2, 4), 
        dropout=0.0, 
        resamp_with_conv=True,
        encode_time: bool = True,
        probs_parametrization_fn: Callable[[Tensor, Tensor], Tensor] = lambda logits, x: torch.softmax(logits, dim=-1),
    ):
        """
        Args:
            num_categories: Number of discrete categories for output channels.
            embedding_dim: Base channel multiplier.
            ch_mult: Channel multipliers per resolution level.
            num_res_blocks: Blocks per resolution.
            attention_resolutions: Levels at which to apply attention.
            dropout: Dropout rate in ResNet blocks.
            resamp_with_conv: Use conv layers for up/down sampling.
            probs_parametrization_fn: Function mapping logits and input to probabilities.
        """
        super(UNetWithAttention, self).__init__()
        self.encode_time = encode_time
        self.probs_parametrization_fn = probs_parametrization_fn
            
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        
        self.time_embedding = SinusoidalPosEmb(embedding_dim)
        self.temb_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            NonLinearity(),
            nn.Linear(4 * embedding_dim, 4 * embedding_dim),
            NonLinearity(),
        )
        
        self.initial_block = nn.Conv2d(embedding_dim, ch_mult[0] * embedding_dim, kernel_size=3, stride=1, padding=1)
        
        self.num_resolutions = len(ch_mult)
        
        self.down_blocks = nn.ModuleList([])
        for i_level in range(self.num_resolutions):
            block = []
            for i_block in range(num_res_blocks):
                if i_block == 0 and i_level != 0:
                    in_ch = ch_mult[i_level - 1] * embedding_dim
                else:
                    in_ch = ch_mult[i_level] * embedding_dim
                block.append(
                    ResNetBlock(
                        in_channels=in_ch,
                        out_channels=ch_mult[i_level] * embedding_dim,
                        dropout=dropout,
                        temb_dim=4 * embedding_dim
                    )
                )
                if i_level in attention_resolutions:
                    block.append(AttentionBlock(ch_mult[i_level] * embedding_dim))
            self.down_blocks.append(nn.Sequential(*block))
            if i_level != self.num_resolutions - 1:
                self.down_blocks.append(DownSample(ch_mult[i_level] * embedding_dim, resamp_with_conv))
                
        self.mid_block = nn.Sequential(
            ResNetBlock(
                in_channels=ch_mult[-1] * embedding_dim,
                out_channels=ch_mult[-1] * embedding_dim,
                dropout=dropout,
                temb_dim=4 * embedding_dim
            ),
            AttentionBlock(ch_mult[-1] * embedding_dim),
            ResNetBlock(
                in_channels=ch_mult[-1] * embedding_dim,
                out_channels=ch_mult[-1] * embedding_dim,
                dropout=dropout,
                temb_dim=4 * embedding_dim
            ),
        )
        
        self.up_blocks = nn.ModuleList([])
        for i_level in reversed(range(self.num_resolutions)):
            block = []
            for i_block in range(num_res_blocks):
                if i_level != 0:
                    out_ch = ch_mult[i_level - 1] * embedding_dim
                else:
                    out_ch = ch_mult[i_level] * embedding_dim
                if i_block == 0:
                    in_ch = 2 * ch_mult[i_level] * embedding_dim
                else:
                    in_ch = out_ch
                block.append(
                    ResNetBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        dropout=dropout,
                        temb_dim=4 * embedding_dim
                    )
                )
                if i_level in attention_resolutions:
                    block.append(AttentionBlock(out_ch))
            self.up_blocks.append(nn.Sequential(*block))
            if i_level != 0:
                self.up_blocks.append(UpSample(out_ch, resamp_with_conv))
        
        self.final_block = nn.Sequential(
            Normalizaton(ch_mult[0] * embedding_dim),
            NonLinearity(),
            nn.Conv2d(ch_mult[0] * embedding_dim, num_categories, kernel_size=3, stride=1, padding=1)
        )
        
        
    def _forward(self, x, t):
        """
        x: (B, C, H, W, num_categories)
        t: (B,)
        """
             
        # Timestep embedding
        temb = self.time_embedding(t)
        temb = self.temb_mlp(temb)
          
        x = x @ self.embedding.weight
        x = x.transpose(1, -1).squeeze(-1)
        B, C, H, W = x.shape
        
        # Downsampling
        h = self.initial_block(x)
        hs = [h]
        for i_level in range(self.num_resolutions):
            h, _ = self.down_blocks[2 * i_level]((h, temb))
            hs.append(h)
            # Downsample
            if i_level != self.num_resolutions - 1:
                h = self.down_blocks[2 * i_level + 1](h)
                
        # Middle
        h, _ = self.mid_block((h, temb))
        
        # Upsampling
        for i_level in range(self.num_resolutions):
            h, _ = self.up_blocks[2 * i_level]((torch.cat([h, hs.pop()], dim=1), temb))
            # Upsample
            if i_level != self.num_resolutions - 1:
                h = self.up_blocks[2 * i_level + 1](h)
                
        # End
        h = self.final_block(h)
        return h.unsqueeze(-1).transpose(1, -1)
    
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        x: (B, C, H, W, num_categories)
        t: (B,)
        """
        if not self.encode_time:
            t = t * 0
        
        logits = self._forward(x, t) # Shape: (B, C, H, W, num_categories)
        B, C, H, W, num_categories = logits.shape
        logits = logits.reshape(B, -1, num_categories)
        x = x.reshape(B, -1, num_categories)
        
        probs = self.probs_parametrization_fn(logits, x)        
        return probs.reshape(B, C, H, W, num_categories)
