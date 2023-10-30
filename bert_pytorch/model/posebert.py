# Copyright (C) 2021-2022 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

# Modified version of https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange, repeat
from .encoder import GCNEncoder


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class FeedForwardResidual(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., out_dim=15 * 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)

    def forward(self, x, init, n_iter=1):
        pred_pose = init
        for _ in range(n_iter):
            xf = torch.cat([x, init], -1)
            pred_pose = pred_pose + self.net(xf)
        return pred_pose

class ClassifierLayer(nn.Module):
    def __init__(self, hidden_dim, classes=120):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, classes),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        """
        Args:
            - x: [batch_size,seq_len,dim]
            - mask: [batch_size,seq_len] - dytpe= torch.bool - default True everywhere, if False it means that we don't pay attention to this timestep
        """
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # [B,H,T,T]
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:  # always true
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask.unsqueeze(1).unsqueeze(1).repeat(1, 1, n, 1)  # updating masked timesteps with context
            dots.masked_fill_(~mask, mask_value)  # ~ do the opposite i.e. move True to False here
            del mask
        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x, start=0):
        x = x + self.pe[:, start:(start + x.size(1))]
        return x


class TransformerRegressor(nn.Module):

    def __init__(self, dim, depth=2, heads=8, dim_head=32, mlp_dim=32, dropout=0.1, out=[22 * 6, 3],
                 share_regressor=False):
        super().__init__()

        self.layers = nn.ModuleList([])
        for i in range(depth):
            list_modules = [
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]

            # Regressor
            if i == 0 or not share_regressor:
                # N regressor per layer
                for out_i in out:
                    list_modules.append(PreNorm(dim, FeedForwardResidual(dim, mlp_dim, dropout=dropout, out_dim=out_i)))
            else:
                # Share regressor across layers
                for j in range(2, len(self.layers[0])):
                    list_modules.append(self.layers[0][j])
            self.layers.append(nn.ModuleList(list_modules))

        self.classifier = ClassifierLayer(dim)

    def forward(self, x, init, mask=None):
        batch_size, seq_len, *_ = x.size()
        y = init
        for layers_i in self.layers:
            # attention and feeforward module
            attn, ff = layers_i[0], layers_i[1]
            x = attn(x, mask=mask) + x
            x = ff(x) + x

            # N regressors
            for j, reg in enumerate(layers_i[2:]):
                y[j] = reg(x, init=y[j], n_iter=1)

        return y[0]


class FC_Embbed(nn.Module):
    def __init__(self, in_kp=15, in_chns=2, window=1, out_features=512):
        super(FC_Embbed, self).__init__()
        self.fc = nn.Linear(in_kp*in_chns, out_features)

    def forward(self, x):
        x = x.view(*x.shape[:-2], -1)
        x = self.fc(x)
        return x

class PoseBERT(nn.Module):
    def __init__(self,
                 out_kp=15, out_chns=2, init_pose=None, window=1,
                 depth=4, heads=8, dropout=0.1, share_regressor=1, 
                 embedding='FC_Embbed', EMB_ARGS={'out_features': 512},
                 *args, **kwargs):
        super(PoseBERT, self).__init__()
        self.out_kp = out_kp
        self.out_chns = out_chns
        self.window = window

        self.pos = PositionalEncoding(EMB_ARGS.out_features, 1024)
        self.emb = eval(embedding)(**EMB_ARGS)
        self.mask_token = nn.Parameter(torch.randn(1, 1, EMB_ARGS.out_features))
        dim_head = EMB_ARGS.out_features//heads

        self.decoder = TransformerRegressor(EMB_ARGS.out_features, depth, heads, 
                                            dim_head, dim_head, dropout, 
                                            [out_kp * out_chns * window],
                                            share_regressor == 1)

        if init_pose is None:
            init_pose = torch.zeros(out_kp * out_chns * window).float()
        self.register_buffer('init_pose', init_pose.reshape(1, 1, -1))

    def forward(self, x, padding=None):
        """
        Args:
            - x: 
        Return:
            - y: 
        """

        # Input embedding
        x = self.emb(x)
        x = self.pos(x)  # inject position info

        batch_size, seq_len, *_ = x.size()
        # mask = None
        mask = torch.arange(x.size(1), device=x.device)
        mask = mask.expand(x.size(0), x.size(1)) 
        mask = mask >= padding.unsqueeze(1)

        # Transformer
        init = [self.init_pose.repeat(batch_size, seq_len, 1)]  # init mean pose
        y = self.decoder(x, init, mask)
        y = y.view(y.shape[0], y.shape[1] * self.window, self.out_kp, self.out_chns)

        return y
    
    