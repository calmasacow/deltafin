import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# K3_SHORTCONV=mulsum swaps the T=1 depthwise conv for an explicit 4-tap
# gather+multiply+sum. Default "conv1d" = the F.conv1d path this shipped with.
SHORTCONV = os.environ.get("K3_SHORTCONV", "mulsum")


class ShortConvolution(nn.Conv1d):
    """Depthwise causal conv1d with rolling [N, D, W] cache of the last W raw inputs.
    Semantics ported from fla/modules/conv/short_conv.py: step = roll cache left,
    insert x_t, y = act(sum(cache * weight)); prefill = causal conv with cache[..., 1:]
    as left history; final state = last W raw inputs."""

    def __init__(self, hidden_size, kernel_size, bias=False, activation='silu', **kw):
        super().__init__(hidden_size, hidden_size, kernel_size, groups=hidden_size,
                         bias=bias, padding=kernel_size - 1)
        self.hidden_size = hidden_size
        self.act = activation

    def forward(self, x, residual=None, mask=None, cache=None,
                output_final_state=False, cu_seqlens=None, **kw):
        # x: [B, T, D]
        B, T, D = x.shape
        W = self.kernel_size[0]
        if SHORTCONV == "mulsum" and T == 1 and cache is not None:
            # Decode step. The causal window the conv would see is exactly
            # [cache[1:], x], which is also exactly the next cache, so the
            # separate cat+slice that builds new_cache below is redundant.
            # 12,288 depthwise groups over a length-4 window is a grouped
            # convolution only in name: 0.63 ms -> 0.38 ms for the three convs.
            z = torch.cat([cache.to(torch.float32)[:, :, 1:],
                           x.reshape(B, D, 1).to(torch.float32)], dim=-1)  # [B,D,W]
            y = (z * self.weight.view(D, W).to(torch.float32)).sum(-1)     # [B,D]
            if self.act in ('silu', 'swish'):
                y = F.silu(y)
            y = y.view(B, 1, D).to(x.dtype)
            if residual is not None:
                y = y + residual
            return y, (z if output_final_state else None)
        w = self.weight.view(D, W).to(torch.float32)  # [D, W]
        xt = x.transpose(1, 2).to(torch.float32)      # [B, D, T]
        if cache is None:
            hist = xt.new_zeros(B, D, W - 1)
        else:
            hist = cache.to(torch.float32)[:, :, 1:]  # last W-1 raw inputs
        z = torch.cat([hist, xt], dim=-1)             # [B, D, W-1+T]
        y = F.conv1d(z, w.unsqueeze(1), groups=D)     # causal: [B, D, T]
        if self.act in ('silu', 'swish'):
            y = F.silu(y)
        y = y.transpose(1, 2).to(x.dtype)
        if residual is not None:
            y = y + residual
        new_cache = None
        if output_final_state:
            full = torch.cat([cache.to(torch.float32) if cache is not None
                              else xt.new_zeros(B, D, W), xt], dim=-1)
            new_cache = full[:, :, -W:].contiguous()
        return y, new_cache


class FusedRMSNormGated(nn.Module):
    """out = RMSNorm(x) * weight * sigmoid(g)  (fp32 internal), per fla fused_norm_gate.py."""

    def __init__(self, hidden_size, eps=1e-5, activation='sigmoid', **kw):
        super().__init__()
        assert activation == 'sigmoid'
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x, g):
        xf, gf = x.to(torch.float32), g.to(torch.float32)
        var = xf.pow(2).mean(-1, keepdim=True)
        y = xf * torch.rsqrt(var + self.variance_epsilon) * self.weight.to(torch.float32)
        return (y * torch.sigmoid(gf)).to(x.dtype)
