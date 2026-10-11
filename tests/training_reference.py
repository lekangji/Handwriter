"""Frozen pre-optimization PyTorch loss for numerical and timing comparisons.

Keep this independent of the optimized implementation so recurrent state,
target alignment, density clamps, and gradients have a regression oracle.
"""
import math
from typing import List

import torch
from torch import Tensor
from torch.nn import functional as F


@torch.jit.script
def reference_lstm(x: Tensor, h: Tensor, c: Tensor, kernel: Tensor, bias: Tensor):
    i, j, f, o = (torch.cat([x, h], dim=-1) @ kernel + bias).chunk(4, dim=-1)
    c = torch.sigmoid(f + 1.0) * c + torch.sigmoid(i) * torch.tanh(j)
    return torch.sigmoid(o) * torch.tanh(c), c


@torch.jit.script
def reference_loss(offsets: Tensor, text: Tensor, weights: List[Tensor],
                   hidden_size: int, attention_mix: int, output_mix: int):
    k1, b1, k2, b2, k3, b3, aw, ab, gw, gb = weights
    targets = offsets[1:]
    inputs = torch.cat([torch.tensor([[0.0, 0.0, 1.0]], dtype=offsets.dtype,
                                    device=offsets.device), targets[:-1]], dim=0)
    h1 = c1 = h2 = c2 = h3 = c3 = offsets.new_zeros((1, hidden_size))
    kappa = offsets.new_zeros((1, attention_mix))
    window = offsets.new_zeros((1, text.size(1)))
    positions = torch.arange(text.size(0), dtype=offsets.dtype, device=offsets.device)[None, :]
    total = offsets.new_zeros(())
    for t in range(targets.size(0)):
        x = inputs[t][None, :]
        h1, c1 = reference_lstm(torch.cat([window, x], dim=-1), h1, c1, k1, b1)
        alpha, beta, delta = F.softplus(torch.cat([window, x, h1], dim=-1) @ aw + ab).chunk(3, dim=-1)
        kappa = kappa + delta / 25.0
        beta = beta.clamp(0.01, 1e6)
        phi = (alpha[:, :, None] * torch.exp(
            -(kappa[:, :, None] - positions[:, None, :]).square() / beta[:, :, None]
        )).sum(dim=1)
        window = phi @ text
        h2, c2 = reference_lstm(torch.cat([x, h1, window], dim=-1), h2, c2, k2, b2)
        h3, c3 = reference_lstm(torch.cat([x, h2, window], dim=-1), h3, c3, k3, b3)
        pis, sigmas, rhos, mus, es = torch.split(
            h3 @ gw + gb, [output_mix, 2 * output_mix, output_mix, 2 * output_mix, 1], dim=-1
        )
        pis = F.log_softmax(pis, dim=-1)
        sigmas = sigmas.clamp(math.log(1e-4), math.log(1e3)).exp()
        rhos = rhos.tanh().clamp(-1 + 1e-6, 1 - 1e-6)
        es = es.sigmoid().clamp(1e-6, 1 - 1e-6)
        mu1, mu2 = mus[:, :output_mix], mus[:, output_mix:]
        s1, s2 = sigmas[:, :output_mix], sigmas[:, output_mix:]
        tx, ty, teos = targets[t, 0], targets[t, 1], targets[t, 2]
        z1, z2 = (tx - mu1) / s1, (ty - mu2) / s2
        zz = z1.square() + z2.square() - 2 * rhos * z1 * z2
        norm = 2 * math.pi * s1 * s2 * (1 - rhos.square()).sqrt()
        log_gauss = -zz / (2 * (1 - rhos.square())) - norm.log()
        log_mix = torch.logsumexp(pis + log_gauss, dim=-1)
        eos_term = teos * es[:, 0].log() + (1 - teos) * (1 - es[:, 0]).log()
        total = total - (log_mix[0] + eos_term[0])
    return total / targets.size(0)
