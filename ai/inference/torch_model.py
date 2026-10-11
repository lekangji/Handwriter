"""Device-resident handwriting inference; synchronize once per block, not per point."""
from typing import List
from collections import OrderedDict
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from ai.inference.hw_model import HandwritingModel, WEIGHTS_PATH


@torch.jit.script
def _cell(x: Tensor, h: Tensor, c: Tensor, kernel: Tensor, bias: Tensor):
    i, j, f, o = (torch.cat([x, h]) @ kernel + bias).chunk(4)
    c = torch.sigmoid(f + 1.0) * c + torch.sigmoid(i) * torch.tanh(j)
    return torch.sigmoid(o) * torch.tanh(c), c


@torch.jit.script
def _advance(text: Tensor, weights: List[Tensor], state: List[Tensor]):
    k1, b1, k2, b2, k3, b3, aw, ab, gw, gb = weights
    h1, c1, h2, c2, h3, c3, kappa, window, x, dwell, terminal = state
    h1, c1 = _cell(torch.cat([window, x]), h1, c1, k1, b1)
    alpha, beta, delta = F.softplus(torch.cat([window, x, h1]) @ aw + ab).chunk(3)
    kappa = kappa + delta / 25.0
    positions = torch.arange(text.size(0), dtype=text.dtype, device=text.device)
    phi = (alpha[:, None] * torch.exp(-(kappa[:, None] - positions[None, :]).square()
                                    / beta.clamp_min(0.01)[:, None])).sum(0)
    window = phi @ text
    h2, c2 = _cell(torch.cat([x, h1, window]), h2, c2, k2, b2)
    h3, c3 = _cell(torch.cat([x, h2, window]), h3, c3, k3, b3)
    return [h1, c1, h2, c2, h3, c3, kappa, window, x, dwell, terminal], phi, h3 @ gw + gb


@torch.jit.script
def _prime(text: Tensor, weights: List[Tensor], state: List[Tensor], points: Tensor):
    for i in range(1, points.size(0)):
        state, _, _ = _advance(text, weights, state)
        state[8] = points[i]
    return state


@torch.jit.script
def _draw_block(text: Tensor, weights: List[Tensor], state: List[Tensor], randoms: Tensor,
                bias: float, max_dwell: int):
    draws = torch.jit.annotate(List[Tensor], [])
    checks = torch.jit.annotate(List[Tensor], [])
    for i in range(randoms.size(0)):
        state, phi, params = _advance(text, weights, state)
        pis, sigmas, rhos, mus, es = params.split([20, 40, 20, 40, 1])
        pis = torch.softmax(pis * (1.0 + bias), dim=0)
        pis = torch.where(pis < 0.01, torch.zeros_like(pis), pis)
        pis = pis / pis.sum()
        sigmas = (sigmas - bias).clamp(-9.0, 7.0).exp()
        rhos = rhos.tanh().clamp(-0.999999, 0.999999)
        k = (pis.cumsum(0) < randoms[i, 0]).sum().clamp_max(19).reshape(1)
        rho = rhos.gather(0, k)[0]
        dx = mus[:20].gather(0, k)[0] + sigmas[:20].gather(0, k)[0] * randoms[i, 1]
        dy = mus[20:].gather(0, k)[0] + sigmas[20:].gather(0, k)[0] * (
            rho * randoms[i, 1] + (1.0 - rho.square()).sqrt() * randoms[i, 2])
        eos = (randoms[i, 3] < es[0].sigmoid()).to(text.dtype)
        x = torch.stack([dx, dy, eos])
        position = phi.argmax().reshape(1)
        dwell = state[9].scatter_add(0, position, torch.ones_like(position, dtype=text.dtype))
        at_end = position[0] == text.size(0) - 1
        terminal = torch.where(at_end, state[10] + 1, torch.zeros_like(state[10]))
        state[8], state[9], state[10] = x, dwell, terminal
        finite = torch.isfinite(params).all() & torch.isfinite(phi).all() & torch.isfinite(x).all()
        valid = finite & (dwell.gather(0, position)[0] <= max_dwell)
        complete = at_end & ((eos > 0) | (terminal >= 8))
        draws.append(x)
        checks.append(torch.stack([position[0].to(text.dtype), valid.to(text.dtype), complete.to(text.dtype)]))
    return state, torch.stack(draws), torch.stack(checks)


class TorchInferenceModel(HandwritingModel):
    backend = 'torch'

    def __init__(self, weights_path=WEIGHTS_PATH, alphabet=None, device='cuda:0'):
        super().__init__(weights_path, alphabet)
        self.device = str(torch.device(device))
        self.device_name = torch.cuda.get_device_name(self.device) if self.device.startswith('cuda') else 'CPU'
        arrays = [self.lstm1.kernel, self.lstm1.bias, self.lstm2.kernel, self.lstm2.bias,
                  self.lstm3.kernel, self.lstm3.bias, self.attn_w, self.attn_b, self.gmm_w, self.gmm_b]
        self.weights = [torch.as_tensor(array, dtype=torch.float32, device=self.device) for array in arrays]
        self._graphs = OrderedDict()

    def initial_state(self, text_length=1):
        zero = self.weights[0].new_zeros
        return [zero(400), zero(400), zero(400), zero(400), zero(400), zero(400),
                zero(10), zero(len(self.alphabet)),
                self.weights[0].new_tensor([0., 0., 1.]), zero(text_length), zero(())]

    def _block(self, text, state, randoms, bias, max_dwell):
        if not self.device.startswith('cuda'):
            return _draw_block(text, self.weights, state, randoms, bias, max_dwell)
        key = (text.size(0), bias, max_dwell)
        if key not in self._graphs:
            # Replaying a graph replaces thousands of tiny host kernel launches with one.
            static_text = text.clone()
            static_state = [value.clone() for value in state]
            static_randoms = randoms.clone()
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    _draw_block(static_text, self.weights, static_state, static_randoms, bias, max_dwell)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.device(self.device), torch.cuda.graph(graph, stream=stream):
                result = _draw_block(static_text, self.weights, static_state, static_randoms, bias, max_dwell)
            self._graphs[key] = (graph, static_text, static_state, static_randoms, result)
            if len(self._graphs) > 16:
                self._graphs.popitem(last=False)
        self._graphs.move_to_end(key)
        graph, static_text, static_state, static_randoms, result = self._graphs[key]
        static_text.copy_(text)
        for destination, source in zip(static_state, state):
            destination.copy_(source)
        static_randoms.copy_(randoms)
        graph.replay()
        return result

    def _sample_once(self, onehot, bias, rng, prime_strokes, max_steps, steps_per_char, prefix_chars=0):
        # Host checks happen once per 32 steps; matrices and recurrent state remain on the GPU.
        with torch.inference_mode():
            text = torch.as_tensor(onehot, dtype=torch.float32, device=self.device)
            state = self.initial_state(len(onehot))
            if prime_strokes is not None:
                state = _prime(text, self.weights, state,
                               torch.as_tensor(prime_strokes, device=self.device))
            transition = max(200, min(2000, steps_per_char * prefix_chars)) if prefix_chars else 0
            total_budget = max_steps + transition
            out = np.empty((max_steps, 3), np.float32)
            count, consumed = 0, 0
            while consumed < total_budget:
                size = min(32, total_budget - consumed)
                block_size = 32 if self.device.startswith('cuda') else size
                randoms = np.column_stack([rng.random(block_size), rng.standard_normal((block_size, 2)), rng.random(block_size)]).astype(np.float32)
                state, draws, checks = self._block(text, state,
                    torch.as_tensor(randoms, device=self.device), float(bias), max(80, 2 * steps_per_char))
                # One small transfer per block, rather than synchronizing every attention/sample decision.
                block = torch.cat([draws, checks], dim=1).cpu().numpy()
                consumed += size
                for row in block[:size]:
                    if not row[4]:
                        return out[:count].copy(), False
                    if row[3] < prefix_chars:
                        transition -= 1
                        if transition <= 0:
                            return out[:count].copy(), False
                        continue
                    out[count] = row[:3]
                    count += 1
                    if row[5]:
                        return out[:count].copy(), True
                    if count >= max_steps:
                        return out[:count].copy(), False
            return out[:count].copy(), False
