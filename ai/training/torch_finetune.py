"""Native Windows/Linux CUDA training with the existing NPZ checkpoint format."""
import json
import math
import os
import time
from typing import List

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from ai.inference.hw_model import LSTM_SIZE, NUM_ATTN_MIX, NUM_OUT_MIX, WEIGHTS_PATH
from ai.training.devices import normalize_device
from ai.training.samples import find_missing_chars, load_profile_alphabet
from ai.training.training_log import TrainingLog


WEIGHT_KEYS = {
    'lstm1_k': 'rnn__LSTMAttentionCell__lstm_cell__kernel',
    'lstm1_b': 'rnn__LSTMAttentionCell__lstm_cell__bias',
    'lstm2_k': 'rnn__LSTMAttentionCell__lstm_cell_1__kernel',
    'lstm2_b': 'rnn__LSTMAttentionCell__lstm_cell_1__bias',
    'lstm3_k': 'rnn__LSTMAttentionCell__lstm_cell_2__kernel',
    'lstm3_b': 'rnn__LSTMAttentionCell__lstm_cell_2__bias',
    'attn_w': 'rnn__LSTMAttentionCell__attention__weights',
    'attn_b': 'rnn__LSTMAttentionCell__attention__biases',
    'gmm_w': 'rnn__gmm__weights',
    'gmm_b': 'rnn__gmm__biases',
}


@torch.jit.script
def _lstm_step(x: Tensor, h: Tensor, c: Tensor, kernel: Tensor, bias: Tensor):
    i, j, f, o = (torch.cat([x, h], dim=-1) @ kernel + bias).chunk(4, dim=-1)
    c = torch.sigmoid(f + 1.0) * c + torch.sigmoid(i) * torch.tanh(j)
    return torch.sigmoid(o) * torch.tanh(c), c


@torch.jit.script
def _forward_loss(offsets: Tensor, text: Tensor, weights: List[Tensor],
                  hidden_size: int, attention_mix: int, output_mix: int):
    k1, b1, k2, b2, k3, b3, aw, ab, gw, gb = weights
    targets = offsets[1:]
    inputs = torch.cat([torch.tensor([[0.0, 0.0, 1.0]], dtype=offsets.dtype,
                                    device=offsets.device), targets[:-1]], dim=0)
    h1 = c1 = h2 = c2 = h3 = c3 = offsets.new_zeros((1, hidden_size))
    kappa = offsets.new_zeros((1, attention_mix))
    window = offsets.new_zeros((1, text.size(1)))
    positions = torch.arange(text.size(0), dtype=offsets.dtype, device=offsets.device)[None, :]
    hidden = torch.jit.annotate(List[Tensor], [])
    for t in range(targets.size(0)):
        x = inputs[t][None, :]
        h1, c1 = _lstm_step(torch.cat([window, x], dim=-1), h1, c1, k1, b1)
        alpha, beta, delta = F.softplus(torch.cat([window, x, h1], dim=-1) @ aw + ab).chunk(3, dim=-1)
        kappa = kappa + delta / 25.0
        beta = beta.clamp(0.01, 1e6)
        phi = (alpha[:, :, None] * torch.exp(
            -(kappa[:, :, None] - positions[:, None, :]).square() / beta[:, :, None]
        )).sum(dim=1)
        window = phi @ text
        h2, c2 = _lstm_step(torch.cat([x, h1, window], dim=-1), h2, c2, k2, b2)
        h3, c3 = _lstm_step(torch.cat([x, h2, window], dim=-1), h3, c3, k3, b3)
        hidden.append(h3)
    # The output head does not feed recurrent state. Evaluate all timesteps in
    # one matrix multiplication and density calculation, keeping the same
    # teacher forcing, float32 clamps, mean loss, and per-sample Adam updates.
    pis, sigmas, rhos, mus, es = torch.split(
        torch.cat(hidden, dim=0) @ gw + gb,
        [output_mix, 2 * output_mix, output_mix, 2 * output_mix, 1], dim=-1
    )
    pis = F.log_softmax(pis, dim=-1)
    sigmas = sigmas.clamp(math.log(1e-4), math.log(1e3)).exp()
    rhos = rhos.tanh().clamp(-1 + 1e-6, 1 - 1e-6)
    es = es.sigmoid().clamp(1e-6, 1 - 1e-6)
    mu1, mu2 = mus[:, :output_mix], mus[:, output_mix:]
    s1, s2 = sigmas[:, :output_mix], sigmas[:, output_mix:]
    tx, ty, teos = targets[:, 0:1], targets[:, 1:2], targets[:, 2]
    z1, z2 = (tx - mu1) / s1, (ty - mu2) / s2
    zz = z1.square() + z2.square() - 2 * rhos * z1 * z2
    norm = 2 * math.pi * s1 * s2 * (1 - rhos.square()).sqrt()
    log_gauss = -zz / (2 * (1 - rhos.square())) - norm.log()
    log_mix = torch.logsumexp(pis + log_gauss, dim=-1)
    eos_term = teos * es[:, 0].log() + (1 - teos) * (1 - es[:, 0]).log()
    return -(log_mix + eos_term).mean()


class TorchHandwritingModel(torch.nn.Module):
    backend = 'torch'

    def __init__(self, alphabet=None, weights_path=WEIGHTS_PATH, seed=0, device='auto'):
        super().__init__()
        request = normalize_device(device)
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        index = int(request.split(':')[1]) if ':' in request else 0
        if request.startswith('cuda') and index >= count:
            raise ValueError(f'Requested GPU {index}, but PyTorch exposes {count} GPU(s). '
                             'Install CUDA-enabled PyTorch and check NVIDIA drivers and CUDA_VISIBLE_DEVICES.')
        self.device = f'cuda:{index}' if request != 'cpu' and count else 'cpu'
        self.device_name = torch.cuda.get_device_name(index) if self.device != 'cpu' else 'CPU'
        source = load_profile_alphabet(weights_path)
        self.alphabet = list(source if alphabet is None else alphabet)
        if self.alphabet[:len(source)] != source or len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError("The new alphabet must preserve the base profile's character order.")
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}
        self.A = len(self.alphabet)
        with np.load(weights_path) as checkpoint:
            weights = {name: checkpoint[key].astype(np.float32) for name, key in WEIGHT_KEYS.items()}
        if weights['lstm1_k'].shape[0] != len(source) + 3 + LSTM_SIZE:
            raise ValueError('Profile weights and alphabet do not match.')
        rng = np.random.default_rng(seed)
        extra = self.A - len(source)
        if extra:
            for name, start in [('lstm1_k', 0), ('attn_w', 0),
                                ('lstm2_k', 3 + LSTM_SIZE), ('lstm3_k', 3 + LSTM_SIZE)]:
                kernel = weights[name]
                end = start + len(source)
                rows = rng.normal(scale=0.01, size=(extra, kernel.shape[1])).astype(np.float32)
                weights[name] = np.concatenate([kernel[:end], rows, kernel[end:]], axis=0)
        for name, array in weights.items():
            setattr(self, name, torch.nn.Parameter(torch.tensor(array, device=self.device)))
        self.trainable_variables = [getattr(self, name) for name in WEIGHT_KEYS]

    def encode_text(self, text):
        indices = [self.char_to_idx[c] for c in text] + [0]
        onehot = np.zeros((len(indices), self.A), dtype=np.float32)
        onehot[np.arange(len(indices)), indices] = 1.0
        return onehot

    def forward_loss(self, offsets, onehot_text):
        offsets = torch.as_tensor(offsets, dtype=torch.float32, device=self.device)
        text = torch.as_tensor(onehot_text, dtype=torch.float32, device=self.device)
        return _forward_loss(offsets, text, self.trainable_variables, LSTM_SIZE, NUM_ATTN_MIX, NUM_OUT_MIX)

    def export_to_npz(self, path):
        np.savez(path, **{key: getattr(self, name).detach().cpu().numpy() for name, key in WEIGHT_KEYS.items()})
        with open(os.path.splitext(path)[0] + '.alphabet.json', 'w', encoding='utf-8') as stream:
            json.dump(self.alphabet, stream)


def fit(model, samples, epochs=10, lr=1e-5, log_every=1, progress_cb=None, sample_progress_cb=None):
    if not samples:
        raise ValueError('No usable handwriting samples.')
    if epochs < 1 or log_every < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError('Epochs, logging interval and learning rate must be positive.')
    missing = find_missing_chars(samples, model.alphabet)
    if missing:
        raise ValueError('Sample characters missing from the profile alphabet: ' + ' '.join(missing))
    prepared = []
    for offsets, text in samples:
        offsets = np.asarray(offsets, dtype=np.float32)
        if (offsets.ndim != 2 or offsets.shape[1] != 3 or len(offsets) < 2
                or not np.isfinite(offsets).all() or not np.isin(offsets[:, 2], [0, 1]).all()):
            raise ValueError('Each sample must contain finite [dx, dy, end-of-stroke] offsets.')
        prepared.append((torch.as_tensor(offsets, device=model.device),
                         torch.as_tensor(model.encode_text(text), device=model.device)))
    # Match TensorFlow Adam's epsilon and avoid version-specific optimizer defaults.
    optimizer = torch.optim.Adam(model.trainable_variables, lr=lr, eps=1e-7)
    history = []
    training_log = TrainingLog(model, samples, epochs, lr, log_every, sample_progress_cb)
    started = training_log.started
    with training_log:
        for epoch in range(epochs):
            epoch_loss = 0.0
            for sample_index, (offsets, text) in enumerate(prepared, 1):
                optimizer.zero_grad(set_to_none=True)
                loss = model.forward_loss(offsets, text)
                if not torch.isfinite(loss).item():
                    raise ValueError('Training loss is not finite; check handwriting samples and learning rate.')
                loss.backward()
                if any(weight.grad is None for weight in model.trainable_variables):
                    raise ValueError('Training graph disconnected from profile weights.')
                torch.nn.utils.clip_grad_norm_(model.trainable_variables, 5.0, error_if_nonfinite=True)
                optimizer.step()
                epoch_loss += loss.detach().item()
                training_log.sample(epoch, sample_index, epoch_loss)
            epoch_loss /= len(samples)
            history.append(epoch_loss)
            elapsed = time.monotonic() - started
            eta = elapsed / (epoch + 1) * (epochs - epoch - 1)
            if progress_cb:
                progress_cb(epoch, epochs, epoch_loss, elapsed, eta)
            training_log.epoch(epoch, epoch_loss, elapsed, eta)
        training_log.finish(history, time.monotonic() - started)
    return history
