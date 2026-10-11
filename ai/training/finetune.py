"""
Fine-tuning pipeline for personalizing the handwriting model on a user's
own samples (PRD section 5: "a profile should not simply function as a
handwriting font").

Why this is a separate implementation from hw_model.py's fast NumPy
inference: training needs gradients, and the pretrained checkpoint's
weight matrices are sized for exactly the 73-character ALPHABET in
hw_model.py. To support characters the base model never saw (uppercase
Q/X/Z, `&`, arrows, etc.) this module can *grow* the alphabet and, with
it, the input dimension of every weight matrix that consumes the
attention window vector — copying over the pretrained rows for existing
characters and randomly initializing new rows for new ones (the same
technique used to extend a token vocabulary in NLP fine-tuning). This is
real surgery on the weights, not a config flag, and it's why a from-
scratch model can't just be swapped in.

After training, call `export_to_npz()` to write weights hw_model.py can
load directly for fast inference — same file format, so nothing else in
the pipeline (layout, rendering, gcode) needs to change.

Honesty about what this can and can't do:
  - Extending letters/punctuation the model already half-understands
    (it saw ~73 chars broadly, including cursive joining behavior) will
    pick up a personal style reasonably fast, plausibly in the hundreds-
    of-words range, consistent with how small Graves-style fine-tunes
    behave in practice.
  - Genuinely novel glyphs with no cursive precedent (arrows, most
    symbols) have nothing to transfer from — the new rows start close to
    random, so quality depends heavily on how many repeated examples of
    that exact glyph you provide. Expect it to need many more repetitions
    than a letter does, and expect a rougher result even then.
  - This is standard supervised sequence learning, not few-shot magic:
    more varied, cleanly-captured samples reliably beats fewer.
"""

import json
import os
import time

import numpy as np
# Hide native informational startup messages while retaining warnings/errors.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')
try:
    import tensorflow as tf
except ImportError:
    tf = None

from ai.inference.hw_model import ALPHABET as BASE_ALPHABET
from ai.inference.hw_model import LSTM_SIZE, NUM_ATTN_MIX, NUM_OUT_MIX, WEIGHTS_PATH


from ai.training.samples import build_alphabet, find_missing_chars, load_profile_alphabet, load_training_samples
from ai.training.devices import configure_device, normalize_device
from ai.training.training_log import TrainingLog


def create_training_model(alphabet=None, weights_path=WEIGHTS_PATH, seed=0, device='auto'):
    """Prefer PyTorch CUDA, including on native Windows; retain CPU/TF support."""
    device = normalize_device(device)
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and (tf is None or (device != 'cpu' and torch.cuda.is_available())):
        from ai.training.torch_finetune import TorchHandwritingModel
        return TorchHandwritingModel(alphabet, weights_path, seed, device)
    if os.name == 'nt' and device.startswith('cuda'):
        raise ValueError('Native Windows CUDA training requires CUDA-enabled PyTorch. '
                         'Install requirements-cuda.txt in the app environment, then restart the app.')
    return TrainableHandwritingModel(alphabet, weights_path, seed, device)


class TrainableHandwritingModel:
    def __init__(self, alphabet=None, weights_path: str = WEIGHTS_PATH, seed=0, device="auto"):
        if tf is None:
            raise ImportError("TensorFlow is required for fine-tuning; sample capture and validation work without it.")
        self.device = configure_device(device, tf)
        source_alphabet = load_profile_alphabet(weights_path)
        self.alphabet = list(source_alphabet if alphabet is None else alphabet)
        if self.alphabet[:len(source_alphabet)] != source_alphabet or len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError("The new alphabet must preserve the base profile's character order.")
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}
        self.A = len(self.alphabet)
        rng = np.random.default_rng(seed)

        w = np.load(weights_path)
        g = lambda name: w[name.replace("/", "__")]

        def grow_rows(kernel, old_a, new_a, w_block_start, w_block_len=None):
            """Insert (new_a-old_a) freshly-initialized rows into `kernel`
            at the position where the window-vector block lives, preserving
            every other row exactly as pretrained."""
            w_block_len = w_block_len or old_a
            if new_a == old_a:
                return kernel
            extra_rows = rng.normal(scale=0.01, size=(new_a - old_a, kernel.shape[1])).astype(np.float32)
            head = kernel[: w_block_start + w_block_len]
            tail = kernel[w_block_start + w_block_len :]
            # split head into the w-block itself (last `w_block_len` rows of head) and what precedes it
            pre = head[:w_block_start]
            w_block = head[w_block_start : w_block_start + w_block_len]
            return np.concatenate([pre, w_block, extra_rows, tail], axis=0)

        old_a = len(source_alphabet)
        k1 = g("rnn/LSTMAttentionCell/lstm_cell/kernel")  # rows: w(A), x(3), h1_prev(400)
        if k1.shape[0] != old_a + 3 + LSTM_SIZE:
            raise ValueError("Profile weights and alphabet do not match.")
        k1 = grow_rows(k1, old_a, self.A, w_block_start=0, w_block_len=old_a)
        aw = g("rnn/LSTMAttentionCell/attention/weights")  # rows: w(A), x(3), h1(400)
        aw = grow_rows(aw, old_a, self.A, w_block_start=0, w_block_len=old_a)
        k2 = g("rnn/LSTMAttentionCell/lstm_cell_1/kernel")  # rows: x(3), h1(400), w(A), h2_prev(400)
        k2 = grow_rows(k2, old_a, self.A, w_block_start=403, w_block_len=old_a)
        k3 = g("rnn/LSTMAttentionCell/lstm_cell_2/kernel")  # rows: x(3), h2(400), w(A), h3_prev(400)
        k3 = grow_rows(k3, old_a, self.A, w_block_start=403, w_block_len=old_a)

        def tv(x, name):
            with tf.device(self.device):
                return tf.Variable(x.astype(np.float32), name=name)

        self.lstm1_k, self.lstm1_b = tv(k1, "lstm1_k"), tv(g("rnn/LSTMAttentionCell/lstm_cell/bias"), "lstm1_b")
        self.lstm2_k, self.lstm2_b = tv(k2, "lstm2_k"), tv(g("rnn/LSTMAttentionCell/lstm_cell_1/bias"), "lstm2_b")
        self.lstm3_k, self.lstm3_b = tv(k3, "lstm3_k"), tv(g("rnn/LSTMAttentionCell/lstm_cell_2/bias"), "lstm3_b")
        self.attn_w, self.attn_b = tv(aw, "attn_w"), tv(g("rnn/LSTMAttentionCell/attention/biases"), "attn_b")
        self.gmm_w, self.gmm_b = tv(g("rnn/gmm/weights"), "gmm_w"), tv(g("rnn/gmm/biases"), "gmm_b")

        self.trainable_variables = [
            self.lstm1_k,
            self.lstm1_b,
            self.lstm2_k,
            self.lstm2_b,
            self.lstm3_k,
            self.lstm3_b,
            self.attn_w,
            self.attn_b,
            self.gmm_w,
            self.gmm_b,
        ]
        w.close()

    def encode_text(self, text):
        idx = [self.char_to_idx[c] for c in text] + [0]
        U = len(idx)
        onehot = np.zeros((U, self.A), dtype=np.float32)
        onehot[np.arange(U), idx] = 1.0
        return onehot

    def _lstm_step(self, x, h_prev, c_prev, kernel, bias):
        z = tf.matmul(tf.concat([x, h_prev], axis=-1), kernel) + bias
        i, j, f, o = tf.split(z, 4, axis=-1)
        c = tf.sigmoid(f + 1.0) * c_prev + tf.sigmoid(i) * tf.tanh(j)
        h = tf.sigmoid(o) * tf.tanh(c)
        return h, c

    def forward_loss(self, offsets, onehot_text):
        """Teacher-forced NLL loss over one (points, text) training example,
        exactly mirroring the original paper's training objective: at each
        step, predict the *next* pen offset as a 20-component bivariate
        Gaussian mixture (+ Bernoulli end-of-stroke), given everything up to
        the current real point. This is standard MDN sequence training.
        offsets: [T,3] numpy (dx,dy,eos). onehot_text: [U,A] numpy.

        TensorFlow traces one dynamic loop for different sample lengths,
        avoiding a separate graph containing thousands of unrolled steps
        for each captured phrase.
        """
        with tf.device(getattr(self, "device", "/CPU:0")):
            return self._forward_loss_impl(offsets, onehot_text)

    @(tf.function(reduce_retracing=True) if tf is not None else lambda fn: fn)
    def _forward_loss_impl(self, offsets, onehot_text):
        offsets = tf.convert_to_tensor(offsets, dtype=tf.float32)
        # The first capture point is the origin, not a motion to predict.
        targets = offsets[1:]
        onehot_text = tf.convert_to_tensor(onehot_text, dtype=tf.float32)
        T = tf.shape(targets)[0]
        U = tf.shape(onehot_text)[0]
        inputs = tf.concat([tf.constant([[0.0, 0.0, 1.0]]), targets[:-1]], axis=0)

        h1 = c1 = h2 = c2 = h3 = tf.zeros([1, LSTM_SIZE])
        c3 = tf.zeros([1, LSTM_SIZE])
        kappa = tf.zeros([1, NUM_ATTN_MIX])
        w = tf.zeros([1, self.A])
        u_range = tf.range(U, dtype=tf.float32)[None, :]

        total_nll = tf.constant(0.0)
        for t in tf.range(T):
            x = inputs[t][None, :]
            s1_in = tf.concat([w, x], axis=-1)
            h1, c1 = self._lstm_step(s1_in, h1, c1, self.lstm1_k, self.lstm1_b)

            attn_in = tf.concat([w, x, h1], axis=-1)
            attn_params = tf.matmul(attn_in, self.attn_w) + self.attn_b
            alpha, beta, kdelta = tf.split(tf.math.softplus(attn_params), 3, axis=-1)
            kappa = kappa + kdelta / 25.0
            beta = tf.clip_by_value(beta, 0.01, 1e6)

            phi = tf.reduce_sum(
                alpha[:, :, None] * tf.exp(-tf.square(kappa[:, :, None] - u_range[:, None, :]) / beta[:, :, None]),
                axis=1,
            )  # [1,U]
            w = tf.matmul(phi, onehot_text)

            s2_in = tf.concat([x, h1, w], axis=-1)
            h2, c2 = self._lstm_step(s2_in, h2, c2, self.lstm2_k, self.lstm2_b)
            s3_in = tf.concat([x, h2, w], axis=-1)
            h3, c3 = self._lstm_step(s3_in, h3, c3, self.lstm3_k, self.lstm3_b)

            params = tf.matmul(h3, self.gmm_w) + self.gmm_b
            pis, sigmas, rhos, mus, es = tf.split(
                params, [NUM_OUT_MIX, 2 * NUM_OUT_MIX, NUM_OUT_MIX, 2 * NUM_OUT_MIX, 1], axis=-1
            )
            pis = tf.nn.log_softmax(pis, axis=-1)
            sigmas = tf.exp(tf.clip_by_value(sigmas, np.log(1e-4), np.log(1e3)))
            rhos = tf.clip_by_value(tf.tanh(rhos), -1 + 1e-6, 1 - 1e-6)
            es = tf.clip_by_value(tf.sigmoid(es), 1e-6, 1 - 1e-6)
            mu1, mu2 = mus[:, :NUM_OUT_MIX], mus[:, NUM_OUT_MIX:]
            s1_, s2_ = sigmas[:, :NUM_OUT_MIX], sigmas[:, NUM_OUT_MIX:]

            tx, ty, teos = targets[t, 0], targets[t, 1], targets[t, 2]
            z1 = (tx - mu1) / s1_
            z2 = (ty - mu2) / s2_
            zz = z1 * z1 + z2 * z2 - 2 * rhos * z1 * z2
            norm_const = 2 * np.pi * s1_ * s2_ * tf.sqrt(1 - rhos * rhos)
            log_gauss = -zz / (2 * (1 - rhos * rhos)) - tf.math.log(norm_const)
            log_mix = tf.reduce_logsumexp(pis + log_gauss, axis=-1)
            eos_term = teos * tf.math.log(es[:, 0]) + (1 - teos) * tf.math.log(1 - es[:, 0])
            total_nll += -(log_mix[0] + eos_term[0])

        return total_nll / tf.cast(T, tf.float32)

    def export_to_npz(self, path):
        d = {
            "rnn__LSTMAttentionCell__lstm_cell__kernel": self.lstm1_k.numpy(),
            "rnn__LSTMAttentionCell__lstm_cell__bias": self.lstm1_b.numpy(),
            "rnn__LSTMAttentionCell__lstm_cell_1__kernel": self.lstm2_k.numpy(),
            "rnn__LSTMAttentionCell__lstm_cell_1__bias": self.lstm2_b.numpy(),
            "rnn__LSTMAttentionCell__lstm_cell_2__kernel": self.lstm3_k.numpy(),
            "rnn__LSTMAttentionCell__lstm_cell_2__bias": self.lstm3_b.numpy(),
            "rnn__LSTMAttentionCell__attention__weights": self.attn_w.numpy(),
            "rnn__LSTMAttentionCell__attention__biases": self.attn_b.numpy(),
            "rnn__gmm__weights": self.gmm_w.numpy(),
            "rnn__gmm__biases": self.gmm_b.numpy(),
        }
        np.savez(path, **d)
        with open(os.path.splitext(path)[0] + ".alphabet.json", "w") as f:
            json.dump(self.alphabet, f)


def fit(model: TrainableHandwritingModel, samples, epochs=10, lr=1e-5, log_every=1, progress_cb=None, sample_progress_cb=None):
    """samples: list of (offsets[T,3], text). Simple per-sample SGD (Adam),
    teacher-forced NLL — this is deliberately the minimal correct training
    loop, not a throughput-optimized one; a personal fine-tune is small data.
    progress_cb(epoch, epochs, epoch_loss, elapsed_s, eta_s), if given, is
    called after every epoch — used by the web UI and CLI to report
    per-epoch timing and an ETA instead of going silent until it's done.
    sample_progress_cb(metrics), if given, receives throttled live metrics
    and completed epoch history, independently of the epoch callback."""
    if getattr(model, 'backend', None) == 'torch':
        from ai.training.torch_finetune import fit as torch_fit
        return torch_fit(model, samples, epochs, lr, log_every, progress_cb, sample_progress_cb)
    if tf is None:
        raise ImportError("TensorFlow is required for fine-tuning.")
    with tf.device(getattr(model, "device", "/CPU:0")):
        return _fit(model, samples, epochs, lr, log_every, progress_cb, sample_progress_cb)


def _fit(model, samples, epochs, lr, log_every, progress_cb, sample_progress_cb=None):
    if not samples:
        raise ValueError("No usable handwriting samples.")
    if epochs < 1 or log_every < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("Epochs, logging interval and learning rate must be positive.")
    missing = find_missing_chars(samples, model.alphabet)
    if missing:
        raise ValueError("Sample characters missing from the profile alphabet: " + " ".join(missing))
    prepared = []
    for offsets, text in samples:
        offsets = np.asarray(offsets, dtype=np.float32)
        if (offsets.ndim != 2 or offsets.shape[1] != 3 or len(offsets) < 2
                or not np.isfinite(offsets).all() or not np.isin(offsets[:, 2], [0, 1]).all()):
            raise ValueError("Each sample must contain finite [dx, dy, end-of-stroke] offsets.")
        prepared.append((tf.convert_to_tensor(offsets), tf.convert_to_tensor(model.encode_text(text))))
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    history = []
    training_log = TrainingLog(model, samples, epochs, lr, log_every, sample_progress_cb)
    t_start = training_log.started
    with training_log:
        for epoch in range(epochs):
            epoch_loss = 0.0
            for sample_index, (offsets, onehot) in enumerate(prepared, 1):
                with tf.GradientTape() as tape:
                    loss = model.forward_loss(offsets, onehot)
                grads = tape.gradient(loss, model.trainable_variables)
                tf.debugging.assert_all_finite(loss, "Training loss is not finite; check handwriting samples and learning rate.")
                if any(g is None for g in grads):
                    raise ValueError("Training graph disconnected from profile weights.")
                for grad in grads:
                    tf.debugging.assert_all_finite(grad, "Training gradient is not finite; profile was not updated.")
                grads, _ = tf.clip_by_global_norm(grads, 5.0)
                opt.apply_gradients(zip(grads, model.trainable_variables))
                epoch_loss += float(loss)
                training_log.sample(epoch, sample_index, epoch_loss)
            epoch_loss /= max(len(samples), 1)
            history.append(epoch_loss)
            elapsed = time.monotonic() - t_start
            eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1)
            if progress_cb:
                progress_cb(epoch, epochs, epoch_loss, elapsed, eta)
            training_log.epoch(epoch, epoch_loss, elapsed, eta)
        training_log.finish(history, time.monotonic() - t_start)
    return history
