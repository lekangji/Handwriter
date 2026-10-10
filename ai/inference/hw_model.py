"""
Handwriting synthesis model — pure NumPy inference.

Reimplements the forward pass of Alex Graves' "Generating Sequences With
Recurrent Neural Networks" (2013) handwriting synthesis network, using
pretrained weights (trained on IAM online handwriting) extracted from
https://github.com/sjvasquez/handwriting-synthesis.

Architecture (confirmed against that repo's rnn_cell.py):
  - 3 stacked LSTM layers (400 units each), skip connections from the raw
    input and the attention window into every layer.
  - A soft "attention window" (mixture of K=10 Gaussians) over the one-hot
    encoded target character sequence, which lets the network decide which
    character it's currently writing at each timestep.
  - A mixture-density-network output layer (M=20 bivariate Gaussians +
    a Bernoulli pen-up/pen-down probability) sampled stochastically at
    each step to produce the next (dx, dy, eos) pen offset.

No TensorFlow/PyTorch is used at inference time — weights were extracted
once (see extract step) into hw_weights.npz.
"""

import os
import numpy as np

ALPHABET = [
    "\x00",
    " ",
    "!",
    '"',
    "#",
    "'",
    "(",
    ")",
    ",",
    "-",
    ".",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ":",
    ";",
    "?",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "Y",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
]
CHAR_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}
VALID_CHARS = set(ALPHABET) - {"\x00"}
LSTM_SIZE = 400
NUM_ATTN_MIX = 10
NUM_OUT_MIX = 20
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "hw_weights.npz")


def sanitize_line(line: str, valid_chars=None) -> str:
    """Replace/drop characters the model wasn't trained on. Never silently
    drops a letter if a same-letter fallback exists (e.g. this alphabet has
    no uppercase Q/X/Z — those fall back to lowercase rather than vanishing).
    Pass valid_chars=set(model.alphabet) when using a fine-tuned/grown model."""
    valid = valid_chars if valid_chars is not None else VALID_CHARS
    out = []
    for ch in line:
        if ch in valid:
            out.append(ch)
        elif ch.isalpha() and ch.lower() in valid:
            out.append(ch.lower())  # e.g. 'Q' -> 'q': alphabet has no uppercase Q/X/Z
        elif ch == "\t":
            out.append(" ")
        elif ch in ("\u2018", "\u2019") and "'" in valid:
            out.append("'")
        elif ch in ("\u201c", "\u201d") and '"' in valid:
            out.append('"')
        elif ch in ("\u2014", "\u2013") and "-" in valid:
            out.append("-")
        # else: drop silently (emoji, unsupported unicode, etc.) — extend
        # the alphabet via finetune.py's build_alphabet() for real symbols.
    return "".join(out)


def missing_uppercase(alphabet=None) -> list:
    """Which A-Z letters this alphabet has no uppercase glyph for (they'll
    fall back to lowercase via sanitize_line). Computed from whatever
    alphabet is passed in — including a fine-tuned/grown one — rather than
    a fixed hardcoded list, so it stays correct after fine-tuning adds
    characters."""
    valid = set(alphabet) if alphabet is not None else VALID_CHARS
    return [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in valid]


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


class _LSTMCell:
    """Standard (non-peephole) LSTM cell matching tf.contrib.rnn.LSTMCell,
    gate order [i, j, f, o], forget_bias=1.0, loaded from a fused kernel."""

    def __init__(self, kernel, bias, hidden_size):
        self.kernel = kernel  # [in_dim+hidden, 4*hidden]
        self.bias = bias  # [4*hidden]
        self.hidden_size = hidden_size

    def __call__(self, x, h_prev, c_prev):
        z = np.concatenate([x, h_prev], axis=-1) @ self.kernel + self.bias
        i, j, f, o = np.split(z, 4, axis=-1)
        c = _sigmoid(f + 1.0) * c_prev + _sigmoid(i) * np.tanh(j)
        h = _sigmoid(o) * np.tanh(c)
        return h, c


class GenerationError(ValueError):
    """The model exhausted its bounded budget without a reliable complete draw."""


class HandwritingModel:
    """Loads pretrained weights once; `sample()` is cheap to call repeatedly."""

    def __init__(self, weights_path: str = WEIGHTS_PATH, alphabet=None):
        """alphabet: pass a custom (possibly grown) alphabet list to load a
        fine-tuned checkpoint produced by finetune.py's export_to_npz() —
        that also writes a `<name>.alphabet.json` next to the weights;
        load it and pass it here. Defaults to the base 73-char alphabet."""
        self.alphabet = list(alphabet) if alphabet else list(ALPHABET)
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}
        w = np.load(weights_path)
        g = lambda name: w[name.replace("/", "__")]
        self.lstm1 = _LSTMCell(
            g("rnn/LSTMAttentionCell/lstm_cell/kernel"), g("rnn/LSTMAttentionCell/lstm_cell/bias"), LSTM_SIZE
        )
        self.lstm2 = _LSTMCell(
            g("rnn/LSTMAttentionCell/lstm_cell_1/kernel"), g("rnn/LSTMAttentionCell/lstm_cell_1/bias"), LSTM_SIZE
        )
        self.lstm3 = _LSTMCell(
            g("rnn/LSTMAttentionCell/lstm_cell_2/kernel"), g("rnn/LSTMAttentionCell/lstm_cell_2/bias"), LSTM_SIZE
        )
        self.attn_w = g("rnn/LSTMAttentionCell/attention/weights")
        self.attn_b = g("rnn/LSTMAttentionCell/attention/biases")  # [30]
        self.gmm_w = g("rnn/gmm/weights")
        self.gmm_b = g("rnn/gmm/biases")  # [121]

    def _encode_text(self, text: str):
        idx = [self.char_to_idx[c] for c in text] + [0]  # trailing null, per encode_ascii
        U = len(idx)
        onehot = np.zeros((U, len(self.alphabet)), dtype=np.float32)
        onehot[np.arange(U), idx] = 1.0
        return onehot  # [U, A]

    def sample(self, text: str, bias: float = 1.0, seed=None, prime_strokes=None,
               prime_text=None, max_steps=None, attempts=3, steps_per_char=50):
        """Generate a bounded, complete draw. Reject failed draws instead of exporting scribbles.

        bias controls sampling variance; attempts limits retries. Short independent
        chunks in document.layout keep attention within the checkpoint's training range.
        prime_strokes are model-space (dx, dy, eos), with an exact transcription.
        """
        if not np.isfinite(bias) or not 0 <= bias <= 3:
            raise ValueError("Neatness must be between 0 and 3.")
        if not 1 <= attempts <= 5 or not 20 <= steps_per_char <= 100:
            raise ValueError("Retries must be 1–5 and steps per character 20–100.")
        text = sanitize_line(text, valid_chars=set(self.alphabet)).strip()
        if not text:
            return np.zeros((1, 3), dtype=np.float32)
        if (prime_strokes is None) != (prime_text is None):
            raise ValueError("Style priming needs both strokes and their transcription.")
        if prime_strokes is not None:
            prime_strokes = np.asarray(prime_strokes, dtype=np.float32)
            if prime_strokes.ndim != 2 or prime_strokes.shape[1] != 3 or not len(prime_strokes) or not np.isfinite(prime_strokes).all():
                raise ValueError("Priming strokes must be a finite, nonempty N×3 array.")
        full_text = (prime_text + " " + text) if prime_text else text
        onehot = self._encode_text(full_text)
        rng = np.random.default_rng(seed)
        budget = max_steps if max_steps is not None else max(200, steps_per_char * len(text))
        if budget < 1:
            raise ValueError("Step budget must be positive.")
        for _ in range(attempts):
            out, completed = self._sample_once(onehot, bias, rng, prime_strokes, budget, steps_per_char)
            width_per_char = float(out[:, 0].sum()) / len(text) if len(out) else 0
            if completed and np.isfinite(out).all() and 2 <= width_per_char <= 20:
                out[-1, 2] = 1
                return _split_missed_pen_lifts(out)
        raise GenerationError(f"Could not finish {text!r} reliably. Try a different seed, shorter chunks, or the default profile.")

    def _sample_once(self, onehot, bias, rng, prime_strokes, max_steps, steps_per_char):
        h1 = c1 = h2 = c2 = h3 = c3 = np.zeros(LSTM_SIZE, dtype=np.float32)
        kappa = np.zeros(NUM_ATTN_MIX, dtype=np.float32)
        w = np.zeros(len(self.alphabet), dtype=np.float32)
        x = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        u = np.arange(len(onehot), dtype=np.float32)

        def step(x_in):
            nonlocal h1, c1, h2, c2, h3, c3, kappa, w
            h1, c1 = self.lstm1(np.concatenate([w, x_in]), h1, c1)
            params = np.concatenate([w, x_in, h1]) @ self.attn_w + self.attn_b
            alpha, beta, delta = np.split(_softplus(params), 3)
            kappa = kappa + delta / 25.0
            beta = np.maximum(beta, 0.01)
            phi = np.sum(alpha[:, None] * np.exp(-np.square(kappa[:, None] - u) / beta[:, None]), axis=0)
            w = phi @ onehot
            h2, c2 = self.lstm2(np.concatenate([x_in, h1, w]), h2, c2)
            h3, c3 = self.lstm3(np.concatenate([x_in, h2, w]), h3, c3)
            return phi

        # Teacher-force the real sample before free-running on the target text.
        if prime_strokes is not None:
            for point in prime_strokes:
                step(x)
                x = point

        out = np.empty((max_steps, 3), dtype=np.float32)
        terminal_steps = 0
        attention_steps = np.zeros(len(onehot), dtype=np.int32)
        for t in range(max_steps):
            phi = step(x)
            params = h3 @ self.gmm_w + self.gmm_b
            if not np.isfinite(params).all() or not np.isfinite(phi).all():
                return out[:t], False
            pis, sigmas, rhos, mus, es = np.split(params, [20, 60, 80, 120])
            pis = _softmax(pis * (1 + bias))
            pis[pis < 0.01] = 0
            pis /= pis.sum()
            sigmas = np.exp(np.clip(sigmas - bias, -9, 7))
            rhos = np.clip(np.tanh(rhos), -0.999999, 0.999999)
            k = rng.choice(NUM_OUT_MIX, p=pis)
            z1, z2 = rng.standard_normal(2)
            # Analytic bivariate draw avoids a covariance decomposition at every point.
            dx = mus[k] + sigmas[k] * z1
            dy = mus[20 + k] + sigmas[20 + k] * (rhos[k] * z1 + np.sqrt(1 - rhos[k] ** 2) * z2)
            eos = float(rng.random() < float(_sigmoid(es[0])))
            out[t] = (dx, dy, eos)
            x = out[t]
            # Final-character attention plus a pen lift matches the checkpoint.
            position = int(np.argmax(phi))
            attention_steps[position] += 1
            # Repeated strokes often mean attention is stuck on one character.
            # Multiple attention components can switch peaks; reject prolonged dwell,
            # rather than treating every legitimate peak switch as skipped text.
            if attention_steps[position] > max(80, 2 * steps_per_char):
                return out[:t + 1].copy(), False
            at_end = position == len(onehot) - 1
            terminal_steps = terminal_steps + 1 if at_end else 0
            if at_end and (eos or terminal_steps >= 8):
                return out[:t + 1].copy(), True
        return out, False


def align_baseline(coords: np.ndarray) -> np.ndarray:
    """Removes global slope/offset drift from a generated line (least-squares
    fit of y vs x, subtracted out) so the line sits flat and centered on the
    ruling rather than drifting up/down or slanting across a long line —
    the RNN's baseline can wander gradually since nothing in the model
    explicitly pins it to a fixed y. Mirrors drawing.align() from the
    reference implementation."""
    xy = coords[:, :2]
    if len(xy) < 3:
        return coords
    X = np.column_stack([np.ones(len(xy)), xy[:, 0]])
    try:
        offset, slope = np.linalg.lstsq(X, xy[:, 1], rcond=None)[0]
    except np.linalg.LinAlgError:
        return coords
    out = np.copy(coords)
    out[:, 1] = xy[:, 1] - (offset + slope * xy[:, 0])
    return out


def offsets_to_coords(offsets: np.ndarray) -> np.ndarray:
    """[N,3] (dx,dy,eos) -> [N,3] absolute (x,y,eos)."""
    coords = np.copy(offsets)
    coords[:, :2] = np.cumsum(offsets[:, :2], axis=0)
    return coords


def _split_missed_pen_lifts(offsets: np.ndarray, jump_factor: float = 6.0) -> np.ndarray:
    """Post-process fix for 'pen doesn't lift between words': the eos
    (end-of-stroke) output is a stochastic Bernoulli sample and sometimes
    just doesn't fire where it obviously should — the model still moves the
    pen a normal word-gap distance, it just fails to flag it as a lift, so
    downstream code draws a stray line connecting two words. This detects
    steps whose offset magnitude is a clear outlier vs. the rest of the
    stroke (a real pen-gap jump, not just fast cursive) and forces eos=1 on
    them. jump_factor is intentionally conservative (large) so normal fast
    strokes are never split — only obvious mis-flagged gaps are."""
    if len(offsets) < 4:
        return offsets
    out = np.copy(offsets)
    step_mag = np.linalg.norm(out[:, :2], axis=1)
    typical = np.median(step_mag[step_mag > 0]) if np.any(step_mag > 0) else 0.0
    if typical <= 0:
        return out
    threshold = typical * jump_factor
    outliers = step_mag > threshold
    # A jump arrives at the next stroke; lift at the point BEFORE it.
    indices = np.flatnonzero(outliers)
    out[indices[indices > 0] - 1, 2] = 1.0
    return out
