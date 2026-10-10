"""Capture data and alphabet validation, without loading TensorFlow."""
import json
import os
import warnings
import numpy as np
from ai.inference.hw_model import ALPHABET as BASE_ALPHABET


def build_alphabet(extra_chars="", alphabet=None, samples=()):
    """Base 73-char alphabet + any new characters you want the model to learn
    (order matters only in that it must stay fixed for a given weight file —
    export_to_npz() always saves alphabet.json alongside the weights)."""
    result = list(dict.fromkeys(list(BASE_ALPHABET if alphabet is None else alphabet) + list(extra_chars)))
    return result + find_missing_chars(samples, result)


def load_profile_alphabet(weights_path):
    path = os.path.splitext(weights_path)[0] + ".alphabet.json"
    if not os.path.exists(path):
        return list(BASE_ALPHABET)
    with open(path, encoding="utf-8") as f:
        alphabet = json.load(f)
    if (not isinstance(alphabet, list) or not alphabet or alphabet[0] != "\x00"
            or any(not isinstance(c, str) or len(c) != 1 for c in alphabet)
            or len(set(alphabet)) != len(alphabet)):
        raise ValueError("Profile alphabet must contain unique characters, with the null token first.")
    return alphabet


def find_missing_chars(samples, alphabet):
    """Which characters in `samples` (list of (offsets, text)) aren't in
    `alphabet`. Call this BEFORE building a TrainableHandwritingModel or
    running fit() — checking every sample's text against the alphabet is a
    cheap, instant pass, whereas the failure used to only surface as a
    KeyError partway through the first training step, after the (much
    slower) TF import, weight-matrix surgery, and optimizer setup had
    already happened. Returns a sorted list of missing characters, or []."""
    valid = set(alphabet)
    missing = set()
    for _, text in samples:
        for ch in text:
            if ch not in valid:
                missing.add(ch)
    return sorted(missing)



def _resample_stroke(xy, spacing):
    if len(xy) < 2:
        return xy
    out = [xy[0]]
    remaining = spacing
    for a, b in zip(xy[:-1], xy[1:]):
        distance = np.linalg.norm(b - a)
        while distance >= remaining:
            a = a + (b - a) * (remaining / distance)
            out.append(a)
            distance -= remaining
            remaining = spacing
        remaining -= distance
    if not np.array_equal(out[-1], xy[-1]):
        out.append(xy[-1])
    return np.asarray(out)


def load_training_samples(json_path):
    """Loads captured samples from webapp/capture_tool/index.html's export format:
    {"calibration_px_per_model_unit": <float>, "samples": [{"text": "...",
    "strokes": [[[x,y],[x,y],...], [[x,y],...]]}, ...]} — nested point
    arrays per stroke, so a stroke boundary itself is the pen-lift signal
    rather than a repeated per-point flag.

    calibration_px_per_model_unit is ONE fixed number for the whole capture
    session (measured once, from a deliberate calibration stroke) and is
    applied identically to every sample here. That matters: the previous
    version of this function normalized each sample independently (its own
    median step size -> 1 unit), which silently erases real handwriting-size
    differences between samples — a big confident hand and a small cramped
    one both get rescaled to look the same size to the model, which is
    exactly the kind of contradictory training signal that can make a
    fine-tune degenerate. A single shared scale preserves that signal.

    Also accepts the older flat-list `{"text","points":[{"x","y","pen_down"}]}`
    export format for backward compatibility with previously captured files
    — those have no calibration factor, so a shared scale is estimated from
    their median phrase height and resampled at half a model unit.
    Calibrated captures retain their recorded sampling, including older
    exports without a resample_model_units marker. Densifying those paths
    changes the recurrent sequence and can break pretrained text attention
    even though their geometry is unchanged.

    Note on pen-up travel: like the previous version, points are only ever
    recorded while the pointer is physically down (no hover tracking), so
    the offset from one stroke's last point to the next stroke's first
    point encodes a real but unsampled "jump" rather than a resampled
    travel path — this is an inherent limitation of 2D pointer capture, not
    something introduced here.
    """
    if hasattr(json_path, "read"):
        data = json.load(json_path)
    else:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    if isinstance(data, list):
        converted = []
        for s in data:
            if not isinstance(s, dict) or not isinstance(s.get("points"), list):
                raise ValueError("Each legacy sample must contain a points list.")
            strokes, current = [], []
            for p in s["points"]:
                current.append([p["x"], p["y"]])
                if p.get("pen_down", True) is False:
                    strokes.append(current)
                    current = []
            if current:
                strokes.append(current)
            converted.append({"text": s.get("text"), "strokes": strokes})
        data = {"samples": converted}

    if not isinstance(data, dict) or not isinstance(data.get("samples"), list):
        raise ValueError("Expected a capture export containing a samples list.")
    prepared = []
    for s in data.get("samples", []):
        if not isinstance(s, dict) or not isinstance(s.get("strokes"), list):
            raise ValueError("Each sample must contain a strokes list.")
        strokes = s.get("strokes", [])
        stroke_arrays = []
        for stroke in strokes:
            if not isinstance(stroke, list):
                raise ValueError("Each stroke must contain a list of points.")
            if not stroke:
                continue
            xy = np.array(stroke, dtype=np.float64)
            if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
                raise ValueError("Stroke coordinates must be finite [x, y] pairs.")
            stroke_arrays.append(xy)
        text = s.get("text")
        if not isinstance(text, str) or not text.strip() or "\x00" in text or "\n" in text or "\r" in text:
            raise ValueError("Each sample must have a nonempty single-line transcription.")
        if sum(len(xy) for xy in stroke_arrays) >= 2:
            prepared.append((stroke_arrays, text))

    if "calibration_px_per_model_unit" in data:
        scale = data["calibration_px_per_model_unit"]
        if not isinstance(scale, (int, float)) or isinstance(scale, bool) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Calibration must be a finite positive number.")
    else:
        heights = [np.ptp(np.concatenate(strokes)[:, 1]) for strokes, _ in prepared]
        # shortcut: old files lack calibration; phrase height estimates scale until a calibrated export is available.
        scale = float(np.median([h for h in heights if h > 0])) / 25 if any(h > 0 for h in heights) else 1.0
        warnings.warn("Uncalibrated samples use a shared scale estimated from phrase height.")

    out = []
    for strokes, text in prepared:
        if "calibration_px_per_model_unit" not in data and data.get("resample_model_units") != 0.5:
            strokes = [_resample_stroke(xy, scale * 0.5) for xy in strokes]
        if sum(len(xy) for xy in strokes) < 2:
            continue
        xy = np.concatenate(strokes)
        eos = np.zeros(len(xy), dtype=np.float32)
        eos[np.cumsum([len(stroke) for stroke in strokes]) - 1] = 1
        dx = np.diff(xy[:, 0], prepend=xy[0, 0])
        dy = -np.diff(xy[:, 1], prepend=xy[0, 1])  # browser y-down -> model y-up
        offsets = np.stack([dx, dy, eos], axis=1).astype(np.float32)
        offsets[:, :2] /= scale  # same fixed scale for every sample this session
        out.append((offsets, text))
    return out


