"""Small, compatible model-space captures retained for style priming."""
import json
import os
import numpy as np


def select_style(samples, alphabet, limit=3):
    valid, seen = [], set()
    for offsets, text in sorted(samples, key=lambda s: abs(len(s[1]) - 20)):
        offsets = np.asarray(offsets, dtype=np.float32)
        if (text in seen or not 1 <= len(text) <= 48 or any(c not in alphabet or c == '\0' for c in text)
                or offsets.ndim != 2 or offsets.shape[1] != 3 or not 2 <= len(offsets) <= 4000
                or not np.isfinite(offsets).all() or not np.isin(offsets[:, 2], [0, 1]).all()):
            continue
        valid.append({'text': text, 'offsets': offsets.tolist()})
        seen.add(text)
        if len(valid) >= limit:
            break
    return {'version': 1, 'samples': valid}


def load_style(weights_path, alphabet):
    path = os.path.splitext(weights_path)[0] + '.style.json'
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding='utf-8') as stream:
            data = json.load(stream)
        if data.get('version') != 1 or not isinstance(data.get('samples'), list) or len(data['samples']) > 3:
            raise ValueError('Unsupported style format.')
        result = []
        for sample in data['samples']:
            text = sample['text']
            offsets = np.asarray(sample['offsets'], dtype=np.float32)
            if (not isinstance(text, str) or not text.strip() or len(text) > 48
                    or any(c not in alphabet or c == '\0' for c in text)
                    or offsets.ndim != 2 or offsets.shape[1] != 3 or not 2 <= len(offsets) <= 4000
                    or not np.isfinite(offsets).all() or not np.isin(offsets[:, 2], [0, 1]).all()):
                raise ValueError('Incompatible style sample.')
            result.append((offsets, text))
        return result
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        raise ValueError('Saved style samples are invalid or incompatible with this profile.') from error
