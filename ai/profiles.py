"""Profile ordering and inspectable character/training metadata."""
import glob
import json
import os
import tempfile

from ai.inference.hw_model import missing_uppercase
from ai.training.samples import load_profile_alphabet
from ai.training.style import load_style

SIDECARS = ('.alphabet.json', '.style.json', '.training.json')


def ordered_profiles(directory):
    names = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(directory, '*.npz')))
    try:
        with open(os.path.join(directory, '.order.json'), encoding='utf-8') as stream:
            saved = json.load(stream)
        if not isinstance(saved, list):
            saved = []
    except (OSError, ValueError):
        saved = []
    ordered = list(dict.fromkeys(n for n in saved if isinstance(n, str) and n in names and n != 'default'))
    ordered.extend(n for n in names if n != 'default' and n not in ordered)
    return (['default'] if 'default' in names else []) + ordered


def save_order(directory, names):
    existing = ordered_profiles(directory)
    order = list(dict.fromkeys(n for n in names if n in existing and n != 'default'))
    order.extend(n for n in existing if n != 'default' and n not in order)
    fd, temporary = tempfile.mkstemp(prefix='.order-', suffix='.json', dir=directory)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(order, stream)
        os.replace(temporary, os.path.join(directory, '.order.json'))
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def move_profile(directory, name, direction):
    names = ordered_profiles(directory)
    if name not in names or name == 'default' or direction not in (-1, 1):
        raise ValueError('Choose a personal profile and a valid reorder direction.')
    index = names.index(name)
    target = index + direction
    if 0 <= target < len(names) and names[target] != 'default':
        names[index], names[target] = names[target], names[index]
    save_order(directory, names)


def describe_profile(directory, name):
    if name not in ordered_profiles(directory):
        raise ValueError('Choose an existing handwriting profile.')
    path = os.path.join(directory, name + '.npz')
    result = {'alphabet': [], 'alphabet_size': 0, 'missing_uppercase': [],
              'style_samples': 0, 'training': {}, 'error': None,
              'size_mb': round(os.path.getsize(path) / (1024 * 1024), 2)}
    try:
        alphabet = load_profile_alphabet(path)
        result.update(alphabet=[c for c in alphabet if c != '\0'], alphabet_size=len(alphabet) - 1,
                      missing_uppercase=missing_uppercase(alphabet))
        result['style_samples'] = len(load_style(path, alphabet))
        metadata_path = os.path.splitext(path)[0] + '.training.json'
        if os.path.exists(metadata_path):
            with open(metadata_path, encoding='utf-8') as stream:
                metadata = json.load(stream)
            if not isinstance(metadata, dict):
                raise ValueError('Training metadata must be an object.')
            result['training'] = metadata
    except (OSError, ValueError, TypeError) as error:
        result['error'] = str(error)
    return result
