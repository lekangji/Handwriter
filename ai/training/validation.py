"""Check free-running generation before publishing a new trained profile."""
import logging
import os
import shutil
import tempfile
import json
import numpy as np

from ai.inference.hw_model import GenerationError, HandwritingModel
from ai.training.samples import load_profile_alphabet


logger = logging.getLogger('ai.training.validation')
# Deterministic smoke checks cover the reported failure and longer phrases.
# They detect broken generation, not legibility or style fidelity.
PROBES = (('Dear', 2029167940), ('Hello world', 7),
          ('the quick brown fox', 19), ('Dear team', 29))


def validate_profile(path, prime_strokes=None, prime_text=None):
    model = HandwritingModel(path, load_profile_alphabet(path))
    passed, failed = 0, []
    for text, seed in PROBES:
        try:
            offsets = model.sample(text, bias=1.5, seed=seed,
                                   prime_strokes=prime_strokes, prime_text=prime_text)
        except GenerationError:
            failed.append(text)
            logger.warning('Profile generation check failed text=%r seed=%d', text, seed)
        else:
            passed += 1
            logger.debug('Profile generation check passed text=%r seed=%d points=%d',
                         text, seed, len(offsets))
    logger.info('Profile generation checks passed=%d total=%d', passed, len(PROBES))
    return {'passed': passed, 'failed': failed}


def save_validated_profile(model, path, samples=None, metadata=None):
    """Export to temporary files; publish only after all generation checks pass."""
    path = os.path.abspath(path)
    alphabet_path = os.path.splitext(path)[0] + '.alphabet.json'
    style_path = os.path.splitext(path)[0] + '.style.json'
    metadata_path = os.path.splitext(path)[0] + '.training.json'
    if any(os.path.exists(p) for p in (path, alphabet_path, style_path, metadata_path)):
        raise ValueError('That profile already exists. Choose a new name to preserve it.')
    directory = os.path.dirname(path)
    with tempfile.TemporaryDirectory(prefix='.training-', dir=directory) as temporary:
        candidate = os.path.join(temporary, 'candidate.npz')
        model.export_to_npz(candidate)
        report = validate_profile(candidate)
        if report['failed']:
            phrases = ', '.join(repr(text) for text in report['failed'])
            raise ValueError(
                f'Trained profile failed the generation check on {phrases}. '
                'Lower training loss did not preserve working generation. No profile was saved. '
                'Try fewer epochs or a lower learning rate from a working base profile.'
            )
        files = [(os.path.splitext(candidate)[0] + '.alphabet.json', alphabet_path)]
        if samples is not None:
            from ai.training.style import select_style
            style = select_style(samples, load_profile_alphabet(candidate))
            retained = []
            for sample in style['samples']:
                primed = validate_profile(candidate, prime_text=sample['text'],
                                          prime_strokes=np.asarray(sample['offsets'], dtype=np.float32))
                if not primed['failed']:
                    retained.append(sample)
            style['samples'] = retained
            report['style_samples_retained'] = len(retained)
            report['style_warning'] = (None if retained else
                'No captured sample passed style-priming checks. The working profile was saved without priming; capture more clear phrases to add your style.')
            if report['style_warning']:
                logger.warning(report['style_warning'])
            source = os.path.splitext(candidate)[0] + '.style.json'
            with open(source, 'w', encoding='utf-8') as stream:
                json.dump(style, stream, allow_nan=False)
            files.append((source, style_path))
        if metadata is not None:
            metadata = dict(metadata, style_samples_retained=report.get('style_samples_retained', 0),
                            style_warning=report.get('style_warning'))
            source = os.path.splitext(candidate)[0] + '.training.json'
            with open(source, 'w', encoding='utf-8') as stream:
                json.dump(metadata, stream, allow_nan=False)
            files.append((source, metadata_path))
        # Publish weights last: a profile only becomes discoverable with all sidecars present.
        files.append((candidate, path))
        created = []
        try:
            # Exclusive creation preserves existing files even if another
            # process publishes the same profile while validation is running.
            for source, target in files:
                with open(source, 'rb') as incoming, open(target, 'xb') as outgoing:
                    created.append(target)
                    shutil.copyfileobj(incoming, outgoing)
        except Exception:
            for target in created:
                os.remove(target)
            raise
    return report
