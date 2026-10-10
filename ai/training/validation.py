"""Check free-running generation before publishing a new trained profile."""
import logging
import os
import shutil
import tempfile

from ai.inference.hw_model import GenerationError, HandwritingModel
from ai.training.samples import load_profile_alphabet


logger = logging.getLogger('ai.training.validation')
# Deterministic smoke checks cover the reported failure and longer phrases.
# They detect broken generation, not legibility or style fidelity.
PROBES = (('Dear', 2029167940), ('Hello world', 7),
          ('the quick brown fox', 19), ('Dear team', 29))


def validate_profile(path):
    model = HandwritingModel(path, load_profile_alphabet(path))
    passed, failed = 0, []
    for text, seed in PROBES:
        try:
            offsets = model.sample(text, bias=1.5, seed=seed)
        except GenerationError:
            failed.append(text)
            logger.warning('Profile generation check failed text=%r seed=%d', text, seed)
        else:
            passed += 1
            logger.debug('Profile generation check passed text=%r seed=%d points=%d',
                         text, seed, len(offsets))
    logger.info('Profile generation checks passed=%d total=%d', passed, len(PROBES))
    return {'passed': passed, 'failed': failed}


def save_validated_profile(model, path):
    """Export to temporary files; publish only after all generation checks pass."""
    path = os.path.abspath(path)
    alphabet_path = os.path.splitext(path)[0] + '.alphabet.json'
    if os.path.exists(path) or os.path.exists(alphabet_path):
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
        created = []
        try:
            # Exclusive creation preserves existing files even if another
            # process publishes the same profile while validation is running.
            for source, target in ((os.path.splitext(candidate)[0] + '.alphabet.json', alphabet_path),
                                   (candidate, path)):
                with open(source, 'rb') as incoming, open(target, 'xb') as outgoing:
                    created.append(target)
                    shutil.copyfileobj(incoming, outgoing)
        except Exception:
            for target in created:
                os.remove(target)
            raise
    return report
