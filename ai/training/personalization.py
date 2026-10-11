"""Select working personalization checkpoints on complete held-out phrases."""
from contextlib import nullcontext
import logging
import os
import tempfile
import numpy as np

from ai.training.validation import validate_profile

logger = logging.getLogger('ai.training.selection')


def split_samples(samples, seed=7):
    if not samples:
        raise ValueError('No usable handwriting samples.')
    groups = {}
    for sample in samples:
        groups.setdefault(' '.join(sample[1].split()), []).append(sample)
    if len(groups) < 5:
        return list(samples), list(samples), 'training_data'
    counts = {}
    for text in groups:
        for char in set(text):
            counts[char] = counts.get(char, 0) + 1
    keys = list(groups)
    held_out = set()
    for index in np.random.default_rng(seed).permutation(len(keys)):
        key = keys[index]
        if all(counts[c] > 1 for c in set(key)):
            held_out.add(key)
            for char in set(key):
                counts[char] -= 1
        if len(held_out) >= max(1, len(keys) // 5):
            break
    if not held_out:
        return list(samples), list(samples), 'training_data'
    training = [s for s in samples if ' '.join(s[1].split()) not in held_out]
    validation = [s for s in samples if ' '.join(s[1].split()) in held_out]
    return training, validation, 'held_out'


def _snapshot(model):
    return [(v.detach().cpu().numpy().copy() if hasattr(v, 'detach') else v.numpy().copy())
            for v in model.trainable_variables]


def _restore(model, weights):
    if getattr(model, 'backend', None) == 'torch':
        import torch
        with torch.no_grad():
            for variable, array in zip(model.trainable_variables, weights):
                variable.copy_(torch.as_tensor(array, device=variable.device))
    else:
        for variable, array in zip(model.trainable_variables, weights):
            variable.assign(array)


def _evaluate(model, samples):
    context = nullcontext()
    if getattr(model, 'backend', None) == 'torch':
        import torch
        context = torch.inference_mode()
    with context:
        scores = [float(model.forward_loss(offsets, model.encode_text(text))) for offsets, text in samples]
    score = float(np.mean(scores))
    return score if np.isfinite(score) else float('inf')


def fit_personalized(model, samples, epochs=10, lr=1e-5, log_every=1,
                     progress_cb=None, sample_progress_cb=None, patience=3, fit_fn=None):
    """Restore the lowest-evaluation-NLL checkpoint that can still generate.

    Epoch zero is the source model, with any new alphabet rows initialized.
    Generation smoke checks are not a readability or style-quality score.
    """
    if epochs < 1 or patience < 1:
        raise ValueError('Epochs and patience must be positive.')
    if fit_fn is None:
        from ai.training.finetune import fit
        fit_fn = fit
    training, validation, kind = split_samples(samples)
    original = _snapshot(model)
    best_weights, best_loss, best_epoch = None, float('inf'), None
    stale, records = 0, []
    with tempfile.TemporaryDirectory(prefix='handwriter-selection-') as directory:
        candidate = os.path.join(directory, 'candidate.npz')

        def consider(epoch, train_loss=None):
            nonlocal best_weights, best_loss, best_epoch, stale
            score = _evaluate(model, validation)
            eligible = False
            if score < best_loss:
                model.export_to_npz(candidate)
                eligible = not validate_profile(candidate)['failed']
                if eligible:
                    best_weights, best_loss, best_epoch = _snapshot(model), score, epoch
                    stale = 0
            if epoch and not eligible:
                stale += 1
            records.append({'epoch': epoch, 'loss': score if np.isfinite(score) else None,
                            'selected': eligible})
            logger.info('Checkpoint epoch=%d evaluation=%s nll=%s selected=%s best_epoch=%s',
                        epoch, kind, score, eligible, best_epoch)
            if sample_progress_cb:
                sample_progress_cb({'validation_kind': kind, 'validation_loss': records[-1]['loss'],
                                    'selected_epoch': best_epoch,
                                    'best_validation_loss': best_loss if best_weights is not None else None})
            return stale < patience

        try:
            consider(0)
            history = fit_fn(model, training, epochs=epochs, lr=lr, log_every=log_every,
                             progress_cb=progress_cb, sample_progress_cb=sample_progress_cb,
                             epoch_cb=consider)
            if best_weights is None:
                raise ValueError('No working checkpoint passed generation checks. No profile was saved.')
            _restore(model, best_weights)
        except Exception:
            _restore(model, original)
            raise
    report = {'selected_epoch': best_epoch, 'best_validation_loss': best_loss,
              'validation_kind': kind, 'training_samples': len(training),
              'validation_samples': len(validation), 'validation_history': records,
              'completed_epochs': len(history), 'requested_epochs': epochs,
              'early_stopped': len(history) < epochs}
    return history, report
