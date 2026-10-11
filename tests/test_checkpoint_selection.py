import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from ai.training import personalization


class Variable:
    def __init__(self):
        self.value = np.array([0.0], np.float32)
    def numpy(self):
        return self.value.copy()
    def assign(self, value):
        self.value[:] = value


class Model:
    alphabet = list('\0 abcdefghijklmnopqrstuvwxyz')
    def __init__(self, scores):
        self.trainable_variables = [Variable()]
        self.scores = scores
    def encode_text(self, text):
        return text
    def forward_loss(self, offsets, text):
        return self.scores[int(self.trainable_variables[0].value[0])]
    def export_to_npz(self, path):
        np.savez(path, epoch=self.trainable_variables[0].value)


def trainer(model, samples, epochs, epoch_cb, **kwargs):
    history = []
    for epoch in range(epochs):
        model.trainable_variables[0].assign([epoch + 1])
        history.append(10 - epoch)
        if epoch_cb(epoch + 1, history[-1]) is False:
            break
    return history


def samples(texts):
    return [(np.array([[0, 0, 0], [1, 1, 1]], np.float32), text) for text in texts]


class CheckpointSelectionTests(unittest.TestCase):
    def test_duplicate_transcriptions_do_not_leak_into_holdout(self):
        data = samples(['ab', 'ba', 'aba', 'bab', 'aabb', 'ab', 'ab'])
        training, validation, kind = personalization.split_samples(data)
        self.assertEqual(kind, 'held_out')
        self.assertTrue(validation)
        self.assertFalse({text for _, text in training} & {text for _, text in validation})
        self.assertEqual(len(training) + len(validation), 7)

    def test_small_dataset_uses_labeled_training_evaluation(self):
        training, validation, kind = personalization.split_samples(samples(['ab', 'ba']))
        self.assertEqual(kind, 'training_data')
        self.assertEqual(len(training), 2)
        self.assertEqual(len(validation), 2)

    def test_rare_character_stays_in_training(self):
        training, _, _ = personalization.split_samples(samples(['ab', 'ba', 'aba', 'bab', 'Z']))
        self.assertIn('Z', ''.join(text for _, text in training))

    def test_best_epoch_is_restored_after_later_regression_and_early_stop(self):
        model = Model([5, 4, 2, 3, 4, 5, 6])
        with patch('ai.training.personalization.validate_profile', return_value={'passed': 4, 'failed': []}):
            history, report = personalization.fit_personalized(model, samples(['ab']), epochs=6, fit_fn=trainer)
        self.assertEqual(len(history), 5)
        self.assertEqual(report['selected_epoch'], 2)
        self.assertEqual(report['best_validation_loss'], 2)
        self.assertTrue(report['early_stopped'])
        self.assertEqual(model.trainable_variables[0].value[0], 2)

    def test_lower_loss_broken_generation_is_not_selected(self):
        model = Model([5, 3, 1])
        def check(path):
            with np.load(path) as weights:
                broken = weights['epoch'][0] == 2
            return {'passed': 0 if broken else 4, 'failed': ['ab'] if broken else []}
        with patch('ai.training.personalization.validate_profile', side_effect=check):
            _, report = personalization.fit_personalized(model, samples(['ab']), epochs=2, fit_fn=trainer)
        self.assertEqual(report['selected_epoch'], 1)
        self.assertEqual(model.trainable_variables[0].value[0], 1)

    def test_source_checkpoint_is_kept_when_training_never_improves(self):
        model = Model([1, 2, 3])
        with patch('ai.training.personalization.validate_profile', return_value={'passed': 4, 'failed': []}):
            _, report = personalization.fit_personalized(model, samples(['ab']), epochs=2, fit_fn=trainer)
        self.assertEqual(report['selected_epoch'], 0)
        self.assertEqual(model.trainable_variables[0].value[0], 0)

    def test_no_working_checkpoint_raises_and_restores_source(self):
        model = Model([5, 4])
        with patch('ai.training.personalization.validate_profile', return_value={'passed': 0, 'failed': ['ab']}):
            with self.assertRaisesRegex(ValueError, 'checkpoint'):
                personalization.fit_personalized(model, samples(['ab']), epochs=1, fit_fn=trainer)
        self.assertEqual(model.trainable_variables[0].value[0], 0)


if __name__ == '__main__':
    unittest.main()
