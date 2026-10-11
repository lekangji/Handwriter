import json
import os
import tempfile
from io import StringIO
import unittest
import warnings
from unittest.mock import mock_open, patch

import numpy as np

try:
    from ai.training import finetune
except ImportError:
    import finetune


class PersonalizationTests(unittest.TestCase):
    def load(self, data):
        return finetune.load_training_samples(StringIO(json.dumps(data)))

    def test_capture_coordinates_scale_and_pen_lifts(self):
        offsets, text = self.load({
            "calibration_px_per_model_unit": 2,
            "resample_model_units": 0.5,
            "samples": [{"text": "hi", "strokes": [[[10, 20], [12, 16]], [[16, 20]]]}],
        })[0]
        np.testing.assert_array_equal(offsets, [[0, 0, 0], [1, 2, 1], [2, -2, 1]])
        self.assertEqual(text, "hi")

    def test_legacy_samples_use_shared_scale_resampling_and_release_endpoints(self):
        samples = [{"text": "a", "points": [
            {"x": 0, "y": 0, "pen_down": True},
            {"x": 0, "y": 50, "pen_down": False},
            {"x": 20, "y": 10, "pen_down": False},
        ]}, {"text": "a", "points": [
            {"x": 0, "y": 0, "pen_down": True},
            {"x": 0, "y": 100, "pen_down": False},
        ]}]
        with warnings.catch_warnings(record=True) as notices:
            loaded = self.load(samples)
        self.assertTrue(notices)
        coords = np.cumsum(loaded[0][0][:, :2], axis=0)
        lifts = np.flatnonzero(loaded[0][0][:, 2])
        np.testing.assert_allclose(coords[lifts], [[0, -50 / 3], [20 / 3, -10 / 3]], atol=1e-5)
        self.assertAlmostEqual(np.abs(loaded[1][0][:, 1].sum()), 100 / 3, places=4)
        self.assertLessEqual(np.linalg.norm(loaded[0][0][1:lifts[0] + 1, :2], axis=1).max(), .50001)

    def test_old_calibrated_samples_preserve_recorded_sampling(self):
        offsets, _ = self.load({"calibration_px_per_model_unit": 2, "samples": [
            {"text": "a", "strokes": [[[0, 0], [4, 0]], [[6, 0]]]},
        ]})[0]
        np.testing.assert_array_equal(offsets, [[0, 0, 0], [2, 0, 1], [1, 0, 1]])

    def test_calibrated_capture_with_other_spacing_preserves_sampling(self):
        offsets, _ = self.load({"calibration_px_per_model_unit": 2,
                               "resample_model_units": 1, "samples": [
            {"text": "a", "strokes": [[[0, 0], [2, 2], [4, 0]]]},
        ]})[0]
        np.testing.assert_array_equal(offsets, [[0, 0, 0], [1, -1, 0], [1, 1, 1]])

    def test_invalid_calibration_and_coordinates(self):
        sample = {"text": "hi", "strokes": [[[0, 0], [1, 1]]]}
        for scale in (None, 0, -1, float("nan"), float("inf"), True):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                self.load({"calibration_px_per_model_unit": scale, "samples": [sample]})
        with self.assertRaises(ValueError):
            self.load({"calibration_px_per_model_unit": 1, "samples": [
                {"text": "hi", "strokes": [[[0, 0], [float("nan"), 1]]]},
            ]})

    def test_growing_alphabet_keeps_profile_order_and_deduplicates(self):
        source = finetune.build_alphabet("&@")
        result = finetune.build_alphabet("@@&Z&", source)
        self.assertEqual(result, source + ["Z"])
        with patch.object(finetune.os.path, "exists", return_value=True), patch("builtins.open", mock_open(read_data=json.dumps(source))):
            self.assertEqual(finetune.load_profile_alphabet("personal.npz"), source)

    def test_sample_characters_are_added_automatically(self):
        source = finetune.build_alphabet("&@")
        result = finetune.build_alphabet(alphabet=source, samples=[(None, "Z→&Z"), (None, "→")])
        self.assertEqual(result, source + ["Z", "→"])

    @unittest.skipIf(finetune.tf is None, "TensorFlow is not installed")
    def test_tensorflow_dynamic_loss_and_gradients(self):
        tf = finetune.tf
        model = finetune.TrainableHandwritingModel.__new__(finetune.TrainableHandwritingModel)
        model.alphabet = ["\0", "a"]
        model.char_to_idx = {"\0": 0, "a": 1}
        model.A = 2
        shapes = {
            "lstm1_k": (9, 16), "lstm1_b": (16,),
            "lstm2_k": (13, 16), "lstm2_b": (16,),
            "lstm3_k": (13, 16), "lstm3_b": (16,),
            "attn_w": (9, 6), "attn_b": (6,),
            "gmm_w": (4, 13), "gmm_b": (13,),
        }
        for name, shape in shapes.items():
            setattr(model, name, tf.Variable(tf.zeros(shape)))
        model.trainable_variables = [getattr(model, name) for name in shapes]
        samples = [(np.array([[0, 0, 0], [1, 1, 0], [1, -1, 1]], np.float32), "a"),
                   (np.array([[0, 0, 0], [1, 1, 0], [1, -1, 0], [1, 0, 1]], np.float32), "aa")]
        with patch.multiple(finetune, LSTM_SIZE=4, NUM_ATTN_MIX=2, NUM_OUT_MIX=2):
            history = finetune.fit(model, samples, epochs=1)
        self.assertTrue(np.isfinite(history).all())
        self.assertTrue(all(np.isfinite(variable.numpy()).all() for variable in model.trainable_variables))

    @unittest.skipIf(finetune.tf is None, 'TensorFlow is not installed')
    def test_tensorflow_epoch_hook_stops_after_completed_epoch(self):
        model = finetune.TrainableHandwritingModel(device='cpu')
        events = []
        def completed(epoch, loss):
            events.append(epoch)
            return False
        data = [(np.array([[0, 0, 0], [1, 1, 1]], np.float32), 'a')]
        history = finetune.fit(model, data, epochs=3, epoch_cb=completed)
        self.assertEqual(len(history), 1)
        self.assertEqual(events, [1])

    def train_and_export_on_device(self, device):
        from ai.inference.hw_model import HandwritingModel
        model = finetune.TrainableHandwritingModel(device=device)
        expected_device = 'GPU:0' if device == 'cuda' else 'CPU:0'
        self.assertTrue(all(expected_device in variable.device for variable in model.trainable_variables))
        before = model.gmm_b.numpy().copy()
        samples = [(np.array([[0, 0, 0], [1, 1, 0], [1, -1, 1]], np.float32), 'a')]
        history = finetune.fit(model, samples, epochs=1)
        self.assertTrue(np.isfinite(history).all())
        self.assertFalse(np.array_equal(before, model.gmm_b.numpy()))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'trained.npz')
            model.export_to_npz(path)
            alphabet = finetune.load_profile_alphabet(path)
            loaded = HandwritingModel(weights_path=path, alphabet=alphabet)
            self.assertEqual(alphabet, model.alphabet)
            np.testing.assert_array_equal(loaded.gmm_b, model.gmm_b.numpy())
            np.testing.assert_array_equal(loaded.lstm1.kernel, model.lstm1_k.numpy())

    @unittest.skipIf(finetune.tf is None, 'TensorFlow is not installed')
    def test_cpu_training_updates_and_exports_compatible_profile(self):
        self.train_and_export_on_device('cpu')

    @unittest.skipIf(finetune.tf is None, 'TensorFlow is not installed')
    def test_cuda_training_updates_and_exports_compatible_profile(self):
        if not finetune.tf.config.list_logical_devices('GPU'):
            self.skipTest('TensorFlow has no CUDA GPU')
        self.train_and_export_on_device('cuda')


if __name__ == "__main__":
    unittest.main()
