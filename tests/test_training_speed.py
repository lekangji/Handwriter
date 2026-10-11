import os
import tempfile
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from ai.training import torch_finetune as training
    if __package__:
        from .training_reference import reference_loss
    else:
        from training_reference import reference_loss


@unittest.skipIf(torch is None, 'PyTorch is not installed')
class TrainingSpeedTests(unittest.TestCase):
    def samples(self):
        return [
            (np.array([[0, 0, 0], [0.4, 0.7, 1]], np.float32), 'a'),
            (np.array([[0, 0, 0], [0.4, 0.7, 0], [0.3, -0.2, 1],
                       [0.2, 0.5, 0], [0.1, -0.4, 1]], np.float32), 'abZ'),
        ]

    def long_sample(self):
        t = np.arange(65, dtype=np.float32)
        offsets = np.column_stack((0.3 + 0.1 * np.cos(t / 7),
                                   0.4 * np.sin(t / 9), t % 9 == 0)).astype(np.float32)
        offsets[0] = 0
        return offsets, 'Hello abZ'

    def assert_loss_and_gradients_match(self, device):
        alphabet = training.load_profile_alphabet(training.WEIGHTS_PATH) + ['Z']
        model = training.TorchHandwritingModel(alphabet=alphabet, device=device)
        for offsets, text in self.samples() + [self.long_sample()]:
            with self.subTest(device=device, points=len(offsets)):
                offsets = torch.as_tensor(offsets, device=device)
                text = torch.as_tensor(model.encode_text(text), device=device)
                expected = reference_loss(offsets, text, model.trainable_variables,
                                          training.LSTM_SIZE, training.NUM_ATTN_MIX,
                                          training.NUM_OUT_MIX)
                expected_grads = torch.autograd.grad(expected, model.trainable_variables)
                actual = model.forward_loss(offsets, text)
                actual_grads = torch.autograd.grad(actual, model.trainable_variables)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                for name, actual_grad, expected_grad in zip(training.WEIGHT_KEYS, actual_grads, expected_grads):
                    with self.subTest(weight=name):
                        torch.testing.assert_close(actual_grad, expected_grad, rtol=3e-4, atol=2e-5)

    def test_cpu_loss_and_all_gradients_preserve_reference(self):
        self.assert_loss_and_gradients_match('cpu')

    def test_cuda_loss_and_all_gradients_preserve_reference(self):
        if not torch.cuda.is_available():
            self.skipTest('PyTorch has no CUDA GPU')
        self.assert_loss_and_gradients_match('cuda:0')

    def test_density_reduction_runs_once_for_all_pen_points(self):
        # Detect reintroducing a separate density reduction for each timestep;
        # operator counts are stable, unlike wall-clock assertions in unit tests.
        model = training.TorchHandwritingModel(device='cpu')
        offsets, text = self.samples()[1]
        text = model.encode_text('aba')
        model.forward_loss(offsets, text)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], acc_events=True) as profile:
            model.forward_loss(offsets, text)
        reductions = sum(event.count for event in profile.key_averages()
                         if event.key == 'aten::logsumexp')
        self.assertLessEqual(reductions, 2)

    def test_density_clamps_and_pen_lifts_preserve_reference(self):
        # Small independent fixture forces sigma, correlation and EOS clamps.
        shapes = [(9, 16), (16,), (13, 16), (16,), (13, 16), (16,),
                  (9, 6), (6,), (4, 13), (13,)]
        weights = [torch.zeros(shape, requires_grad=True) for shape in shapes]
        with torch.no_grad():
            weights[-1].copy_(torch.tensor([0, 0, -12, 10, -12, 10, 9, -9, 0, 0, 0, 0, 20]))
        offsets = torch.tensor([[0, 0, 0], [0.1, 0.2, 0], [0.2, -0.1, 1]])
        text = torch.tensor([[0., 1.], [1., 0.]])
        expected = reference_loss(offsets, text, weights, 4, 2, 2)
        expected_grads = torch.autograd.grad(expected, weights)
        actual = training._forward_loss(offsets, text, weights, 4, 2, 2)
        actual_grads = torch.autograd.grad(actual, weights)
        self.assertTrue(torch.isfinite(actual).item())
        torch.testing.assert_close(actual, expected)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_fit_preserves_sample_updates_progress_and_export(self):
        from ai.inference.hw_model import HandwritingModel

        alphabet = training.load_profile_alphabet(training.WEIGHTS_PATH) + ['Z']
        actual = training.TorchHandwritingModel(alphabet=alphabet, device='cpu')
        expected = training.TorchHandwritingModel(alphabet=alphabet, device='cpu')
        samples = self.samples()
        optimizer = torch.optim.Adam(expected.trainable_variables, lr=1e-5, eps=1e-7)
        expected_history = []
        for _ in range(2):
            total = 0.0
            for offsets, text in samples:
                optimizer.zero_grad(set_to_none=True)
                loss = reference_loss(torch.as_tensor(offsets), torch.as_tensor(expected.encode_text(text)),
                                      expected.trainable_variables, training.LSTM_SIZE,
                                      training.NUM_ATTN_MIX, training.NUM_OUT_MIX)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(expected.trainable_variables, 5.0, error_if_nonfinite=True)
                optimizer.step()
                total += loss.detach().item()
            expected_history.append(total / len(samples))
        epoch_events, sample_events = [], []
        history = training.fit(actual, samples, epochs=2,
                               progress_cb=lambda *event: epoch_events.append(event),
                               sample_progress_cb=sample_events.append)
        np.testing.assert_allclose(history, expected_history, rtol=1e-5, atol=1e-6)
        for actual_weight, expected_weight in zip(actual.trainable_variables, expected.trainable_variables):
            torch.testing.assert_close(actual_weight, expected_weight, rtol=1e-5, atol=2e-6)
        self.assertEqual([event[:2] for event in epoch_events], [(0, 2), (1, 2)])
        self.assertEqual(sample_events[-1]['progress'], 95)
        self.assertEqual(sample_events[-1]['processed_steps'], 10)
        self.assertEqual(len(sample_events[-1]['loss_history']), 2)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'trained.npz')
            actual.export_to_npz(path)
            alphabet = training.load_profile_alphabet(path)
            loaded = HandwritingModel(path, alphabet)
            self.assertEqual(alphabet, actual.alphabet)
            np.testing.assert_array_equal(loaded.gmm_w, actual.gmm_w.detach().numpy())
            np.testing.assert_array_equal(loaded.lstm1.kernel, actual.lstm1_k.detach().numpy())

    def test_epoch_hook_can_stop_training_after_completed_epoch(self):
        model = training.TorchHandwritingModel(device='cpu')
        events = []
        def completed(epoch, loss):
            events.append((epoch, loss))
            return epoch < 2
        history = training.fit(model, self.samples()[:1], epochs=5, epoch_cb=completed)
        self.assertEqual(len(history), 2)
        self.assertEqual([epoch for epoch, _ in events], [1, 2])
        self.assertTrue(np.isfinite(history).all())


if __name__ == '__main__':
    unittest.main()
