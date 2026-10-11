import unittest
from unittest.mock import patch
import numpy as np

from ai.inference import runtime
from ai.inference.hw_model import HandwritingModel, WEIGHTS_PATH


class InferenceDeviceTests(unittest.TestCase):
    def test_auto_falls_back_and_explicit_cuda_fails_without_gpu(self):
        with patch('ai.inference.runtime.cuda_count', return_value=0):
            self.assertEqual(runtime.resolve_device('auto'), 'cpu')
            with self.assertRaisesRegex(ValueError, 'CUDA'):
                runtime.resolve_device('cuda')

    def test_visible_gpu_index_is_honored(self):
        with patch('ai.inference.runtime.cuda_count', return_value=2):
            self.assertEqual(runtime.resolve_device('auto'), 'cuda:0')
            self.assertEqual(runtime.resolve_device('cuda:1'), 'cuda:1')
            with self.assertRaises(ValueError):
                runtime.resolve_device('cuda:2')
            self.assertEqual(runtime.resolve_device('cpu'), 'cpu')

    def test_invalid_device_never_starts_generation(self):
        for device in ('cuda:-1', 'gpu', 'cpu:1', None):
            with self.subTest(device=device), self.assertRaises(ValueError):
                runtime.resolve_device(device)

    def test_torch_cpu_sampler_is_repeatable_and_completes(self):
        from ai.inference.torch_model import TorchInferenceModel
        model = TorchInferenceModel(WEIGHTS_PATH, device='cpu')
        a = model.sample('Hello', bias=1.5, seed=7)
        b = model.sample('Hello', bias=1.5, seed=7)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.isfinite(a).all())
        self.assertEqual(a[-1, 2], 1)

    def test_torch_recurrent_step_matches_numpy_with_existing_weights(self):
        import torch
        from ai.inference.torch_model import TorchInferenceModel, _advance
        numpy_model = HandwritingModel()
        model = TorchInferenceModel(WEIGHTS_PATH, device='cpu')
        text = numpy_model._encode_text('Hello')
        state = model.initial_state()
        h1 = c1 = h2 = c2 = h3 = c3 = np.zeros(400, np.float32)
        kappa, window = np.zeros(10, np.float32), np.zeros(len(model.alphabet), np.float32)
        for x in ([0, 0, 1], [1, 2, 0], [2, -1, 1]):
            x = np.array(x, np.float32)
            h1, c1 = numpy_model.lstm1(np.concatenate([window, x]), h1, c1)
            from ai.inference.hw_model import _softplus
            alpha, beta, delta = np.split(_softplus(np.concatenate([window, x, h1]) @ numpy_model.attn_w + numpy_model.attn_b), 3)
            kappa += delta / 25
            phi = np.sum(alpha[:, None] * np.exp(-(kappa[:, None] - np.arange(len(text))) ** 2 / np.maximum(beta[:, None], .01)), axis=0)
            window = phi @ text
            h2, c2 = numpy_model.lstm2(np.concatenate([x, h1, window]), h2, c2)
            h3, c3 = numpy_model.lstm3(np.concatenate([x, h2, window]), h3, c3)
            state[8] = torch.as_tensor(x)
            state, actual_phi, params = _advance(torch.as_tensor(text), model.weights, state)
            np.testing.assert_allclose(actual_phi.numpy(), phi, rtol=1e-4, atol=1e-5)
            np.testing.assert_allclose(params.numpy(), h3 @ numpy_model.gmm_w + numpy_model.gmm_b, rtol=2e-4, atol=1e-4)

    def test_cuda_sampler_completes_with_profile_priming(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA hardware is unavailable')
        from ai.inference.torch_model import TorchInferenceModel
        model = TorchInferenceModel(WEIGHTS_PATH, device='cuda:0')
        primer = model.sample('Hello', bias=1.5, seed=7)
        primer = np.vstack([np.zeros((1, 3), np.float32), primer])
        result = model.sample('world', bias=1.5, seed=19, prime_text='Hello', prime_strokes=primer)
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(result[-1, 2], 1)

    def test_cuda_graph_replay_refreshes_text_randoms_and_recurrent_state(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA hardware is unavailable')
        from ai.inference.torch_model import TorchInferenceModel, _draw_block
        model = TorchInferenceModel(WEIGHTS_PATH, device='cuda:0')
        with torch.inference_mode():
            for seed, phrase in [(7, 'Hello'), (19, 'world'), (7, 'Hello')]:
                text = torch.as_tensor(model._encode_text(phrase), device='cuda:0')
                rng = np.random.default_rng(seed)
                randoms = np.column_stack([rng.random(32), rng.standard_normal((32, 2)), rng.random(32)]).astype(np.float32)
                randoms = torch.as_tensor(randoms, device='cuda:0')
                state = model.initial_state(len(text))
                expected_state, expected_draws, expected_checks = _draw_block(text, model.weights, state, randoms, 1.5, 100)
                actual_state, actual_draws, actual_checks = model._block(text, state, randoms, 1.5, 100)
                torch.testing.assert_close(actual_draws, expected_draws)
                torch.testing.assert_close(actual_checks, expected_checks)
                for actual, expected in zip(actual_state, expected_state):
                    torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
