"""Exercise actual checkpoint selection, publication and primed CUDA inference together."""
import os
import tempfile
import unittest
import numpy as np

from ai.inference.hw_model import HandwritingModel
from ai.training.personalization import fit_personalized
from ai.training.style import load_style
from ai.training.validation import save_validated_profile
from ai.training.samples import load_profile_alphabet


class AIPipelineTests(unittest.TestCase):
    def test_generated_captures_train_select_publish_and_prime_an_unseen_phrase(self):
        import torch
        from ai.training.torch_finetune import TorchHandwritingModel
        from ai.inference.runtime import create_model
        source = HandwritingModel()
        samples = []
        for index, text in enumerate(['hello', 'world', 'hello world', 'Dear', 'Dear team']):
            points = source.sample(text, bias=1.5, seed=7 + index)
            samples.append((np.vstack([np.zeros((1, 3), np.float32), points]), text))
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = TorchHandwritingModel(device=device)
        history, selection = fit_personalized(model, samples, epochs=1)
        self.assertEqual(len(history), 1)
        self.assertEqual(selection['validation_kind'], 'held_out')
        self.assertIn(selection['selected_epoch'], (0, 1))
        self.assertLessEqual(selection['best_validation_loss'], selection['validation_history'][0]['loss'])
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'personal.npz')
            report = save_validated_profile(model, path, samples=samples, metadata=selection)
            self.assertEqual(report['passed'], 4)
            alphabet = load_profile_alphabet(path)
            primer, text = load_style(path, alphabet)[0]
            inference = create_model(path, alphabet, device=device)
            result = inference.sample('Dear world', bias=1.5, seed=19,
                                      prime_text=text, prime_strokes=primer)
            self.assertTrue(np.isfinite(result).all())
            self.assertEqual(result[-1, 2], 1)


if __name__ == '__main__':
    unittest.main()
