import json
import os
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from ai.inference.hw_model import HandwritingModel
from ai.training.validation import save_validated_profile
from ai.training import style


class StylePrimingTests(unittest.TestCase):
    def samples(self):
        return [(np.array([[0, 0, 0], [1, 2, 0], [1, -2, 1]], np.float32), 'Hello')]

    def test_profile_saves_and_loads_captured_model_space_strokes(self):
        from ai.training.samples import load_profile_alphabet
        source = os.path.abspath('profiles/default.npz')
        class Exporter:
            def export_to_npz(self, path):
                import shutil
                shutil.copyfile(source, path)
                with open(os.path.splitext(path)[0] + '.alphabet.json', 'w') as stream:
                    json.dump(load_profile_alphabet(source), stream)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'personal.npz')
            with patch('ai.training.validation.validate_profile', return_value={'passed': 4, 'failed': []}):
                save_validated_profile(Exporter(), path, samples=self.samples(), metadata={'selected_epoch': 2})
            loaded = style.load_style(path, load_profile_alphabet(path))
            self.assertEqual(loaded[0][1], 'Hello')
            np.testing.assert_array_equal(loaded[0][0], self.samples()[0][0])
            with open(os.path.splitext(path)[0] + '.training.json') as stream:
                self.assertEqual(json.load(stream)['selected_epoch'], 2)

    def test_style_rejects_incompatible_or_malformed_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'personal.npz')
            for sample in [{'text': 'Z', 'offsets': [[0, 0, 0], [1, 1, 1]]},
                           {'text': 'a', 'offsets': [[0, 0, 0], [1, float('nan'), 1]]},
                           {'text': 'a', 'offsets': [[0, 0, 0], [1, 1, 0.2]]}]:
                with open(os.path.splitext(path)[0] + '.style.json', 'w') as stream:
                    json.dump({'version': 1, 'samples': [sample]}, stream)
                with self.assertRaises(ValueError):
                    style.load_style(path, ['\0', 'a'])

    def test_old_profiles_work_without_style_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(style.load_style(os.path.join(directory, 'old.npz'), ['\0', 'a']), [])

    def test_saved_style_never_silently_overwrites_an_existing_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'personal.npz')
            with open(os.path.splitext(path)[0] + '.style.json', 'w') as stream:
                stream.write('existing')
            with self.assertRaisesRegex(ValueError, 'exists'):
                save_validated_profile(None, path, samples=self.samples())
            with open(os.path.splitext(path)[0] + '.style.json') as stream:
                self.assertEqual(stream.read(), 'existing')

    def test_priming_prefix_is_passed_to_completion_checks(self):
        model = HandwritingModel.__new__(HandwritingModel)
        model.alphabet = list('\0 ab')
        model.char_to_idx = {c: i for i, c in enumerate(model.alphabet)}
        with patch.object(model, '_sample_once', return_value=(np.array([[4, 0, 1]], np.float32), True)) as draw:
            model.sample('b', prime_text='a', prime_strokes=np.array([[0, 0, 0], [4, 0, 1]], np.float32))
            self.assertEqual(draw.call_args.kwargs['prefix_chars'], 2)

    def test_invalid_priming_transcription_is_rejected_clearly(self):
        model = HandwritingModel.__new__(HandwritingModel)
        model.alphabet = list('\0 ab')
        model.char_to_idx = {c: i for i, c in enumerate(model.alphabet)}
        with self.assertRaisesRegex(ValueError, 'Priming'):
            model.sample('b', prime_text='Z', prime_strokes=self.samples()[0][0])

    def test_unusable_primers_are_filtered_before_a_profile_uses_them(self):
        from ai.training.samples import load_profile_alphabet
        source = os.path.abspath('profiles/default.npz')
        class Exporter:
            def export_to_npz(self, path):
                import shutil
                shutil.copyfile(source, path)
                with open(os.path.splitext(path)[0] + '.alphabet.json', 'w') as stream:
                    json.dump(load_profile_alphabet(source), stream)
        captures = self.samples() + [(self.samples()[0][0], 'world')]
        def checks(path, prime_strokes=None, prime_text=None):
            failed = ['Dear'] if prime_text == 'Hello' else []
            return {'passed': 0 if failed else 4, 'failed': failed}
        with tempfile.TemporaryDirectory() as directory, patch('ai.training.validation.validate_profile', side_effect=checks):
            path = os.path.join(directory, 'personal.npz')
            report = save_validated_profile(Exporter(), path, samples=captures)
            retained = style.load_style(path, load_profile_alphabet(path))
            self.assertEqual([text for _, text in retained], ['world'])
            self.assertEqual(report['style_samples_retained'], 1)

    def test_working_weights_remain_usable_when_no_primer_passes(self):
        from ai.training.samples import load_profile_alphabet
        source = os.path.abspath('profiles/default.npz')
        class Exporter:
            def export_to_npz(self, path):
                import shutil
                shutil.copyfile(source, path)
                with open(os.path.splitext(path)[0] + '.alphabet.json', 'w') as stream:
                    json.dump(load_profile_alphabet(source), stream)
        def checks(path, prime_strokes=None, prime_text=None):
            return {'passed': 0 if prime_text else 4, 'failed': ['Dear'] if prime_text else []}
        with tempfile.TemporaryDirectory() as directory, patch('ai.training.validation.validate_profile', side_effect=checks):
            path = os.path.join(directory, 'personal.npz')
            report = save_validated_profile(Exporter(), path, samples=self.samples())
            self.assertTrue(os.path.exists(path))
            self.assertEqual(style.load_style(path, load_profile_alphabet(path)), [])
            self.assertTrue(report['style_warning'])


if __name__ == '__main__':
    unittest.main()
