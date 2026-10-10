import json
import os
import tempfile
import unittest

from ai.inference.hw_model import HandwritingModel
from ai.training.samples import load_profile_alphabet
from ai.training.validation import validate_profile, save_validated_profile


ROOT = os.path.dirname(os.path.dirname(__file__))
SOURCE = os.path.join(ROOT, 'profiles', 'default.npz')


class ProfileValidationTests(unittest.TestCase):
    def test_working_profile_passes_free_running_checks(self):
        report = validate_profile(SOURCE)
        self.assertEqual(report['passed'], 4)
        self.assertEqual(report['failed'], [])

    def test_degenerate_export_is_not_published(self):
        # Actual inference on a zero-output checkpoint reaches neither readable
        # geometry nor the end marker. The exporter stands in for GPU training.
        import numpy as np
        class DegenerateModel:
            def export_to_npz(self, path):
                with np.load(SOURCE) as checkpoint:
                    np.savez(path, **{key: np.zeros_like(checkpoint[key]) for key in checkpoint.files})
                with open(os.path.splitext(path)[0] + '.alphabet.json', 'w') as stream:
                    json.dump(load_profile_alphabet(SOURCE), stream)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'bad.npz')
            with self.assertLogs('ai.training.validation', level='WARNING') as logs:
                with self.assertRaisesRegex(ValueError, 'generation check'):
                    save_validated_profile(DegenerateModel(), path)
            self.assertEqual(len(logs.records), 4)
            self.assertEqual(os.listdir(directory), [])

    def test_valid_export_can_be_loaded_and_existing_profile_is_preserved(self):
        import shutil
        class WorkingModel:
            def export_to_npz(self, path):
                shutil.copyfile(SOURCE, path)
                with open(os.path.splitext(path)[0] + '.alphabet.json', 'w') as stream:
                    json.dump(load_profile_alphabet(SOURCE), stream)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'working.npz')
            report = save_validated_profile(WorkingModel(), path)
            self.assertEqual(report['passed'], 4)
            model = HandwritingModel(path, load_profile_alphabet(path))
            self.assertGreater(len(model.sample('Dear', bias=1.5, seed=2029167940)), 1)
            with open(path, 'rb') as stream:
                before = stream.read()
            with self.assertRaisesRegex(ValueError, 'already exists'):
                save_validated_profile(WorkingModel(), path)
            with open(path, 'rb') as stream:
                self.assertEqual(stream.read(), before)


if __name__ == '__main__':
    unittest.main()
