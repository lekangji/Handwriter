import io
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import app as web
from ai.inference.hw_model import ALPHABET


class AIWebFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = self.temp.name
        self.patch = patch('app.PROFILES_DIR', self.directory)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.client = web.app.test_client()
        with web._JOB_LOCK:
            web._JOBS.clear()
        web._MODEL_CACHE.clear()
        for name in ['alpha', 'default', 'zeta']:
            shutil.copyfile('profiles/default.npz', os.path.join(self.directory, name + '.npz'))
            with open(os.path.join(self.directory, name + '.alphabet.json'), 'w') as stream:
                json.dump(ALPHABET, stream)
        self.style = {'version': 1, 'samples': [{'text': 'Hello', 'offsets': [[0, 0, 0], [3, 2, 1]]}]}
        for suffix, data in [('.style.json', self.style), ('.training.json', {'selected_epoch': 2})]:
            with open(os.path.join(self.directory, 'alpha' + suffix), 'w') as stream:
                json.dump(data, stream)

    def test_details_and_reordering_reach_all_profile_selectors(self):
        response = self.client.post('/profiles', data={'profile': 'zeta', 'action': 'move_up'})
        self.assertEqual(response.status_code, 302)
        self.assertEqual(web.list_profiles(), ['default', 'zeta', 'alpha'])
        for route in ['/', '/train', '/profiles']:
            html = self.client.get(route).data
            if route != '/profiles':
                self.assertLess(html.index(b'value="default"'), html.index(b'value="zeta"'))
                self.assertLess(html.index(b'value="zeta"'), html.index(b'value="alpha"'))
            else:
                self.assertIn(b'Supported characters', html)
                self.assertIn(b'character', html)
                self.assertIn(b'Selected checkpoint', html)
        self.assertEqual(self.client.post('/profiles', data={'profile': 'default', 'action': 'move_down'}).status_code, 400)

    def test_profile_rename_delete_preserves_sidecars_and_custom_order(self):
        self.client.post('/profiles', data={'profile': 'zeta', 'action': 'move_up'})
        response = self.client.post('/profiles', data={'profile': 'alpha', 'action': 'rename', 'profile_name': 'personal'})
        self.assertEqual(response.status_code, 302)
        self.assertEqual(web.list_profiles(), ['default', 'zeta', 'personal'])
        for suffix in ['.npz', '.alphabet.json', '.style.json', '.training.json']:
            self.assertTrue(os.path.exists(os.path.join(self.directory, 'personal' + suffix)))
            self.assertFalse(os.path.exists(os.path.join(self.directory, 'alpha' + suffix)))
        self.assertEqual(self.client.post('/profiles', data={'profile': 'personal', 'action': 'delete'}).status_code, 302)
        self.assertEqual(web.list_profiles(), ['default', 'zeta'])
        self.assertFalse(any(n.startswith('personal.') for n in os.listdir(self.directory)))

    def test_profile_loading_retains_style_and_caches_by_device(self):
        model = web.get_model('alpha', device='cpu')
        self.assertEqual(model.device, 'cpu')
        self.assertEqual(model.style_samples[0][1], 'Hello')
        self.assertIs(model, web.get_model('alpha', device='cpu'))

    def test_generation_validates_device_before_background_job(self):
        with patch('app._start_job', return_value='test'):
            response = self.client.post('/generate', data={'text': 'Hello', 'generation_device': 'cuda:-1'})
        self.assertEqual(response.status_code, 400)
        self.assertIn('Generation device', response.json['error'])
        self.assertEqual(web._JOBS, {})

    def test_generation_passes_saved_style_to_document_layout(self):
        from document.layout import Document, Page
        model = web.get_model('alpha', device='cpu')
        results = {}
        def start(kind, work):
            results.update(work('test', lambda **values: None))
            return 'test'
        with patch('app._start_job', side_effect=start), patch('app.generate_document', return_value=Document([Page(210, 297)])) as generate, patch('app.document_to_svg', return_value=[]), patch('app.document_to_png', return_value=[]), patch('app.generate_gcode', return_value='G21'), patch('app.DOWNLOADS_DIR', self.directory), patch('app.get_model', return_value=model):
            response = self.client.post('/generate', data={'text': 'Hello', 'profile': 'alpha', 'generation_device': 'cpu'})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(generate.call_args.kwargs['prime_text'], 'Hello')
        np.testing.assert_array_equal(generate.call_args.kwargs['prime_strokes'], [[0, 0, 0], [3, 2, 1]])
        self.assertEqual(results['generation_device'], 'cpu')


if __name__ == '__main__':
    unittest.main()
