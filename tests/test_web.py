import io
import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch
from app import app, _JOBS, _JOB_LOCK, _MODEL_CACHE

class WebTests(unittest.TestCase):
    def setUp(self):
        self.client=app.test_client()
        with _JOB_LOCK: _JOBS.clear()

    def test_pages_and_generation_validation(self):
        for path in ['/', '/train', '/capture-tool', '/profiles']:
            with self.client.get(path) as response:
                self.assertEqual(response.status_code, 200)
                self.assertIn(b'aria-label="Main navigation"', response.data)
                self.assertNotIn(b'brand-dot', response.data)
        train_html = self.client.get('/train').data
        self.assertNotIn(b'extra_chars', train_html)
        self.assertNotIn(b'add unsupported', train_html)
        for path, heading in [('/', b'Write &amp; Export'), ('/train', b'Personalize'), ('/capture-tool', b'Capture samples'), ('/profiles', b'Profiles')]:
            with self.client.get(path) as response:
                self.assertIn(b'<h1>' + heading + b'</h1>', response.data)
        self.assertGreater(train_html.index(b'id="progressBox"'), train_html.index(b'<h2>Result</h2>'))
        for fields in [{'text':''}, {'text':'Hi','profile':'../default'}, {'text':'Hi','line_spacing_mm':'nan'}, {'text':'Hi','chunk_chars':'0'}, {'text':'Hi','page_width_mm':'20'}]:
            self.assertEqual(self.client.post('/generate',data=fields).status_code, 400)

    def test_profile_management_preserves_weights_alphabet_and_cache(self):
        with tempfile.TemporaryDirectory() as directory, patch('app.PROFILES_DIR', directory):
            for name in ['default', 'personal']:
                for suffix, content in [('.npz', b'model'), ('.alphabet.json', b'["a"]')]:
                    with open(os.path.join(directory, name + suffix), 'wb') as stream:
                        stream.write(content)
            _MODEL_CACHE['personal'] = object()
            try:
                response = self.client.post('/profiles', data={'profile': 'personal', 'action': 'rename', 'profile_name': 'renamed'})
                self.assertEqual(response.status_code, 302)
                self.assertNotIn('personal', _MODEL_CACHE)
                for suffix, content in [('.npz', b'model'), ('.alphabet.json', b'["a"]')]:
                    with open(os.path.join(directory, 'renamed' + suffix), 'rb') as stream:
                        self.assertEqual(stream.read(), content)
                    self.assertFalse(os.path.exists(os.path.join(directory, 'personal' + suffix)))
                for path in ['/', '/train', '/profiles']:
                    self.assertIn(b'renamed', self.client.get(path).data)
                _MODEL_CACHE['renamed'] = object()
                self.assertEqual(self.client.post('/profiles', data={'profile': 'renamed', 'action': 'delete'}).status_code, 302)
                self.assertNotIn('renamed', _MODEL_CACHE)
                self.assertEqual(sorted(os.listdir(directory)), ['default.alphabet.json', 'default.npz'])
            finally:
                _MODEL_CACHE.pop('personal', None)
                _MODEL_CACHE.pop('renamed', None)

    def test_profile_management_rejects_unsafe_changes_and_running_jobs(self):
        with tempfile.TemporaryDirectory() as directory, patch('app.PROFILES_DIR', directory):
            for name in ['default', 'personal']:
                with open(os.path.join(directory, name + '.npz'), 'wb') as stream:
                    stream.write(b'model')
            for fields in [
                {'profile': 'default', 'action': 'delete'},
                {'profile': 'default', 'action': 'rename', 'profile_name': 'renamed'},
                {'profile': '../personal', 'action': 'delete'},
                {'profile': 'personal', 'action': 'rename', 'profile_name': '../escape'},
                {'profile': 'personal', 'action': 'rename', 'profile_name': 'default'},
                {'profile': 'personal', 'action': 'rename', 'profile_name': 'personal'},
                {'profile': 'personal', 'action': 'unexpected'},
            ]:
                self.assertEqual(self.client.post('/profiles', data=fields).status_code, 400)
            with _JOB_LOCK:
                _JOBS['active'] = {'status': 'training'}
            self.assertEqual(self.client.post('/profiles', data={'profile': 'personal', 'action': 'delete'}).status_code, 400)
            self.assertEqual(sorted(os.listdir(directory)), ['default.npz', 'personal.npz'])

    def test_profile_rename_restores_alphabet_when_weights_cannot_move(self):
        with tempfile.TemporaryDirectory() as directory, patch('app.PROFILES_DIR', directory):
            for suffix in ['.npz', '.alphabet.json']:
                with open(os.path.join(directory, 'personal' + suffix), 'wb') as stream:
                    stream.write(b'original')
            rename = os.rename
            def fail_weights(source, target):
                if source.endswith('.npz'):
                    raise OSError('File locked')
                rename(source, target)
            with patch('app.os.rename', side_effect=fail_weights):
                response = self.client.post('/profiles', data={'profile': 'personal', 'action': 'rename', 'profile_name': 'renamed'})
            self.assertEqual(response.status_code, 400)
            self.assertEqual(sorted(os.listdir(directory)), ['personal.alphabet.json', 'personal.npz'])

    def test_generation_returns_before_work_finishes_and_polls_result(self):
        from document.layout import Document, Page
        def generate(*args, **kwargs):
            time.sleep(.12)
            kwargs['progress_cb'](1,1)
            return Document([Page(210,297)])
        with patch('app.generate_document',side_effect=generate), patch('app.document_to_svg',return_value=['page_page1.svg']), patch('app.document_to_png',return_value=['page_page1.png']), patch('app.generate_gcode',return_value='G21'), patch('builtins.open',unittest.mock.mock_open()), patch('app._cleanup_old_outputs'), patch('app.get_model',return_value=type('Model',(),{'alphabet':list('Hello ')})()), patch('app.os.makedirs'):
            response=self.client.post('/generate',data={'text':'Hello','lined':'on'})
            self.assertEqual(response.status_code,202)
            url=response.json['progress_url']
            self.assertNotEqual(self.client.get(url).json['status'],'done')
            self.assertEqual(self.client.post('/generate',data={'text':'Hello'}).status_code,400)
            for _ in range(100):
                result=self.client.get(url).json
                if result['status'] in ('done','error'):break
                time.sleep(.01)
            self.assertEqual(result['status'],'done',result)
            self.assertEqual(result['result']['num_pages'],1)
            self.assertEqual(result['progress'],100)

    def test_failed_background_job_has_visible_error(self):
        with patch('app.get_model',side_effect=ValueError('Unreadable profile')), patch('app._cleanup_old_outputs'):
            response=self.client.post('/generate',data={'text':'Hello'})
            for _ in range(100):
                result=self.client.get(response.json['progress_url']).json
                if result['status']=='error':break
                time.sleep(.01)
            self.assertEqual(result['error'],'Unreadable profile')

    def test_training_preserves_existing_profile_and_validates_upload(self):
        self.assertEqual(self.client.post('/train/run',data={'profile_name':'default','samples_json':(io.BytesIO(b'{}'),'samples.json')}).status_code,400)
        self.assertEqual(self.client.post('/train/run',data={'profile_name':'new_profile','samples_json':(io.BytesIO(b'{}'),'samples.json')}).status_code,400)
        self.assertEqual(self.client.get('/jobs/missing').status_code,404)

    def test_training_adds_uploaded_characters_without_manual_input(self):
        data = json.dumps({'calibration_px_per_model_unit': 2, 'samples': [
            {'text': 'Z&→', 'strokes': [[[0, 0], [4, 2]]]},
        ]}).encode()
        fields = lambda: {'profile_name': 'new_profile', 'samples_json': (io.BytesIO(data), 'samples.json')}
        checked = self.client.post('/train/check-alphabet', data=fields())
        self.assertEqual(checked.status_code, 200)
        self.assertEqual(checked.json['missing_chars'], [])
        def start(kind, work):
            work('test', lambda **values: None)
            return 'test'
        with patch('app._start_job', side_effect=start), patch('ai.training.finetune.create_training_model') as model, patch('ai.training.finetune.fit', return_value=[1.0]), patch('app.save_validated_profile', return_value={'passed': 4, 'failed': []}):
            model.return_value.device = '/CPU:0'
            response = self.client.post('/train/run', data=fields())
        self.assertEqual(response.status_code, 202, response.json)
        self.assertTrue(set('Z&→').issubset(model.call_args.kwargs['alphabet']))

    def test_training_rejects_invalid_device_before_starting_job(self):
        data = json.dumps({'calibration_px_per_model_unit': 2, 'samples': [
            {'text': 'hi', 'strokes': [[[0, 0], [4, 2]]]},
        ]}).encode()
        response = self.client.post('/train/run', data={
            'profile_name': 'new_profile', 'device': 'cuda:-1',
            'samples_json': (io.BytesIO(data), 'samples.json'),
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn('Training device', response.json['error'])

    def test_training_reports_selected_external_gpu(self):
        data = json.dumps({'calibration_px_per_model_unit': 2, 'samples': [
            {'text': 'hi', 'strokes': [[[0, 0], [4, 2]]]},
        ]}).encode()
        result = {}
        updates = []
        def start(kind, work):
            result.update(work('test', lambda **values: updates.append(values)))
            return 'test'
        with patch('app._start_job', side_effect=start), patch('ai.training.finetune.create_training_model') as model, patch('ai.training.finetune.fit', return_value=[1.0]), patch('app.save_validated_profile', return_value={'passed': 4, 'failed': []}):
            model.return_value.device = '/GPU:1'
            response = self.client.post('/train/run', data={
                'profile_name': 'new_profile', 'device': 'cuda:1',
                'samples_json': (io.BytesIO(data), 'samples.json'),
            })
        self.assertEqual(response.status_code, 202, response.json)
        self.assertEqual(model.call_args.kwargs['device'], 'cuda:1')
        self.assertEqual(result['device'], '/GPU:1')
        self.assertIn({'device': '/GPU:1'}, updates)

if __name__=='__main__':unittest.main()
