import logging
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ai.logging_utils import QuietAccessFilter
from ai.training.training_log import TrainingLog


class LoggingTests(unittest.TestCase):
    def setUp(self):
        ai_logger = logging.getLogger('ai')
        self.addCleanup(ai_logger.setLevel, ai_logger.level)

    def test_polling_and_static_success_are_quiet_but_errors_remain(self):
        access_filter = QuietAccessFilter()
        cases = [
            ('GET /jobs/abc HTTP/1.1', 200, False),
            ('GET /train/progress/abc?poll=1 HTTP/1.1', 200, False),
            ('GET /static/style.css HTTP/1.1', 304, False),
            ('GET /jobs/abc HTTP/1.1', 404, True),
            ('GET /static/missing.css HTTP/1.1', 500, True),
            ('POST /train/run HTTP/1.1', 202, True),
            ('GET /train HTTP/1.1', 200, True),
        ]
        logging.getLogger('ai').setLevel(logging.INFO)
        for request, status, visible in cases:
            record = logging.LogRecord('werkzeug', logging.INFO, '', 0,
                                       '%s - - [%s] %s',
                                       ('127.0.0.1', 'now', f'"{request}" \x1b[36m{status}\x1b[0m -'), None)
            with self.subTest(request=request, status=status):
                self.assertEqual(access_filter.filter(record), visible)
        warning = logging.LogRecord('werkzeug', logging.WARNING, '', 0, 'server warning', (), None)
        self.assertTrue(access_filter.filter(warning))
        logging.getLogger('ai').setLevel(logging.DEBUG)
        record = logging.LogRecord('werkzeug', logging.INFO, '', 0,
                                   '"GET /jobs/abc HTTP/1.1" 200 -', (), None)
        self.assertTrue(access_filter.filter(record))

    def test_training_metrics_and_interval_use_all_epochs(self):
        model = SimpleNamespace(backend='torch', device='cuda:1', device_name='Test GPU', alphabet=['a', 'b'])
        samples = [([[0, 0, 0]] * 4, 'ab'), ([[0, 0, 0]] * 3, 'a')]
        with patch('ai.training.training_log.time.monotonic', return_value=100), self.assertLogs('ai.training', 'INFO') as captured:
            reporter = TrainingLog(model, samples, epochs=3, lr=1e-5, log_every=2)
            reporter.epoch(0, 3.0, 10.0, 20.0)
            reporter.epoch(1, 1.0, 30.0, 15.0)
            reporter.epoch(2, 2.0, 40.0, 0.0)
            reporter.finish([3.0, 1.0, 2.0], 40.0)
        messages = [record.getMessage() for record in captured.records]
        self.assertEqual(len(messages), 4)
        for detail in ('backend=torch', 'device=cuda:1', 'Test GPU', 'samples=2', 'steps=5', 'characters=3', 'alphabet=2', 'lr=1e-05'):
            self.assertIn(detail, messages[0])
        for detail in ('epoch=3/3', 'nll=2.0000', 'delta=+1.0000', 'best_nll=1.0000', 'epoch_s=10.0', 'eta_s=0.0'):
            self.assertIn(detail, messages[2])
        self.assertIn('best_nll=1.0000', messages[3])

    def test_sample_progress_is_throttled(self):
        model = SimpleNamespace(device='cpu', alphabet=['a'])
        samples = [([[0, 0, 0]] * 3, 'a')] * 3
        with patch('ai.training.training_log.time.monotonic', return_value=100), self.assertLogs('ai.training', 'INFO') as captured:
            reporter = TrainingLog(model, samples, epochs=2, lr=1e-5, log_every=1)
            reporter.sample(0, 1, 4.0, now=105)
            reporter.sample(0, 2, 6.0, now=131)
            reporter.sample(0, 3, 9.0, now=132)
        self.assertEqual(len(captured.records), 2)
        message = captured.records[1].getMessage()
        self.assertIn('epoch=1/2', message)
        self.assertIn('sample=2/3', message)
        self.assertIn('avg_nll=3.0000', message)

    def test_live_training_progress_tracks_samples_steps_and_epoch_history(self):
        model = SimpleNamespace(device='cpu', alphabet=['a'])
        samples = [([[0, 0, 0]] * 4, 'a'), ([[0, 0, 0]] * 3, 'a')]
        events = []
        with patch('ai.training.training_log.time.monotonic', return_value=100):
            reporter = TrainingLog(model, samples, 2, 1e-5, 1, sample_progress_cb=events.append)
        reporter.sample(0, 1, 4.0, now=102)
        reporter.sample(0, 2, 6.0, now=104)
        reporter.epoch(0, 3.0, 4.0, 4.0)
        reporter.sample(1, 1, 2.0, now=106)
        self.assertEqual(events[0]['sample'], 0)
        completed = events[2]
        self.assertEqual(completed['sample'], 2)
        self.assertEqual(completed['samples'], 2)
        self.assertEqual(completed['processed_steps'], 5)
        self.assertEqual(completed['total_steps'], 10)
        self.assertEqual(completed['loss'], 3.0)
        self.assertEqual(completed['progress'], 47.5)
        self.assertEqual(completed['eta_s'], 4.0)
        self.assertEqual(completed['steps_per_s'], 1.25)
        self.assertEqual(events[-1]['processed_steps'], 8)
        self.assertEqual(events[-1]['loss_history'], [3.0])
        reporter.close()


if __name__ == '__main__':
    unittest.main()
