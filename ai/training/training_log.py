"""Consistent, bounded progress logs for both training backends."""
import logging
import time


logger = logging.getLogger('ai.training')


class TrainingLog:
    def __init__(self, model, samples, epochs, lr, log_every):
        self.epochs = epochs
        self.log_every = log_every
        self.sample_count = len(samples)
        self.steps = sum(len(offsets) - 1 for offsets, _ in samples)
        self.started = time.monotonic()
        self.last_report = self.started
        self.previous_elapsed = 0.0
        self.previous_loss = None
        self.best_loss = float('inf')
        logger.info(
            'Training started backend=%s device=%s device_name=%s samples=%d '
            'steps=%d characters=%d alphabet=%d epochs=%d lr=%g',
            getattr(model, 'backend', 'tensorflow'), getattr(model, 'device', '/CPU:0'),
            getattr(model, 'device_name', getattr(model, 'device', '/CPU:0')),
            self.sample_count, self.steps, sum(len(text) for _, text in samples),
            len(model.alphabet), epochs, lr,
        )

    def sample(self, epoch, completed, total_loss, now=None):
        now = time.monotonic() if now is None else now
        # Long epochs still show activity, without printing every sample.
        if now - self.last_report < 30 or completed == self.sample_count:
            return
        self.last_report = now
        logger.info('Training progress epoch=%d/%d sample=%d/%d avg_nll=%.4f elapsed_s=%.1f',
                    epoch + 1, self.epochs, completed, self.sample_count,
                    total_loss / completed, now - self.started)

    def epoch(self, epoch, loss, elapsed, eta):
        duration = elapsed - self.previous_elapsed
        delta = 0.0 if self.previous_loss is None else loss - self.previous_loss
        self.best_loss = min(self.best_loss, loss)
        if epoch % self.log_every == 0 or epoch == self.epochs - 1:
            logger.info(
                'Training epoch=%d/%d nll=%.4f delta=%+.4f best_nll=%.4f '
                'epoch_s=%.1f elapsed_s=%.1f eta_s=%.1f steps_per_s=%.1f',
                epoch + 1, self.epochs, loss, delta, self.best_loss,
                duration, elapsed, eta, self.steps / max(duration, 1e-9),
            )
        self.previous_elapsed = elapsed
        self.previous_loss = loss
        self.last_report = self.started + elapsed

    def finish(self, history, elapsed):
        logger.info('Training finished epochs=%d nll_start=%.4f nll_end=%.4f '
                    'best_nll=%.4f elapsed_s=%.1f',
                    len(history), history[0], history[-1], min(history), elapsed)
