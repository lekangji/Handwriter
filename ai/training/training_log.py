"""Consistent, bounded progress logs for both training backends."""
import logging
import time

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


logger = logging.getLogger('ai.training')


class TrainingLog:
    def __init__(self, model, samples, epochs, lr, log_every, sample_progress_cb=None):
        self.epochs = epochs
        self.log_every = log_every
        self.sample_count = len(samples)
        self.steps = sum(len(offsets) - 1 for offsets, _ in samples)
        self.started = time.monotonic()
        self.last_report = self.started
        self.previous_elapsed = 0.0
        self.previous_loss = None
        self.best_loss = float('inf')
        self.loss_history = []
        self.sample_progress_cb = sample_progress_cb
        self.last_update = self.started
        self.step_ends = [0]
        for offsets, _ in samples:
            self.step_ends.append(self.step_ends[-1] + len(offsets) - 1)
        self.bar = (tqdm(total=epochs * self.sample_count, desc='Training', unit='sample',
                         mininterval=0.5, leave=False, disable=None) if tqdm else None)
        logger.info(
            'Training started backend=%s device=%s device_name=%s samples=%d '
            'steps=%d characters=%d alphabet=%d epochs=%d lr=%g',
            getattr(model, 'backend', 'tensorflow'), getattr(model, 'device', '/CPU:0'),
            getattr(model, 'device_name', getattr(model, 'device', '/CPU:0')),
            self.sample_count, self.steps, sum(len(text) for _, text in samples),
            len(model.alphabet), epochs, lr,
        )
        self._emit(0, 0, 0.0, self.started)

    def _emit(self, epoch, completed, total_loss, now):
        if self.sample_progress_cb is None:
            return
        elapsed = max(0.0, now - self.started)
        processed = epoch * self.steps + self.step_ends[completed]
        total_steps = self.epochs * self.steps
        rate = processed / elapsed if elapsed > 0 else 0.0
        self.sample_progress_cb({
            'epoch': epoch + 1, 'epochs': self.epochs,
            'sample': completed, 'samples': self.sample_count,
            'processed_steps': processed, 'total_steps': total_steps,
            'progress': round(95 * (epoch * self.sample_count + completed)
                              / (self.epochs * self.sample_count), 2),
            'loss': round(total_loss / completed, 4) if completed else None,
            'best_loss': None if self.best_loss == float('inf') else round(self.best_loss, 4),
            'loss_history': list(self.loss_history),
            'training_elapsed_s': round(elapsed, 1),
            'eta_s': round((total_steps - processed) / rate, 1) if rate else None,
            'steps_per_s': round(rate, 2),
        })
        self.last_update = now

    def sample(self, epoch, completed, total_loss, now=None):
        now = time.monotonic() if now is None else now
        if self.bar is not None:
            self.bar.set_description(f'Epoch {epoch + 1}/{self.epochs}', refresh=False)
            self.bar.set_postfix(nll=f'{total_loss / completed:.4f}', refresh=False)
            self.bar.update(1)
        if now - self.last_update >= 0.5 or completed == self.sample_count:
            self._emit(epoch, completed, total_loss, now)
        if self.bar is not None and not self.bar.disable:
            return
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
        self.loss_history.append(round(loss, 4))
        self._emit(epoch, self.sample_count, loss * self.sample_count, self.started + elapsed)
        if epoch % self.log_every == 0 or epoch == self.epochs - 1:
            if self.bar is not None:
                self.bar.clear()
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
        self.close()
        logger.info('Training finished epochs=%d nll_start=%.4f nll_end=%.4f '
                    'best_nll=%.4f elapsed_s=%.1f',
                    len(history), history[0], history[-1], min(history), elapsed)

    def close(self):
        if self.bar is not None:
            self.bar.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
