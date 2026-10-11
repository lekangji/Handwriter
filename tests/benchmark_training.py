"""Compare full per-sample training steps without saving or changing profiles.

Run from the project root:
    python tests/benchmark_training.py --device cuda
    python tests/benchmark_training.py --device cpu
"""
import argparse
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.training import torch_finetune as training
from training_reference import reference_loss
import torch


def benchmark(loss_fn, points, device, repeats, warmup):
    model = training.TorchHandwritingModel(device=device)
    t = torch.arange(points + 1, dtype=torch.float32, device=device)
    offsets = torch.stack((0.3 + 0.1 * torch.cos(t / 7),
                           0.4 * torch.sin(t / 9), (t % 17 == 0).float()), dim=1)
    offsets[0] = 0
    text = torch.as_tensor(model.encode_text('Hello world'), device=device)
    optimizer = torch.optim.Adam(model.trainable_variables, lr=1e-5, eps=1e-7)

    def synchronize():
        if device.startswith('cuda'):
            torch.cuda.synchronize(device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(offsets, text, model.trainable_variables,
                       training.LSTM_SIZE, training.NUM_ATTN_MIX, training.NUM_OUT_MIX)
        if not torch.isfinite(loss).item():
            raise ValueError('Benchmark loss is not finite')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_variables, 5.0, error_if_nonfinite=True)
        optimizer.step()
        return loss.detach().item()

    first_step = None
    for _ in range(warmup):
        synchronize()
        started = time.perf_counter()
        step()
        synchronize()
        if first_step is None:
            first_step = time.perf_counter() - started
    synchronize()
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        step()
        synchronize()
        times.append(time.perf_counter() - started)
    peak = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if device.startswith('cuda') else None)
    return statistics.median(times), peak, first_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--points', type=int, nargs='+', default=[128, 512])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=5)
    args = parser.parse_args()
    if min(args.points + [args.repeats, args.warmup]) < 1:
        parser.error('Point counts, repeats, and warmup must be positive')
    # Match the application's small-matvec CPU thread policy.
    torch.set_num_threads(1)
    model = training.TorchHandwritingModel(device=args.device)
    device, device_name = model.device, model.device_name
    del model
    print(f'torch={torch.__version__} device={device} name={device_name}', flush=True)
    print(f'median of {args.repeats} full training steps after {args.warmup} warmups; synthetic samples', flush=True)
    for points in args.points:
        before, before_peak, before_first = benchmark(reference_loss, points, device, args.repeats, args.warmup)
        after, after_peak, after_first = benchmark(training._forward_loss, points, device, args.repeats, args.warmup)
        memory = (f' peak_MiB={before_peak:.1f}->{after_peak:.1f}' if before_peak is not None else '')
        print(f'points={points} before_s={before:.4f} after_s={after:.4f} speedup={before / after:.2f}x'
              f' first_step_s={before_first:.4f}->{after_first:.4f}{memory}', flush=True)


if __name__ == '__main__':
    main()
