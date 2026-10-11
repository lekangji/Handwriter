"""Compare cached CPU and CUDA document generation, excluding rendering/export."""
import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.inference.hw_model import WEIGHTS_PATH
from ai.inference.runtime import create_model, resolve_device
from document.calibration import PageTemplate
from document.layout import generate_document


def measure(model, text, repeats):
    times, points = [], []
    for _ in range(repeats + 1):
        started = time.perf_counter()
        doc = generate_document(text, model, PageTemplate(), seed=7, bias=1.5)
        times.append(time.perf_counter() - started)
        points.append(sum(len(stroke.points_mm) for page in doc.pages for line in page.lines for stroke in line.strokes))
    return times[0], statistics.median(times[1:]), points[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('Repeats must be positive.')
    device = resolve_device(args.device)
    text = 'Dear team,\nHello world.\nfarmingcattle-herding'
    cpu = create_model(WEIGHTS_PATH, device='cpu')
    candidate = create_model(WEIGHTS_PATH, device=device)
    cpu_first, cpu_median, cpu_points = measure(cpu, text, args.repeats)
    first, median, points = measure(candidate, text, args.repeats)
    print(f'CPU NumPy: first={cpu_first:.3f}s median={cpu_median:.3f}s points={cpu_points}')
    print(f'{device} {candidate.device_name}: first={first:.3f}s median={median:.3f}s points={points}')
    print(f'Elapsed speed ratio CPU/device={cpu_median / median:.2f}x (different backend samples may have different point counts)')


if __name__ == '__main__':
    main()
