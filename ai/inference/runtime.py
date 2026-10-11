"""Resolve generation devices without importing training backends."""
import re


def normalize_device(request):
    if not isinstance(request, str):
        raise ValueError('Generation device must be auto, cpu, cuda, or cuda:N.')
    request = request.strip().lower()
    if request not in ('auto', 'cpu', 'cuda') and not re.fullmatch(r'cuda:[0-9]+', request):
        raise ValueError('Generation device must be auto, cpu, cuda, or cuda:N.')
    return request


def cuda_count():
    try:
        import torch
    except ImportError:
        return 0
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def resolve_device(request='auto'):
    request = normalize_device(request)
    if request == 'cpu':
        return 'cpu'
    count = cuda_count()
    if request == 'auto':
        return 'cuda:0' if count else 'cpu'
    index = int(request.split(':')[1]) if ':' in request else 0
    if index >= count:
        raise ValueError(f'CUDA generation requested GPU {index}, but {count} CUDA GPUs are available. '
                         'Install CUDA-enabled PyTorch in the app environment and check NVIDIA drivers, '
                         'or choose Auto/CPU.')
    return f'cuda:{index}'


def create_model(weights_path, alphabet=None, device='auto'):
    device = resolve_device(device)
    if device.startswith('cuda'):
        from ai.inference.torch_model import TorchInferenceModel
        return TorchInferenceModel(weights_path, alphabet, device=device)
    from ai.inference.hw_model import HandwritingModel
    model = HandwritingModel(weights_path, alphabet)
    model.device, model.device_name = 'cpu', 'CPU'
    model.backend = 'numpy'
    return model
