"""Device selection shared by web and command-line training."""
import re


def normalize_device(request):
    if not isinstance(request, str):
        raise ValueError("Training device must be auto, cpu, cuda, or cuda:N (GPU index).")
    request = request.strip().lower()
    if request not in ("auto", "cpu", "cuda") and not re.fullmatch(r"cuda:[0-9]+", request):
        raise ValueError("Training device must be auto, cpu, cuda, or cuda:N (GPU index).")
    return request


def configure_device(request, tf):
    """Resolve a TensorFlow device before allocating model weights.

    Explicit CUDA requests fail when unavailable. GPU indices follow TensorFlow's
    visible devices, including CUDA_VISIBLE_DEVICES. CPU selection is independent
    of GPU availability. Memory growth avoids reserving all available GPU RAM.
    """
    request = normalize_device(request)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            if not tf.config.experimental.get_memory_growth(gpu):
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # A previous job may already have initialized the shared runtime.
            pass
    if request == "cpu":
        return "/CPU:0"
    gpus = tf.config.list_logical_devices("GPU")
    if request == "auto" and not gpus:
        return "/CPU:0"
    index = int(request.split(":")[1]) if ":" in request else 0
    if index >= len(gpus):
        raise ValueError(
            f"Requested GPU {index}, but TensorFlow exposes {len(gpus)} GPU(s). "
            "Install tensorflow[and-cuda] on Linux/WSL2 and check NVIDIA drivers "
            "and CUDA_VISIBLE_DEVICES. Modern TensorFlow does not support CUDA "
            "on native Windows. Use auto or cpu for CPU training."
        )
    return f"/GPU:{index}"
