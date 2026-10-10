import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call

from ai.training import devices


class TrainingDeviceTests(unittest.TestCase):
    def runtime(self, count):
        physical = [SimpleNamespace(name=f"GPU:{i}") for i in range(count)]
        logical = [SimpleNamespace(name=f"/device:GPU:{i}") for i in range(count)]
        return SimpleNamespace(config=SimpleNamespace(
            list_physical_devices=Mock(return_value=physical),
            list_logical_devices=Mock(return_value=logical),
            experimental=SimpleNamespace(
                get_memory_growth=Mock(return_value=False),
                set_memory_growth=Mock(),
            ),
        ))

    def test_auto_prefers_gpu_and_falls_back_to_cpu(self):
        self.assertEqual(devices.configure_device("auto", self.runtime(2)), "/GPU:0")
        self.assertEqual(devices.configure_device("auto", self.runtime(0)), "/CPU:0")

    def test_explicit_cpu_and_external_gpu_index(self):
        self.assertEqual(devices.configure_device("cpu", self.runtime(2)), "/CPU:0")
        self.assertEqual(devices.configure_device("cuda:1", self.runtime(2)), "/GPU:1")
        self.assertEqual(devices.configure_device("cuda", self.runtime(1)), "/GPU:0")

    def test_explicit_cuda_never_silently_uses_cpu_or_wrong_gpu(self):
        for request, count in (("cuda", 0), ("cuda:1", 1)):
            with self.subTest(request=request), self.assertRaisesRegex(ValueError, "GPU"):
                devices.configure_device(request, self.runtime(count))

    def test_initialized_runtime_allows_subsequent_training(self):
        runtime = self.runtime(2)
        runtime.config.experimental.set_memory_growth.side_effect = RuntimeError("already initialized")
        self.assertEqual(devices.configure_device("cuda:1", runtime), "/GPU:1")

    def test_memory_growth_is_configured_before_first_cpu_job(self):
        runtime = self.runtime(2)
        self.assertEqual(devices.configure_device('cpu', runtime), '/CPU:0')
        gpus = runtime.config.list_physical_devices.return_value
        runtime.config.experimental.set_memory_growth.assert_has_calls([
            call(gpus[0], True), call(gpus[1], True),
        ])

    def test_invalid_device_is_rejected(self):
        for request in ("cuda:-1", "cuda:x", "gpu:0", "cpu:1", "", None):
            with self.subTest(request=request), self.assertRaises(ValueError):
                devices.normalize_device(request)


if __name__ == "__main__":
    unittest.main()
