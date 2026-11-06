"""Sample device-wide NVML memory independently of either framework allocator."""
from threading import Event, Thread


class GpuMemory:
    interval_seconds = 0.02

    def __enter__(self):
        import pynvml
        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.start = self.used()
        self.peak = self.start
        self.stop = Event()
        self.thread = Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def used(self):
        return self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used

    def sample(self):
        while not self.stop.wait(self.interval_seconds):
            self.peak = max(self.peak, self.used())

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.used())
        self.nvml.nvmlShutdown()

    def results(self):
        return {
            "memory_scope": "whole GPU including display and non-PyTorch allocations",
            "memory_sample_interval_ms": self.interval_seconds * 1000,
            "memory_at_measurement_start_bytes": self.start,
            "sampled_peak_device_used_bytes": self.peak,
        }
