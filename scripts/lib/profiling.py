"""Compute-efficiency profiling helpers for Stage A.

Everything degrades gracefully: if a GPU, NVML, or a FLOP-counting library is
missing, the relevant numbers come back as None and the run still completes.
This is what feeds the efficiency story of the manuscript (compute per slide,
energy per slide, throughput, memory, model size).

Optional dependencies for richer numbers:
    pip install nvidia-ml-py     # GPU power/energy (import name: pynvml)
    pip install thop             # FLOP/MAC counting  (or fvcore / ptflops)
"""

import os
import time
import threading
import platform

import numpy as np


# ============================================================
# ENVIRONMENT / HARDWARE
# ============================================================

def get_env_info():
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available() else None
        )
        info["num_gpus"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_total_mem_mb"] = round(props.total_memory / 1e6, 1)
    except Exception as e:
        info["torch_error"] = str(e)
    return info


# ============================================================
# MODEL SIZE / FLOPS
# ============================================================

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "params_total": int(total),
        "params_trainable": int(trainable),
        "params_millions": round(total / 1e6, 4),
    }


def measure_flops(model, input_shape, device):
    """MACs / FLOPs for a single image. input_shape = (C, H, W).

    Tries thop, then fvcore, then ptflops. Returns Nones if none are installed.
    """
    import copy
    import torch

    model_copy = copy.deepcopy(model).eval().to(device)
    dummy = torch.randn(1, *input_shape, device=device)

    try:
        from thop import profile as thop_profile
        macs, _ = thop_profile(model_copy, inputs=(dummy,), verbose=False)
        tool = "thop"
    except Exception:
        try:
            from fvcore.nn import FlopCountAnalysis
            macs = FlopCountAnalysis(model_copy, dummy).total()
            tool = "fvcore"
        except Exception:
            try:
                from ptflops import get_model_complexity_info
                macs, _ = get_model_complexity_info(
                    model_copy, tuple(input_shape), as_strings=False,
                    print_per_layer_stat=False, verbose=False,
                )
                tool = "ptflops"
            except Exception:
                return {"gmacs_per_image": None, "gflops_per_image": None,
                        "flops_tool": None, "input_shape": list(input_shape)}

    return {
        "gmacs_per_image": round(macs / 1e9, 4),
        "gflops_per_image": round(2 * macs / 1e9, 4),  # 1 MAC = 2 FLOPs
        "flops_tool": tool,
        "input_shape": list(input_shape),
    }


# ============================================================
# GPU ENERGY METER (NVML, background sampling)
# ============================================================

class EnergyMeter:
    """Context manager. Samples GPU power in a thread and integrates to energy.

    Usage:
        with EnergyMeter() as m:
            ... work ...
        print(m.result)   # elapsed_sec, energy_joules, energy_wh, power stats

    Caveat for the paper: NVML power is whole-GPU. Report it only from runs
    where the GPU is not shared with other jobs (exclusive SLURM allocation).
    """

    def __init__(self, device_index=0, interval=0.1):
        self.interval = interval
        self.device_index = device_index
        self._nvml = None
        self._handle = None
        self._thread = None
        self._stop = threading.Event()
        self._watts = []
        self._times = []
        self.available = False
        self.result = {}
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self.available = True
        except Exception:
            self.available = False

    def _poll(self):
        while not self._stop.is_set():
            try:
                mw = self._nvml.nvmlDeviceGetPowerUsage(self._handle)
                self._watts.append(mw / 1000.0)
                self._times.append(time.perf_counter())
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._t0 = time.perf_counter()
        if self.available:
            self._watts, self._times = [], []
            self._stop.clear()
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self.elapsed_sec = time.perf_counter() - self._t0
        if self.available and self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2)
        self.result = self._summarize()
        return False

    def _summarize(self):
        r = {"elapsed_sec": round(self.elapsed_sec, 4),
             "energy_available": bool(self.available)}
        if self.available and len(self._watts) >= 2:
            t = np.asarray(self._times)
            w = np.asarray(self._watts)
            dt = np.diff(t)
            energy_j = float(np.sum((w[:-1] + w[1:]) / 2.0 * dt))  # trapezoid
            r["energy_joules"] = round(energy_j, 3)
            r["energy_wh"] = round(energy_j / 3600.0, 6)
            r["mean_power_w"] = round(float(w.mean()), 2)
            r["max_power_w"] = round(float(w.max()), 2)
            r["n_power_samples"] = int(len(w))
        return r
