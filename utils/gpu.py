import torch
import os
from typing import Optional, Dict, Any
from .logging import get_logger

logger = get_logger("gpu")


class GPUManager:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self._device = None
            self._device_name = None
            self._memory_info = {}
            self._cuda_available = False
            self._detect_gpu()

    def _detect_gpu(self):
        self._cuda_available = torch.cuda.is_available()

        if self._cuda_available:
            device_count = torch.cuda.device_count()
            logger.info("CUDA available", device_count=device_count)

            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                memory_total = props.total_memory / (1024 ** 3)
                logger.info(
                    "GPU detected",
                    device_id=i,
                    name=props.name,
                    memory_gb=round(memory_total, 2),
                    compute_capability=f"{props.major}.{props.minor}"
                )

            self._device = torch.device("cuda:0")
            self._device_name = torch.cuda.get_device_name(0)
            self._update_memory_info()
        else:
            self._device = torch.device("cpu")
            self._device_name = "CPU"
            logger.warning("CUDA not available, using CPU")

    def _update_memory_info(self):
        if self._cuda_available:
            self._memory_info = {
                "allocated_gb": round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2),
                "reserved_gb": round(torch.cuda.memory_reserved(0) / (1024 ** 3), 2),
                "max_allocated_gb": round(torch.cuda.max_memory_allocated(0) / (1024 ** 3), 2),
            }

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def is_cuda_available(self) -> bool:
        return self._cuda_available

    @property
    def memory_info(self) -> Dict[str, float]:
        self._update_memory_info()
        return self._memory_info

    def set_device(self, device_id: int):
        if not self._cuda_available:
            logger.warning("Cannot set GPU device, CUDA not available")
            return

        if device_id >= torch.cuda.device_count():
            raise ValueError(f"Device {device_id} not available. Only {torch.cuda.device_count()} devices found.")

        self._device = torch.device(f"cuda:{device_id}")
        self._device_name = torch.cuda.get_device_name(device_id)
        torch.cuda.set_device(device_id)
        logger.info("GPU device changed", device_id=device_id, name=self._device_name)

    def empty_cache(self):
        if self._cuda_available:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.debug("GPU cache cleared")

    def get_memory_fraction(self, fraction: float = 0.8):
        if not self._cuda_available:
            return

        total_memory = torch.cuda.get_device_properties(0).total_memory
        limit = int(total_memory * fraction)
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        logger.info("GPU memory limit set", fraction=fraction, limit_gb=round(limit / (1024 ** 3), 2))

    def get_optimal_batch_size(self, model_memory_mb: float, safety_factor: float = 0.8) -> int:
        if not self._cuda_available:
            return 1

        free_memory = (
            torch.cuda.get_device_properties(0).total_memory -
            torch.cuda.memory_reserved(0)
        ) / (1024 ** 2)

        available = free_memory * safety_factor
        batch_size = max(1, int(available / model_memory_mb))
        return min(batch_size, 64)

    def to_device(self, obj: Any, non_blocking: bool = True) -> Any:
        if hasattr(obj, "to"):
            return obj.to(self._device, non_blocking=non_blocking)
        return obj

    def synchronize(self):
        if self._cuda_available:
            torch.cuda.synchronize()

    def get_device_properties(self) -> Optional[Dict[str, Any]]:
        if not self._cuda_available:
            return None

        props = torch.cuda.get_device_properties(0)
        return {
            "name": props.name,
            "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
            "multi_processor_count": props.multi_processor_count,
            "compute_capability": f"{props.major}.{props.minor}",
            "current_memory": self.memory_info
        }


def get_device() -> torch.device:
    return GPUManager().device


def get_device_name() -> str:
    return GPUManager().device_name


def is_cuda_available() -> bool:
    return GPUManager().is_cuda_available


def empty_cache():
    GPUManager().empty_cache()


def set_memory_fraction(fraction: float = 0.8):
    GPUManager().get_memory_fraction(fraction)


def auto_configure_for_model(model_name: str, estimated_memory_mb: float = 2000) -> Dict[str, Any]:
    gpu = GPUManager()
    config = {
        "device": "cpu",
        "compute_type": "int8",
        "batch_size": 1
    }

    if gpu.cuda_available:
        config["device"] = "cuda"
        config["compute_type"] = "float16"
        config["batch_size"] = gpu.get_optimal_batch_size(estimated_memory_mb)
        logger.info("Auto-configured for GPU", model=model_name, **config)
    else:
        config["device"] = "cpu"
        config["compute_type"] = "int8"
        config["batch_size"] = 1
        logger.info("Auto-configured for CPU", model=model_name, **config)

    return config