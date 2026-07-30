"""Safe GPU provider selection for onnxruntime / InsightFace.

Background — the 30% crash
--------------------------
The pipeline transcribes with faster-whisper (ctranslate2) on the GPU FIRST.
That loads a cuDNN 9 DLL (``cudnn64_9.dll``) into the process. When InsightFace
later starts onnxruntime on the GPU, ORT calls ``cudnnGetLibConfig``. If the
cuDNN 9 DLL that is *already resident* is an older 9.x build that predates that
symbol (ctranslate2 ships one), ORT hits "Could not load symbol
cudnnGetLibConfig. Error code 127" — a hard C-level abort no try/except can
catch. Windows resolves ``cudnn64_9.dll`` by base name, so whichever copy loads
first wins for the whole process regardless of PATH order.

The safe rule
-------------
Only trust the GPU if BOTH hold:
  1. onnxruntime reports CUDAExecutionProvider is available, AND
  2. the cuDNN 9 DLL that will actually be used exports ``cudnnGetLibConfig``.

If (2) fails we return CPU providers — face detection still runs, just on CPU,
and the process never aborts. This gives real GPU acceleration on a correctly
configured machine and a graceful CPU fallback everywhere else, with no cuDNN
download required.

Set ``force_cpu=True`` to skip the probe entirely and always use CPU.
"""

from __future__ import annotations

import ctypes
import os
from typing import List, Tuple

from utils.logging import get_logger

logger = get_logger("onnx_gpu")

# Cache the probe result so we only pay for it once per process.
_CACHED: Tuple[bool, str] | None = None


def _cudnn9_has_symbol() -> Tuple[bool, str]:
    """Return (ok, detail). ok=True means a resident/loadable cuDNN 9 exports
    cudnnGetLibConfig — the symbol onnxruntime-gpu 1.20.x needs.

    In cuDNN 9 that symbol lives in the GRAPH sub-DLL (cudnn_graph64_9.dll), not
    the thin cudnn64_9.dll dispatcher. utils.cuda_path.preload_cudnn() should
    have already made a full cuDNN 9 resident; we check the graph DLL (with a
    fallback to the dispatcher for older monolithic layouts)."""
    for dll in ("cudnn_graph64_9.dll", "cudnn64_9.dll"):
        try:
            handle = ctypes.WinDLL(dll)
        except OSError:
            continue
        try:
            getattr(handle, "cudnnGetLibConfig")
            return True, f"{dll} exports cudnnGetLibConfig"
        except AttributeError:
            continue
    return False, "no resident cuDNN 9 exports cudnnGetLibConfig (stub-only 9.x)"


def gpu_is_safe() -> Tuple[bool, str]:
    """Probe whether onnxruntime can safely use CUDA in THIS process.

    Cached after the first call. Returns (ok, reason)."""
    global _CACHED
    if _CACHED is not None:
        return _CACHED

    # Expose torch's bundled CUDA runtime DLLs so ORT can find the CUDA provider.
    try:
        from utils.cuda_path import ensure_cuda_dll_path
        ensure_cuda_dll_path()
    except Exception:
        pass

    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            _CACHED = (False, "CUDAExecutionProvider not registered in onnxruntime")
            return _CACHED
    except Exception as e:
        _CACHED = (False, f"onnxruntime import failed ({e})")
        return _CACHED

    ok, detail = _cudnn9_has_symbol()
    _CACHED = (ok, detail)
    return _CACHED


def pick_providers(force_cpu: bool = False) -> Tuple[List[str], int, bool]:
    """Choose onnxruntime providers + ctx_id.

    Args:
        force_cpu: if True, always return CPU (skips the probe).

    Returns:
        (providers, ctx_id, cuda_used)
          providers : list for FaceAnalysis(providers=...)
          ctx_id    : 0 for GPU, -1 for CPU (InsightFace convention)
          cuda_used : True if the GPU path was selected
    """
    if force_cpu:
        logger.info("Face detection provider: CPU (force_cpu=True)")
        return ["CPUExecutionProvider"], -1, False

    ok, reason = gpu_is_safe()
    if ok:
        logger.info("Face detection provider: GPU (CUDA)", reason=reason)
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], 0, True

    logger.info("Face detection provider: CPU (GPU not safe)", reason=reason)
    return ["CPUExecutionProvider"], -1, False
