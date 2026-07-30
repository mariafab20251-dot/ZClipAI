"""Find CUDA DLLs at runtime when installed via another venv (e.g. dub_venv).

The ClippingTool uses a CPU-only torch (2.12.1+cpu) but needs the CUDA 12 runtime
DLLs that live in the sister project's (ChangeGUI) dub_venv.  This module searches
several known locations and adds the directory to the DLL search path so
faster-whisper, onnxruntime, and other CUDA-dependent libraries can load.

cuDNN 9 load-order fix (the "30% crash")
----------------------------------------
faster-whisper (ctranslate2) and onnxruntime-gpu both need cuDNN 9, but they
resolve ``cudnn64_9.dll`` by *base name* — whichever copy loads into the process
first wins for everyone.  ctranslate2/torch ship a stripped ~0.3MB copy that
lacks ``cudnnGetLibConfig``; if that one loads first, onnxruntime hard-aborts
("Could not load symbol cudnnGetLibConfig. Error code 127").  ``preload_cudnn()``
finds a FULL cuDNN 9 that exports the symbol and ``WinDLL``-loads it before
anything else, so the good copy is the resident one.  When no good copy exists
the function is a silent no-op and utils/onnx_gpu.py keeps face detection on CPU.
"""

import ctypes
import os
from pathlib import Path
from typing import List, Optional, Tuple

# Relative paths from this file to look for CUDA DLLs.
_CANDIDATE_ROOTS = [
    # The sister project's dub_venv (most common)
    "../../ChangeGUI/setup/dub_venv/Lib/site-packages/torch/lib",
    # ClippingTool's own venv (when torch+cu is installed directly)
    "../.venv/Lib/site-packages/torch/lib",
    # Global python installation
    "../../../AppData/Local/Programs/Python/Python311/Lib/site-packages/torch/lib",
]

# Marker DLL that must exist
_MARKER = "cublas64_12.dll"


def _find_cuda_torch_lib() -> Optional[Path]:
    """Search for a torch/lib directory that contains cublas64_12.dll."""
    this_file = Path(__file__).resolve().parent
    for rel in _CANDIDATE_ROOTS:
        candidate = (this_file / rel).resolve()
        if candidate.is_dir() and (candidate / _MARKER).exists():
            return candidate
    # Also search via os.environ known paths
    for env_var in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        val = os.environ.get(env_var, "")
        if val:
            candidate = Path(val) / "bin"
            if candidate.is_dir() and (candidate / _MARKER).exists():
                return candidate
    return None


def ensure_cuda_dll_path() -> bool:
    """Add CUDA DLL directory to the search path. Returns True if found."""
    lib = _find_cuda_torch_lib()
    if lib is not None:
        lib_str = str(lib)
        # Prepend to PATH (stronger than add_dll_directory for some libraries)
        cur_path = os.environ.get("PATH", "")
        if lib_str not in cur_path:
            os.environ["PATH"] = lib_str + os.pathsep + cur_path
        # Also use add_dll_directory as a secondary measure
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(lib_str)
            except Exception:
                pass
    # Make a FULL cuDNN 9 resident before anything loads the stripped copy.
    preload_cudnn()
    return lib is not None


# --- cuDNN 9 preload (the "C" fix: unlock GPU face detection) --------------

# Directories that may hold a full cuDNN 9 install (the split sub-DLL set).
_CUDNN_CANDIDATE_DIRS = [
    "../.venv/Lib/site-packages/nvidia/cudnn/bin",
    "../../ChangeGUI/setup/dub_venv/Lib/site-packages/nvidia/cudnn/bin",
]

# The symbol onnxruntime-gpu 1.20.x requires. In cuDNN 9 it lives in the graph
# sub-DLL, NOT the thin cudnn64_9.dll dispatcher.
_CUDNN_SYMBOL = "cudnnGetLibConfig"
_CUDNN_GRAPH_DLL = "cudnn_graph64_9.dll"

# Load order matters: dependencies (graph/ops/cnn/...) before the dispatcher.
_CUDNN_LOAD_ORDER = [
    "cudnn_graph64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn64_9.dll",
]

_PRELOAD_CACHE: Optional[Tuple[bool, str]] = None


def _dll_exports(path: str, symbol: str) -> bool:
    """True if the DLL at `path` loads and exports `symbol`."""
    try:
        handle = ctypes.WinDLL(path)
    except OSError:
        return False
    try:
        getattr(handle, symbol)
        return True
    except AttributeError:
        return False


def _find_full_cudnn9_dir() -> Optional[Path]:
    """Find a dir holding a FULL cuDNN 9 whose graph DLL exports the symbol."""
    this_file = Path(__file__).resolve().parent
    for rel in _CUDNN_CANDIDATE_DIRS:
        cand = (this_file / rel).resolve()
        graph = cand / _CUDNN_GRAPH_DLL
        if cand.is_dir() and graph.exists():
            # Make CUDA runtime deps discoverable, then verify the symbol.
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(str(cand))
                except Exception:
                    pass
            if _dll_exports(str(graph), _CUDNN_SYMBOL):
                return cand
    return None


def preload_cudnn() -> Tuple[bool, str]:
    """Load a full cuDNN 9 into the process BEFORE ctranslate2/onnxruntime.

    cuDNN 9 splits its API across sub-DLLs; ``cudnnGetLibConfig`` lives in
    ``cudnn_graph64_9.dll``. ctranslate2/torch ship only a ~0.3MB stub
    ``cudnn64_9.dll`` with no graph backing, so whichever loads first by base
    name wins for the whole process — and if the stub wins, onnxruntime aborts.
    Preloading the FULL set first makes the good copies resident, so both
    faster-whisper and onnxruntime bind to a cuDNN that has every symbol.

    Cached. Returns (ok, detail). ok=True means a symbol-complete cuDNN 9 is now
    resident; False means none was found (caller keeps face detection on CPU).
    """
    global _PRELOAD_CACHE
    if _PRELOAD_CACHE is not None:
        return _PRELOAD_CACHE

    cudnn_dir = _find_full_cudnn9_dir()
    if cudnn_dir is None:
        _PRELOAD_CACHE = (False, "no full cuDNN 9 (with graph DLL) found")
        return _PRELOAD_CACHE

    dir_str = str(cudnn_dir)
    cur_path = os.environ.get("PATH", "")
    if dir_str not in cur_path:
        os.environ["PATH"] = dir_str + os.pathsep + cur_path
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(dir_str)
        except Exception:
            pass

    loaded = 0
    for name in _CUDNN_LOAD_ORDER:
        dll = cudnn_dir / name
        if not dll.exists():
            continue
        try:
            ctypes.WinDLL(str(dll))
            loaded += 1
        except OSError:
            # A sub-DLL failing to load isn't fatal to the ones that did.
            pass

    _PRELOAD_CACHE = (True, f"preloaded full cuDNN 9 from {dir_str} ({loaded} DLLs)")
    return _PRELOAD_CACHE
