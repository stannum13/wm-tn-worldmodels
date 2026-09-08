"""Small native reference for a fixed affine evidence program and offline FWHT."""

from __future__ import annotations

import ctypes
import hashlib
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np

_LIBRARY = None


class ProgramState(ctypes.Structure):
    _fields_ = [("detector_weights", ctypes.c_void_p), ("pairs", ctypes.c_void_p),
                ("pair_weights", ctypes.c_void_p), ("detector_count", ctypes.c_size_t),
                ("pair_count", ctypes.c_size_t), ("bias", ctypes.c_double),
                ("probability", ctypes.c_double), ("switch_probability", ctypes.c_double),
                ("threshold", ctypes.c_double)]


def library():
    global _LIBRARY
    if _LIBRARY is None:
        source = Path(__file__).parent / "csrc/surface_frontend.c"
        directory = Path(tempfile.mkdtemp(prefix="wmtn-native-surface-"))
        target = directory / ("frontend.dylib" if platform.system() == "Darwin" else "frontend.so")
        subprocess.run(["cc", "-O3", "-std=c11", "-shared", "-fPIC", str(source), "-o", str(target), "-lm"],
                       check=True, capture_output=True, text=True)
        _LIBRARY = ctypes.CDLL(str(target))
        _LIBRARY.surface_score.argtypes = [ctypes.POINTER(ProgramState), ctypes.c_void_p]
        _LIBRARY.surface_score.restype = ctypes.c_double
        _LIBRARY.surface_choose.argtypes = [ctypes.POINTER(ProgramState), ctypes.c_void_p]
        _LIBRARY.surface_choose.restype = ctypes.c_int
        _LIBRARY.surface_fwht.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        _LIBRARY.surface_fwht.restype = None
        _LIBRARY.source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        _LIBRARY.compiler = subprocess.check_output(["cc", "--version"], text=True).splitlines()[0]
    return _LIBRARY


class NativeProgram:
    def __init__(self, program, *, threshold=0.5):
        if not 0 < threshold < 1:
            raise ValueError("threshold must be in (0, 1)")
        self.detector_weights = np.ascontiguousarray(program.detector_weights, dtype=np.float64)
        self.pairs = np.ascontiguousarray(program.pairs, dtype=np.int32)
        self.pair_weights = np.ascontiguousarray(program.pair_weights, dtype=np.float64)
        if self.pairs.shape != (len(self.pair_weights), 2):
            raise ValueError("pair index and coefficient dimensions differ")
        if len(self.pairs) and (self.pairs.min() < 0 or self.pairs.max() >= len(self.detector_weights)):
            raise ValueError("pair index outside detector buffer")
        self.state = ProgramState(self.detector_weights.ctypes.data, self.pairs.ctypes.data,
                                  self.pair_weights.ctypes.data, len(self.detector_weights), len(self.pairs),
                                  program.bias, .5, .02, threshold)
        self.pointer = ctypes.pointer(self.state)
        self.lib = library()

    def reset(self):
        self.state.probability = .5

    def choose_address(self, address):
        """Caller owns a live contiguous uint8 detector buffer of the declared length."""
        return self.lib.surface_choose(self.pointer, address)

    def score_address(self, address):
        return self.lib.surface_score(self.pointer, address)


def fwht(values):
    if (values.ndim != 1 or values.dtype != np.float64 or not values.flags.c_contiguous
            or not len(values) or len(values) & (len(values) - 1)):
        raise ValueError("FWHT requires a contiguous float64 vector of power-of-two length")
    library().surface_fwht(values.ctypes.data, len(values))
    return values
