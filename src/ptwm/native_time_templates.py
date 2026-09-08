"""Native four-mode graph selector; graph storage and matching are separate."""
from __future__ import annotations

import ctypes
import hashlib
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np

_LIBRARY = None


class TimeProgramState(ctypes.Structure):
    _fields_ = [("detector_weights", ctypes.c_void_p), ("pair_weights", ctypes.c_void_p),
                ("pairs", ctypes.c_void_p), ("bias", ctypes.c_void_p), ("risks", ctypes.c_void_p),
                ("detector_count", ctypes.c_size_t), ("pair_count", ctypes.c_size_t),
                ("rule", ctypes.c_int32), ("joint", ctypes.c_double * 20),
                ("posterior", ctypes.c_double * 4), ("logits", ctypes.c_double * 4)]


def library():
    global _LIBRARY
    if _LIBRARY is None:
        source = Path(__file__).parent / "csrc/time_template_frontend.c"
        directory = Path(tempfile.mkdtemp(prefix="wmtn-native-time-"))
        target = directory / ("frontend.dylib" if platform.system() == "Darwin" else "frontend.so")
        subprocess.run(["cc", "-O3", "-std=c11", "-shared", "-fPIC", str(source), "-o", str(target), "-lm"],
                       check=True, capture_output=True, text=True)
        _LIBRARY = ctypes.CDLL(str(target))
        _LIBRARY.time_program_reset.argtypes = [ctypes.POINTER(TimeProgramState)]
        _LIBRARY.time_program_reset.restype = None
        _LIBRARY.time_program_choose.argtypes = [ctypes.POINTER(TimeProgramState), ctypes.c_void_p]
        _LIBRARY.time_program_choose.restype = ctypes.c_int
        _LIBRARY.source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        _LIBRARY.compiler = subprocess.check_output(["cc", "--version"], text=True).splitlines()[0]
    return _LIBRARY


class NativeTimeProgram:
    def __init__(self, feature_map, model, tables, *, rule):
        if rule not in ("mode", "global", "binned"):
            raise ValueError("unknown action rule")
        expected = len(feature_map.detector_times) + len(feature_map.pairs) + 4 + int(feature_map.detector_times.max())
        if (np.shape(model["weights"]) != (expected, 4) or np.shape(model["bias"]) != (4,)
                or np.shape(model["mean"]) != (expected,) or np.shape(model["scale"]) != (expected,)):
            raise ValueError("expected four aligned affine mode models")
        if any(not np.all(np.isfinite(model[k])) for k in ("weights", "bias", "mean", "scale")) or np.any(np.asarray(model["scale"]) <= 0):
            raise ValueError("invalid affine evidence parameters")
        programs = [feature_map.compile({"weights": np.asarray(model["weights"])[:, m], "bias": model["bias"][m],
                                         "mean": np.asarray(model["mean"]), "scale": np.asarray(model["scale"])}) for m in range(4)]
        self.detector_weights = np.ascontiguousarray(np.column_stack([p.detector_weights for p in programs]), dtype=np.float64)
        self.pair_weights = np.ascontiguousarray(np.column_stack([p.pair_weights for p in programs]), dtype=np.float64)
        self.pairs = np.ascontiguousarray(feature_map.pairs, dtype=np.int32)
        if (self.pairs.shape != (len(self.pair_weights), 2) or
                (len(self.pairs) and (self.pairs.min() < 0 or self.pairs.max() >= len(feature_map.detector_times)))):
            raise ValueError("invalid pair index buffer")
        self.bias = np.ascontiguousarray([p.bias for p in programs], dtype=np.float64)
        risk = np.asarray(tables["binned"]) if rule == "binned" else np.repeat(np.asarray(tables["global"])[:, :, None], 4, axis=2)
        if risk.shape != (4, 4, 4) or not np.all(np.isfinite(risk)) or np.any((risk < 0) | (risk > 1)):
            raise ValueError("invalid risk table")
        self.risks = np.ascontiguousarray(risk, dtype=np.float64)
        self.state = TimeProgramState(self.detector_weights.ctypes.data, self.pair_weights.ctypes.data,
            self.pairs.ctypes.data, self.bias.ctypes.data, self.risks.ctypes.data,
            len(feature_map.detector_times), len(feature_map.pairs), ("mode", "global", "binned").index(rule))
        self.pointer, self.lib = ctypes.pointer(self.state), library()
        self.reset()

    def reset(self):
        self.lib.time_program_reset(self.pointer)

    def choose_address(self, address):
        """Caller owns a live contiguous byte buffer of binary detector values."""
        return self.lib.time_program_choose(self.pointer, address)

    @property
    def persistent_bytes(self):
        return 20 * 8

    @property
    def constant_bytes(self):
        return sum(x.nbytes for x in (self.detector_weights, self.pair_weights, self.pairs, self.bias, self.risks))
