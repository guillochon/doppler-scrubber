"""One process pool for the Python-bound work (ridge fits, line removal), with arrays passed
through shared memory instead of being pickled.

Windows (and macOS) start workers with "spawn", which re-imports the main module: a script
that calls the scrubber at top level must do so under `if __name__ == "__main__":`. If the
pool cannot start, callers fall back to doing the work in this process.
"""
from __future__ import annotations

import atexit
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from multiprocessing import shared_memory

import numpy as np

_STATE: dict = {}
_BLAS_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def _noop(i):
    return i


def get(workers: int) -> ProcessPoolExecutor | None:
    """The shared pool with `workers` processes, or None if pools do not work here."""
    if workers <= 1 or _STATE.get("broken") or mp.parent_process() is not None:
        return None
    if _STATE.get("n") != workers:
        if "pool" in _STATE:
            _STATE.pop("pool").shutdown()
        # one BLAS thread per worker: the pool is the parallelism, and n workers each starting
        # a full-width BLAS pool oversubscribe the cores (the variables are read at the
        # workers' numpy import, so they are set only while the pool starts them)
        saved = {k: os.environ.get(k) for k in _BLAS_VARS}
        os.environ.update({k: "1" for k in _BLAS_VARS})
        try:
            pool = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"))
            # workers start lazily: start them all now, while the variables are set
            list(pool.map(_noop, range(workers)))
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        _STATE.update(n=workers, pool=pool)
        if not _STATE.get("atexit"):
            atexit.register(lambda: _STATE["pool"].shutdown() if "pool" in _STATE else None)
            _STATE["atexit"] = True
    return _STATE["pool"]


def broken() -> None:
    """The pool failed (e.g. a script without a main guard): stop using pools."""
    if "pool" in _STATE:
        _STATE.pop("pool").shutdown(wait=False, cancel_futures=True)
    _STATE.update(broken=True, n=None)


class Shared:
    """A float array in shared memory, created by the parent: `.name` is what workers need."""

    def __init__(self, shape, dtype=np.float64):
        dtype = np.dtype(dtype)
        self.shape, self.dtype = tuple(shape), dtype.str
        self.shm = shared_memory.SharedMemory(create=True, size=max(1, int(np.prod(shape)) * dtype.itemsize))
        self.name = self.shm.name
        self.array = np.ndarray(self.shape, dtype, buffer=self.shm.buf)

    def close(self):
        self.array = None
        self.shm.close()
        self.shm.unlink()


@contextmanager
def attach(name: str, shape, dtype):
    """In a worker: the parent's shared array `name`, for the duration of a job."""
    shm = shared_memory.SharedMemory(name=name)
    try:
        yield np.ndarray(tuple(shape), np.dtype(dtype), buffer=shm.buf)
    finally:
        shm.close()
