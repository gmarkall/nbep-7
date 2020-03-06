from contextlib import contextmanager

from numba import cuda
from numba.cuda import HostOnlyCUDAMemoryManager, MemoryPointer, MemoryInfo

import ctypes
import cupy


class CuPyNumbaManager(HostOnlyCUDAMemoryManager):
    def __init__(self, *args, **kwargs):
        self._logging = kwargs.pop('logging', False)
        super().__init__(*args, **kwargs)
        self._allocations = {}
        self._mp = None

    def memalloc(self, nbytes, stream=0):
        if stream != 0:
            print("Warning: non-default stream has no effect")
        cp_mp = self._mp.malloc(nbytes)
        if self._logging:
            print("Allocated %d bytes at %x" % (nbytes, cp_mp.ptr))
        self._allocations[cp_mp.ptr] = cp_mp
        return MemoryPointer(
            cuda.current_context(),
            ctypes.c_uint64(int(cp_mp.ptr)),
            nbytes,
            finalizer=self._make_finalizer(cp_mp, nbytes)
        )

    def _make_finalizer(self, cp_mp, nbytes):
        allocations = self._allocations
        ptr = cp_mp.ptr
        logging = self._logging

        def finalizer():
            if logging:
                print("Freeing %d bytes at %x" % (nbytes, ptr))
            allocations.pop(ptr)

        return finalizer

    def get_ipc_handle(self, memory, stream=0):
        raise NotImplementedError

    def get_memory_info(self):
        return MemoryInfo(free=self._mp.free_bytes(),
                          total=self._mp.total_bytes())

    def initialize(self):
        super().initialize()
        self._mp = cupy.get_default_memory_pool()

    def reset(self):
        # Note: can't seem to force everything to be freed?
        if self._mp:
            self._mp.free_all_blocks()

    @contextmanager
    def defer_cleanup(self):
        with super().defer_cleanup():
            yield

    @property
    def interface_version(self):
        return 1


def use_cupy_mm_for_numba():
    cuda.cudadrv.driver.set_memory_manager(CuPyNumbaManager)


_numba_memory_manager = CuPyNumbaManager
