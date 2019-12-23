# nbep-7
Numba Enhancement Proposal (NBEP) 7: External Memory Management Plugins

# Document status

This is a *pre-draft* - many sections are incomplete / work-in-progress and all
subject to change significantly in the near future. A large proportion of this
document simply consists of notes that may need significant revision or moving
out entirely.


# Background

- CUDA Array interface allows sharing of data between different GPU-accelerated
  Python libraries.
- However, each manages its own memory:
  - Numba has internal memory management for creation of device arrays
  - RAPIDS (cuDF, etc.) use the Rapids Memory Manager (RMM).
  - CuPy includes a memory pool implementation for both device and pinned
    memory.

The goal of this NBEP is to enable Numba's internal memory management to be
replaced by the user with a plugin interface, so that Numba requests allocations
and frees from an external memory manager. I.e. Numba no longer directly
allocates and frees device memory when creating device arrays, but instead
requests allocations from the external manager.


# Interface

Based on outlining functions from driver module:

```python
class BaseCUDAMemoryManager(object):
    def memalloc(self, bytesize):
        raise NotImplementedError

    def memhostalloc(self, bytesize, mapped, portable, wc):
        raise NotImplementedError

    def mempin(self, owner, pointer, size, mapped):
        raise NotImplementedError

    def prepare_for_use(self, memory_info):
        raise NotImplementedError
```

Maybe want to have the option for the Numba memory management for host
allocations? (e.g. if RMM or other library does not support it?


# Prototyping / experimental implementation

See:

- Numba branch: https://github.com/gmarkall/numba/tree/grm-numba-nbep-7.
- RMM branch: https://github.com/gmarkall/rmm/tree/grm-numba-nbep-7.
- CuPy branch: To be addressed in future.
  - See [CuPy memory management docs](https://docs-cupy.chainer.org/en/stable/reference/memory.html).


## Current implementation status

A simple allocation and free using RMM appears to work. For the example code:

```python
import rmm
import numpy as np

from numba import cuda

rmm.use_rmm_for_numba()

a = np.zeros(10)
d_a = cuda.to_device(a)
del(d_a)
print(rmm.csv_log())
```

We see:

```
Event Type,Device ID,Address,Stream,Size (bytes),Free Memory,Total Memory,Current Allocs,Start,End,Elapsed,Location
Alloc,0,0x7fae06600000,0,80,0,0,1,1.10549,1.1074,0.00191666,/home/nfs/gmarkall/numbadev/numba/numba/cuda/cudadrv/driver.py:683
Free,0,0x7fae06600000,0,0,0,0,0,1.10798,1.10921,0.00122238,/home/nfs/gmarkall/numbadev/numba/numba/utils.py:678
```

### Numba CUDA Unit tests

A summary of the unit test results at present running with:

```
NUMBA_CUDA_MEMORY_MANAGER=RMM python -m numba.runtests numba.cuda.tests
```

is:

```
Ran 517 tests in 173.017s

FAILED (failures=11, errors=83, skipped=11)
```

Many errors are due to unprintable characters in the CSV (at present the log is
printed out for every allocation and free, to make it easy to see what's going
on), a lack of support for host/pinned memory allocation, and a small number of
actual errors.


# To think about / expand on

1. Interaction with context. Does the memory manager plugin need to know about
   about the context, or just use the current context?
   - Should each context have its own memory manager?
     - Does RMM have one manager per context?
2. How about resetting the context?
3. What about streams?
   - RMM accepts stream parameter
     - (How) does this map to `cudaMalloc`?
   - Does the Numba CUDA target ignore the stream for allocations anyway? (I
     believe so but need to check)


# Notes

Mainly about implementation details / changes.


## Devicearray

Device array creates memory pointer from driver. Should be abstracted into
method to get a pointer instead of constructing directly? e.g. at the moment:

```python
gpu_data = _memory.MemoryPointer(context=devices.get_context(),
                                 pointer=c_void_p(0), size=0)
```


## Allocations / deallocations in driver

Whilst the exposure of these is useful for testing, it's not clear that these
should be anything other than an implementation detail. At present with memory
manager, they're exposed as:

```python
    @property
    def allocations(self):
        return self._memory_manager.allocations

    @allocations.setter
    def allocations(self, value):
        self._memory_manager.allocations = value

    @allocations.deleter
    def allocations(self):
        del self._memory_manager.allocations

    @property
    def deallocations(self):
        return self._memory_manager.deallocations

    @deallocations.setter
    def deallocations(self, value):
        self._memory_manager.deallocations = value

    @deallocations.deleter
    def deallocations(self):
        del self._memory_manager.deallocations
```

There are some uses of them in the driver, e.g.:


```python
def _module_finalizer(context, handle):
    dealloc = context.deallocations
    modules = context.modules

    def core():
        shutting_down = utils.shutting_down  # early bind

        def module_unload(handle):
            # If we are not shutting down, we must be called due to
            # Context.reset() of Context.unload_module().  Both must have
            # cleared the module reference from the context.
            assert shutting_down() or handle.value not in modules
            driver.cuModuleUnload(handle)

        dealloc.add_item(module_unload, handle)

    return core
```

Here the deallocations list is used for unloading the module rather than
deallocating memory. Probably wants separating out!


## Relying on driver for memory info

We don't want to rely on the driver for info about memory, e.g. when
initialising the memory manager from the context:

```python
    def prepare_for_use(self):
        """Initialize the context for use.
        It's safe to be called multiple times.
        """
        self._memory_manager.prepare_for_use(self.get_memory_info().total)
```

Maybe the context could be passed in to the memory manager for its
initialisation if necessary.


## Testing

Will need some test refactoring, e.g.

```diff
--- a/numba/cuda/tests/cudadrv/test_cuda_memory.py
+++ b/numba/cuda/tests/cudadrv/test_cuda_memory.py
@@ -2,7 +2,7 @@ import ctypes

 import numpy as np

-from numba.cuda.cudadrv import driver, drvapi, devices
+from numba.cuda.cudadrv import driver, drvapi, devices, memory
 from numba.cuda.testing import unittest, CUDATestCase
 from numba.utils import IS_PY3
 from numba.cuda.testing import skip_on_cudasim
@@ -77,14 +77,14 @@ class TestCudaMemory(CUDATestCase):
             dtor_invoked[0] += 1

         # Ensure finalizer is called when pointer is deleted
-        ptr = driver.MemoryPointer(context=self.context, pointer=fake_ptr,
-                                   size=40, finalizer=dtor)
+        ptr = memory.MemoryPointer(context=self.context, pointer=fake_ptr,
+                                           size=40, finalizer=dtor)
         self.assertEqual(dtor_invoked[0], 0)
         del ptr
         self.assertEqual(dtor_invoked[0], 1)

         # Ensure removing derived pointer doesn't call finalizer
-        ptr = driver.MemoryPointer(context=self.context, pointer=fake_ptr,
+        ptr = memory.MemoryPointer(context=self.context, pointer=fake_ptr,
                                    size=40, finalizer=dtor)
         owned = ptr.own()
         del owned
```
