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


# Current model / implementation

- Numba Driver keeps list of allocations and deallocations
- Finalizers for Numba-allocated objects add the object to the list of
  deallocation.
- Allocation and deallocations lists shared between several primitives:
  - Device memory
  - Pinned memory
  - Mapped memory
  - Streams
  - Modules
  - ... ?
- Numba uses total memory size to determine how many pending deallocations it
  will keep around - a fraction of total GPU memory determined by
  `CUDA_DEALLOC_RATIO`.


# Potential requirements

- Allow Numba to continue managing host memory (mapped / pinned)
- Allow Numba to carry on managing streams / modules etc.
- Enable a different deallocation strategy to be used by plugins
  - Will need some test modifications - quite a few check the allocations /
    deallocations list, which will become tests of Numba's "bundled" memory
    manager.
- Ensure that Numba goes through the plugin for ALL allocations, and never
  directly to the driver.
- May need to fix some ambiguity around lifetimes for `__cuda_array_interface__`
  - see e.g.[ Numba Issue #4886](https://github.com/numba/numba/issues/4886).


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

FAILED (failures=7, errors=1, skipped=11)
```

Most failures are due to mismatches between test expectations for the state of
the deallocations list and its actual state - probably the expectation of the
tests needs updating for external memory management, since plugins needn't be
bound to manage memory using the same strategy as Numba internally.

One fail is:


```
======================================================================
FAIL: test_staged (numba.cuda.tests.cudapy.test_ipc.TestIpcStaged)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/nfs/gmarkall/numbadev/numba/numba/cuda/tests/cudapy/test_ipc.py", line 273, in test_staged
    self.fail(out)
AssertionError: Traceback (most recent call last):
  File "/home/nfs/gmarkall/numbadev/numba/numba/cuda/tests/cudapy/test_ipc.py", line 22, in core_ipc_handle_test
    arr = the_work()
  File "/home/nfs/gmarkall/numbadev/numba/numba/cuda/tests/cudapy/test_ipc.py", line 207, in the_work
    hostarray, deviceptr,  size=handle.size,
  File "/home/nfs/gmarkall/numbadev/numba/numba/cuda/cudadrv/driver.py", line 1554, in device_to_host
    fn(host_pointer(dst), device_pointer(src), size, *varargs)
  File "/home/nfs/gmarkall/numbadev/numba/numba/cuda/cudadrv/driver.py", line 1482, in device_pointer
    return device_ctypes_pointer(obj).value
  File "/home/nfs/gmarkall/numbadev/numba/numba/cuda/cudadrv/driver.py", line 1489, in device_ctypes_pointer
    require_device_memory(obj)
  File "/home/nfs/gmarkall/numbadev/numba/numba/cuda/cudadrv/driver.py", line 1507, in require_device_memory
    if not is_device_memory(obj):
  File "/home/nfs/gmarkall/numbadev/numba/numba/cuda/cudadrv/driver.py", line 1501, in is_device_memory
    return getattr(obj, '__cuda_memory__', False)
  File "/home/nfs/gmarkall/numbadev/numba/numba/cuda/cudadrv/memory.py", line 359, in __getattr__
    return getattr(self._view, fname)
ReferenceError: weakly-referenced object no longer exists
```

Interactions with IPC have not yet been considered, so this issue needs some
investigations.

The error is:

```
======================================================================
ERROR: test_consuming_strides (numba.cuda.tests.cudapy.test_cuda_array_interface.TestCudaArrayInterface)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/nfs/gmarkall/numbadev/numba/numba/cuda/tests/cudapy/test_cuda_array_interface.py", line 252, in test_consuming_strides
    got = cuda.from_cuda_array_interface(face).copy_to_host()
  File "/home/nfs/gmarkall/numbadev/numba/numba/cuda/cudadrv/devices.py", line 225, in _require_cuda_context
    return fn(*args, **kws)
  File "/home/nfs/gmarkall/numbadev/numba/numba/cuda/api.py", line 49, in from_cuda_array_interface
    devptr = driver.get_devptr_for_active_ctx(desc['data'][0])
  File "/home/nfs/gmarkall/numbadev/numba/numba/cuda/cudadrv/driver.py", line 1390, in get_devptr_for_active_ctx
    driver.cuPointerGetAttribute(byref(devptr), attr, ptr)
  File "/home/nfs/gmarkall/numbadev/numba/numba/cuda/cudadrv/driver.py", line 302, in safe_cuda_api_call
    self._check_error(fname, retcode)
  File "/home/nfs/gmarkall/numbadev/numba/numba/cuda/cudadrv/driver.py", line 337, in _check_error
    raise CudaAPIError(retcode, msg)
numba.cuda.cudadrv.error.CudaAPIError: [1] Call to cuPointerGetAttribute results in CUDA_ERROR_INVALID_VALUE
```

which will be resolved by [Numba PR
#5007](https://github.com/numba/numba/pull/5007).


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
should be anything other than an implementation detail (perhaps this can be
kept internal to Numba and not visible to the memory manager plugin interface).
At present with memory manager, they're exposed as:

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

### Ownership

Who should keep a list of pending deallocations?
- Numba has them for GPU arrays because it might eventually want to free them
- It also has them for streams, modules, and host allocations (and maybe other
  things not mentioned here), which it may need to keep (the memory manager
  plugin may not manage them).
- The deallocation policy of the external plugin may not match what Numba would
  do, so it will want some control over it.


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
