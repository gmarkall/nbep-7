# nbep-7
Numba Enhancement Proposal (NBEP) 7: External Memory Management Plugins

# Document status

This is a *pre-draft* - many sections are incomplete / work-in-progress and all
subject to change significantly in the near future. A large proportion of this
document simply consists of notes that may need significant revision or moving
out entirely.


# Background and goals

The [CUDA Array
interface](https://numba.pydata.org/numba-doc/latest/cuda/cuda_array_interface.html)
enables sharing of data between different Python libraries that provide access
to the GPU. However, each library manages its own memory distinctly from the
others. For example:

- [Numba](https://numba.pydata.org/) internally manages memory for the creation
  of device and mapped host arrays.
- [The RAPIDS libraries](https://rapids.ai/) (cuDF, cuML, etc.) use the [Rapids
  Memory Manager (RMM)](https://github.com/rapidsai/rmm) for allocating device
  memory.
- [CuPy](https://cupy.chainer.org/) includes a [memory pool
  implementation](https://docs-cupy.chainer.org/en/stable/reference/memory.html)
  for both device and pinned memory.

The goal of this NBEP is to enable Numba's internal memory management to be
replaced by the user with a plugin interface, so that Numba requests all
allocations from an external memory manager. When the plugin interface is in
use, Numba no longer directly allocates or frees memory when creating arrays,
but instead requests allocations and frees through the external manager.


# Requirements

Provide an *External Memory Manager (EMM)* interface in Numba.

- When the EMM is in use, Numba will make all memory allocation using the EMM.
  It will never directly call functions such as `CuMemAlloc`, `cuMemFree`, etc.
- When not using an *External Memory Manager (EMM)*, Numba's present behaviour
  is unchanged (at the time of writing, the current version is the 0.47
  release).

If an EMM is to be used, it will entirely replace Numba's internal memory
management for the entire execution - an interface for setting the memory
manager will be provided.

## Device v.s. Host memory

An EMM will always take responsibility for the management of device memory.
However, not all CUDA memory management libraries also support managing host
memory, so an option should be provided for Numba to continue the management of
host memory, whilst ceding control of device memory to the EMM.

## Deallocation strategies

Numba's internal memory management uses a [deallocation
strategy](https://numba.pydata.org/numba-doc/latest/cuda/memory.html#deallocation-behavior) designed to
increase efficiency by deferring deallocations until a significant quantity are
pending. It also provides a mechanism for preventing deallocations entirely for
a critical section, using the
[`defer_cleanup`](https://numba.pydata.org/numba-doc/latest/cuda/memory.html#numba.cuda.defer_cleanup)
context manager.

- When the EMM is not in use, the deallocation strategy and operation of
  `defer_cleanup` remain the same.
- When the EMM is in use, the deallocation strategy is implemented by the EMM,
  and Numba's internal deallocation mechanism is not used. For example:
  - A similar strategy could be implemented by the EMM, or
  - Deallocated memory might immediately be returned to a memory pool.
- The `defer_cleanup` context manager may behave differently with an EMM - an
  EMM should be accompanied by documentation of the behaviour of the
  `defer_cleanup` context manager when it is in use.
  - For example, a pool allocator may always immediately return memory to a
    pool immediately even when the context manager is in use, but may choose not
    to free empty pools until `defer_cleanup` is not in use.


## Management of other objects

In addition to memory, Numba manages the allocation and deallocation of
[streams](http://numba.pydata.org/numba-doc/latest/cuda-reference/host.html?highlight=stream#numba.cuda.stream)
and modules (a module is a compiled object). The management of streams and
modules should be unchanged by the presence or absence of an EMM.

## Non-requirements

In order to minimise complexity for an initial implementation, the following
will not be supported:

- Using different memory managers for different contexts. All contexts will use
  the same memory manager.
- Changing the memory manager once execution has begun. It is not practical to
  change the memory manager and retain all allocations. Cleaning up the entire
  state and then changing to a different memory allocator (rather than starting
  a new process) appears to be a rather niche use case.
- Any changes to the `__cuda_array_interface__` to further define its semantics,
  e.g. for acquiring / releasing memory as discussed in [Numba Issue
  #4886](https://github.com/numba/numba/issues/4886) - these are independent,
  and can be addressed as part of separate proposals.


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



# Interface for Plugin developers

A new module, `numba.cuda.cudadrv.memory` will be added. The relevant globals of
this module to external memory management are:

- `BaseCUDAMemoryManager` and `HostOnlyCUDAMemoryManager`: base classes for
  external memory management plugins.
- `MemoryPointer`: used to encapsulate information about a pointer to device- or
  host-memory.
- `set_memory_manager`: a method for registering an external memory manager with
  Numba.


## Plugin Base Classes

An external memory management plugin is implemented by inheriting from the
`BaseCUDAMemoryManager` class, and registering the memory manager with Numba
prior to the execution of any CUDA operations. The `BaseCUDAMemoryManager` class
is defined as:

```python
class BaseCUDAMemoryManager(object):
    def memalloc(self, nbytes):
        """
        Allocate `nbytes` of on-device memory in the current context.
        Returns a MemoryPointer to the allocated memory.
        """
        raise NotImplementedError

    def memhostalloc(self, bytesize, mapped, portable, wc):
        """
        Allocate `nbytes` of pinned host memory.
        """
        raise NotImplementedError

    def mempin(self, owner, pointer, size, mapped):
        """
        Pin memory that is already allocated
        FIXME: Better description.
        """
        raise NotImplementedError

    def prepare_for_use(self, memory_info):
        raise NotImplementedError
```

The thre`prepare_for_use` method is called by Numba prior to any memory
allocations being requested. This gives the EMM an opportunity to initialize any
data structures etc. that it needs for its normal operations. The method may be
called multiple times during the lifetime of the program - subsequent calls
should not invalidate or reset the state of the EMM.


## Representing pointers

The `MemoryPointer` class is used to repre

```
class MemoryPointer:
    def __init__(self, context, pointer, size, finalizer=None, owner=None):
```

- `context`: The context in which the pointer was allocated.
- `pointer`: A `ctypes` pointer type (e.g. `ctypes.c_uint64`) holding the
  address of the memory.
- `size`: The size of the allocation in bytes.
- `finalizer`: A method that is called when the last reference to the
  `MemoryPointer` object is released. Usually this will make a call to the
  external memory management library to inform it that the memory is no longer
  required, and that it could potentially be freed (though the EMM is not
  required to free it immediately).
- `owner`: (FIXME: find a way to describe this and what it is used for).



Some external memory managers will support management of on-device memory only.
In order to implement an external memory manager using these easily, Numba will
provide a memory manager class with implementations of the `memhostalloc` and
`mempin` methods. Im


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

All relevant unit tests pass with the prototype branch. A summary of the unit
test results at present running with:

```
NUMBA_CUDA_MEMORY_MANAGER=RMM python -m numba.runtests numba.cuda.tests
```

is:

```
Ran 517 tests in 146.964s

OK (skipped=10)
```

When running with the built-in Numba memory management, the output is:

```
Ran 517 tests in 137.271s

OK (skipped=4)
```

i.e. the changes for using an external memory manager do not break the built-in
Numba memory management.


# To think about / expand on

1. Interaction with context. Does the memory manager plugin need to know about
   about the context, or just use the current context?
   - Should each context have its own memory manager?
     - Does RMM have one manager per context?
2. How about resetting the context?
   - Need to notify the memory manager plugin that the context was reset so it
     can invalidate its own handles, maybe?
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

Staged IPC should not own the memory it allocates:

```
diff --git a/numba/cuda/cudadrv/driver.py b/numba/cuda/cudadrv/driver.py
index 7832955..f2c1352 100644
--- a/numba/cuda/cudadrv/driver.py
+++ b/numba/cuda/cudadrv/driver.py
@@ -922,7 +922,11 @@ class _StagedIpcImpl(object):
         with cuda.gpus[srcdev.id]:
             impl.close()

-        return newmem.own()
+        # This used to be newmem.own() but the own() was removed - when the
+        # Numba CUDA memory manager is used, the pointer is already owned -
+        # when another memory manager is used, it is incorrect to take
+        # ownership of the pointer.
+        return newmem
```

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

## Defer cleanup

Deferring cleanup needs to be taken into account (See S3.3.7 -
http://numba.pydata.org/numba-doc/latest/cuda/memory.html)


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

- Enabling a different deallocation strategy to be used by plugins:
  - Will need some test modifications - quite a few check the allocations /
    deallocations list, which will become tests of Numba's "bundled" memory
    manager.

# Questions / Discussion

## Stream parameter to alloc / free

Does the interface need to provide a stream to use for the allocation /
deallocation? Some memory managers accept a stream parameter (e.g. RMM and
underneath it CNMeM, etc).

## IPC handles

In RMM:

```python
def get_ipc_handle(ary, stream=0):
    """
    Get an IPC handle from the DeviceArray ary with offset modified by
    the RMM memory pool.
    """
    ipch = cuda.devices.get_context().get_ipc_handle(ary.gpu_data)
    ptr = ary.device_ctypes_pointer.value
    offset = librmm.rmm_getallocationoffset(ptr, stream)
    # replace offset with RMM's offset
    ipch.offset = offset
    desc = dict(shape=ary.shape, strides=ary.strides, dtype=ary.dtype)
    return cuda.cudadrv.devicearray.IpcArrayHandle(
        ipc_handle=ipch, array_desc=desc
    )
```

Figure out the consequences of this.
