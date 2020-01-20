# Very simple allocation / free test
#
# Expected output is along the lines of:
#
# $ python simple01.py
#
# One allocation results from the creation of d_a in cuda.to_device.
# On free results from the deletion of d_a.

from cupy_mempool import use_cupy_mm_for_numba
import numpy as np

from numba import cuda

use_cupy_mm_for_numba()

a = np.zeros(10)
d_a = cuda.to_device(a)
del(d_a)
