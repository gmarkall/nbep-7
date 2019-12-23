# Very simple allocation / free test
#
# Expected output is along the lines of:
#
# $ python simple01.py
# Event Type,Device ID,Address,Stream,Size (bytes),Free Memory,Total Memory,Current Allocs,Start,End,Elapsed,Location
# Alloc,0,0x7fae06600000,0,80,0,0,1,1.10549,1.1074,0.00191666,/home/nfs/gmarkall/numbadev/numba/numba/cuda/cudadrv/driver.py:683
# Free,0,0x7fae06600000,0,0,0,0,0,1.10798,1.10921,0.00122238,/home/nfs/gmarkall/numbadev/numba/numba/utils.py:678
#
# One allocation results from the creation of d_a in cuda.to_device.
# On free results from the deletion of d_a.

import rmm
import numpy as np

from numba import cuda

rmm.use_rmm_for_numba()

a = np.zeros(10)
d_a = cuda.to_device(a)
del(d_a)
print(rmm.csv_log())
