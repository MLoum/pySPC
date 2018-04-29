import cython
cimport cython

import numpy as np
cimport numpy as np



# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.uint64
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.uint64_t DTYPE_t


DTYPE2 = np.int
ctypedef np.int_t DTYPE_t2

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False) # turn off negative index wrapping for entire function
@cython.nonecheck(False)
def bin(np.ndarray[DTYPE_t, ndim=1] data, np.ndarray[DTYPE_t2, ndim=1] binnedData, int binSizeInClick):
    data =  (data/binSizeInClick).astype(np.uint64)

    cdef int i
    for i in range(np.size(data)):
        binnedData[data[i]] += 1





