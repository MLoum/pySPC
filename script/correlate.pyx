import cython
cimport cython

import numpy as np
cimport numpy as np

# DTYPE = np.float64
# ctypedef np.float64_t DTYPE_t

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.uint64
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.uint64_t DTYPE_t

DTYPE2 = np.int
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int_t DTYPE_t2

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def correlate(np.ndarray[DTYPE_t, ndim=1] timestamps, np.ndarray[DTYPE_t2, ndim=1] G):
    cdef int nbOfPhoton = np.size(timestamps)

    cdef int n, max_lag, max_lag_minus_1
    cdef int j

    max_lag = int(1/(25E-9))
    max_lag_minus_1 = max_lag - 1

    for n in range(nbOfPhoton):
        j = n + 1   # We don't take into account the 0 lag time.
        if j == nbOfPhoton:
            break

        while True:
            lag = timestamps[j] - timestamps[n]
            if lag > max_lag_minus_1:
                break
            else:
                G[lag] += 1
                j += 1
                if j == nbOfPhoton:
                    break
    return G




