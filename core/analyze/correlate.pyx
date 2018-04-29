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

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
#def correlate(np.ndarray[DTYPE_t, ndim=1] timestamps, np.ndarray[DTYPE_t2, ndim=1] G, np.ndarray[DTYPE_t2, ndim=1] correlation_idxs, int maxCorrelationTick, int nbOfPhoton):
# def correlateLinearFull(np.ndarray[DTYPE_t, ndim=1] timestamps, np.ndarray[DTYPE_t2, ndim=1] G,  int maxCorrelationTick):
#
#
#     #G = np.zeros(maxCorrelationTick, dtype=np.uint)
#     #correlation_idxs = np.zeros(maxCorrelationTick, dtype=np.uint)
#     cdef int nbOfPhoton = np.size(timestamps)
#
#
#     cdef unsigned long long lastIdxMaxCorrelation = 0
#     cdef unsigned long long idx_maxCorrelationTime = 0
#     cdef int nbCorrelationPoint = maxCorrelationTick
#     cdef int n
#     cdef int j, i, idx
#
#     for n in range(nbOfPhoton):
#
#         maxCorrelationTime = timestamps[n] + nbCorrelationPoint
#
#         j = lastIdxMaxCorrelation
#         while(timestamps[j] < maxCorrelationTime):
#             if j == nbOfPhoton:
#                 j -= 1
#                 break
#             else:
#                 j += 1
#         #print(idx_maxCorrelationTime)
#         idx_maxCorrelationTime = j
#         lastIdxMaxCorrelation = idx_maxCorrelationTime
#
#         for i in range(n+1, idx_maxCorrelationTime):
#             idx = timestamps[i] - timestamps[n]
#             G[idx] += 1
#
#         # correlation_idxs = timestamps[n:idx_maxCorrelationTime] - timestamps[n]
#         # G[correlation_idxs] += 1
#     #return G

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def correlate(np.ndarray[DTYPE_t, ndim=1] timestamps, np.ndarray[DTYPE_t2, ndim=1] G, np.ndarray[DTYPE_t2, ndim=1] taus, int numLastPhoton):
    cdef int nbOfPhoton = np.size(timestamps)
    cdef int nbOfTau = np.size(taus)

    #tau holds the delay at which we want to compute the correlation
    cdef int n
    cdef int j, i, idx_tau

    for n in range(numLastPhoton):
        idx_tau = 0
        j = n + 1
        while(idx_tau < nbOfTau):
            while(timestamps[j] - timestamps[n]  < taus[idx_tau]):
                G[idx_tau] += 1
                j += 1
                # if j == nbOfPhoton:
                #     break
            idx_tau += 1

# def inverseCorrelation(np.ndarray[DTYPE_t, ndim=1] timestamps, np.ndarray[DTYPE_t2, ndim=1] Gn, DTYPE_t t_start, DTYPE_t t_end):
#     cdef int nbOfPhoton = np.size(timestamps)
#     cdef int n, j
#
#     j = 1
#     for n in range(nbOfPhoton):
#         j =
#
#
#         j = n + 1
#         while(timestamps[j] < taus[idx_tau]):
#                 G[idx_tau] += 1
#                 j += 1
#                 if j == nbOfPhoton:
#                     return
#
#             idx_tau += 1


# def correlate_log_lag(np.ndarray[np.int64_t, ndim=1] timestamps, np.ndarray[int, ndim=1] lags, int nbOfPhoton):
#     for n in range(nbOfPhoton):
#         j = n + 1
#         idx_lag = 0
#         for lag in lags:
#             maxCorrelationTime = timestamps[n] + lag
#             correlationValue = 0
#             j = idx_maxCorrelationTime
#             while(timestamps[j] < maxCorrelationTime):
#                 # On accumule tous les photons compris entre t[n] + lag[j-1] et t[n] + lag[j]
#                 correlationValue += 1
#                 if j == nbOfPhoton:
#                     break
#                 else:
#                     j += 1
#             idx_maxCorrelationTime = j
#
#             G[idx_lag] = correlationValue
#
#             idx_lag += 1





