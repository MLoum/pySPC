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

DTYPE3 = np.uint
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.uint_t DTYPE_t3

DTYPE4 = np.uint32
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.uint32_t DTYPE_t4

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
def correlate(np.ndarray[DTYPE_t, ndim=1] timestamps, np.ndarray[DTYPE_t2, ndim=1] G, np.ndarray[DTYPE_t2, ndim=1] taus, int numLastPhoton, int nb_of_pt_per_cascade):
    cdef int nbOfPhoton = np.size(timestamps)
    cdef int nbOfTau = np.size(taus)

    #tau holds the delay at which we want to compute the correlation
    cdef int n, tau
    cdef int j, i, idx_tau
    cdef int nb_of_cascade=10


    for tau in range(nb_of_cascade):
        timestamps /= 2
        binned_timestamps = np.bincount(timestamps)
        binned_timestamps = binned_timestamps[np.nonzero(binned_timestamps)]
        for n in range(nb_of_pt_per_cascade):
            binned_timestamps = np.diff(binned_timestamps)
            G[idx_tau + n] = np.size(np.where( binned_timestamps == n))

        idx_tau += nb_of_pt_per_cascade

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def whal_auto(np.ndarray[DTYPE_t, ndim=1] t_stamp_a, np.ndarray[DTYPE_t4, ndim=1] coeff_a, np.ndarray[DTYPE_t4, ndim=1] G,  np.ndarray[DTYPE_t4, ndim=1] lags, int B=10):
    """
    lags in tick

    :param t_stamps_a: make a copy BEFORE !
    :param lags:
    :return:
    """

    cdef int coarsening_counter, idx, idx_dl = 0
    cdef int idx_G = 0
    cdef int lag, corrected_lag = 0
    cdef int is_delay_turn = 0

    # for idx_G, lag in enumerate(lags):
    for idx_G in range(lags.size):
        lag = lags[idx_G]
        if coarsening_counter == B:
            #coaserning
            # NB : //= -> integer division
            t_stamp_a //= 2
            # find the position of consecutive idx with same timestamp (i.e difference = 0)
            consecutive_idxs = np.argwhere(np.diff(t_stamp_a) == 0)
            # Merge weighting (coeff) of same timestamps
            coeff_a[consecutive_idxs] += coeff_a[consecutive_idxs + 1]
            # Removing duplicate timestamps
            t_stamp_a = np.delete(t_stamp_a, consecutive_idxs + 1)
            coeff_a = np.delete(coeff_a, consecutive_idxs + 1)

            # idx_to_keep = np.nonzero(np.diff(t_stamp_a))
            # # idx_to_keep = np.invert(consecutive_idxs)
            # t_stamp_a = t_stamp_a[idx_to_keep]
            # coeff_a = coeff_a[idx_to_keep]

            coarsening_counter = 0
        else:
            coarsening_counter += 1

        # Pair calculation

        # Lag as also to be divided by 2 for each cascade
        corrected_lag = lag / np.power(2, idx_G//B)
        t_stamp_a_dl = t_stamp_a + corrected_lag

        # Short numpy implementation that is quite slow because it does'nt take into account the fact that the list are ordered.
        # correlation_match = np.in1d(t_stamp_a, t_stamp_a_dl, assume_unique=True)
        # G[idx_G] = np.sum(coeff_a[correlation_match])

        # Numba or Cython implementation
        idx, idx_dl = 0,0
        is_delay_turn = False
        nb_stamp_a_minus_1 = t_stamp_a.size - 1
        nb_stamp_b_minus_1 = t_stamp_a_dl.size - 1
        while(idx < nb_stamp_a_minus_1 and idx_dl < nb_stamp_b_minus_1):
            if is_delay_turn is False:
                while(t_stamp_a[idx] < t_stamp_a_dl[idx_dl] and idx < nb_stamp_a_minus_1):
                    idx += 1
                if t_stamp_a[idx] == t_stamp_a_dl[idx_dl]:
                    G[idx_G] += coeff_a[idx] * coeff_a[idx]
                    idx_dl += 1

                is_delay_turn = True

            else:
                while(t_stamp_a_dl[idx_dl] < t_stamp_a[idx] and idx_dl < nb_stamp_b_minus_1):
                    idx_dl += 1
                if t_stamp_a[idx] == t_stamp_a_dl[idx_dl]:
                    G[idx_G] += coeff_a[idx] * coeff_a[idx]
                    idx += 1

                is_delay_turn = False

    return G


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def find_pair_whal_auto(np.ndarray[DTYPE_t, ndim=1] t_stamp_a, np.ndarray[DTYPE_t4, ndim=1] coeff_a, np.ndarray[DTYPE_t, ndim=1] t_stamp_b, np.ndarray[DTYPE_t4, ndim=1] coeff_b):

    cdef int G = 0
    cdef int idx, idx_dl = 0
    cdef int nb_stamp_a_minus_1, nb_stamp_b_minus_



    while(idx < nb_stamp_a_minus_1 and idx_dl < nb_stamp_b_minus_1):
        if is_delay_turn is False:
            while(t_stamp_a[idx] < t_stamp_a_dl[idx_dl] and idx < nb_stamp_a_minus_1):
                idx += 1
            if t_stamp_a[idx] == t_stamp_a_dl[idx_dl]:
                G += coeff_a[idx] * coeff_a[idx]
                idx_dl += 1

            is_delay_turn = True

        else:
            while(t_stamp_a_dl[idx_dl] < t_stamp_a[idx] and idx_dl < nb_stamp_b_minus_1):
                idx_dl += 1
            if t_stamp_a[idx] == t_stamp_a_dl[idx_dl]:
                G += coeff_a[idx] * coeff_a[idx]
                idx += 1

            is_delay_turn = False

    return G



    # for n in range(numLastPhoton):
    #     if n%100000==0:
    #         print(n)
    #     idx_tau = 0
    #     j = n + 1
    #     while(idx_tau < nbOfTau):
    #         # First tau is not necesseraly 1
    #         while timestamps[j] - timestamps[n]  < taus[0] - 1:
    #             j += 1
    #
    #         while timestamps[j] - timestamps[n]  < taus[idx_tau]:
    #             G[idx_tau] += 1
    #             j += 1
    #             # if j == nbOfPhoton:
    #             #     break
    #         idx_tau += 1





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





