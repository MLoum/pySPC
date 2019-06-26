"""
Functions to compute linear correlation on discrete signals (uniformly
sampled in time) **or** on point-processes (e.g. timestamps of events).
"""

import numpy as np
import numba

@numba.jit(nopython=True)
def find_pair_Whal(t_stamp_a, coeff_a, t_stamp_b, coeff_b, lag):
    G = 0
    idx, idx_dl = 0, 0
    is_delay_turn = False
    t_stamp_b_dl = t_stamp_b + lag
    nb_stamp_a_minus_1 = t_stamp_a.size - 1
    nb_stamp_b_minus_1 = t_stamp_b.size - 1

    while (idx < nb_stamp_a_minus_1 and idx_dl < nb_stamp_b_minus_1):
        if is_delay_turn is False:
            while (t_stamp_a[idx] < t_stamp_b_dl[idx_dl] and idx < nb_stamp_a_minus_1):
                idx += 1
            if t_stamp_a[idx] == t_stamp_b_dl[idx_dl]:
                G += coeff_a[idx] * coeff_b[idx_dl]
                idx_dl += 1

            is_delay_turn = True

        else:
            while (t_stamp_b_dl[idx_dl] < t_stamp_a[idx] and idx_dl < nb_stamp_b_minus_1):
                idx_dl += 1
            if t_stamp_a[idx] == t_stamp_b_dl[idx_dl]:
                G += coeff_a[idx] * coeff_b[idx_dl]
                idx += 1

            is_delay_turn = False
    return G


# @numba.jit(nopython=True)
def whal_auto(t_stamp_a, coeff_a, lags, B=10):
    """
    lags in tick
    :param t_stamps_a:
    :param lags:
    :return:
    """
    G = np.zeros(lags.size)
    t_stamp_a = np.copy(t_stamp_a)
    coeff_a = np.copy(coeff_a)

    coarsening_counter = 0
    for idx_G, lag in enumerate(lags):
        if coarsening_counter == B:
            #coaserning
            # NB : //= -> integer division
            # for idx in range(t_stamp_a.size):
            #     t_stamp_a[idx] = t_stamp_a[idx]//2
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


        # Short numpy implementation that is quite slow bevause it does'nt take into account the fact that the list are ordered.
        correlation_match = np.in1d(t_stamp_a, t_stamp_a_dl, assume_unique=True)
        G[idx_G] = np.sum(coeff_a[correlation_match])

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

@numba.jit(nopython=True)
def whal(t_stamp_a, coeff_a, t_stamp_b, coeff_b, lags, B=10):
    """
    lags in tick
    :param t_stamps_a:
    :param lags:
    :return:
    """
    G = np.zeros(lags.size)
    t_stamp_a = np.copy(t_stamp_a)
    t_stamp_b = np.copy(t_stamp_b)

    coarsening_counter = 0
    for lag in lags:
        if coarsening_counter%B == 0:
            #coaserning
            t_stamp_a /= 2
            # find the position of consecutive idx with same timestamp (i.e difference = 0)
            consecutive_idxs = np.argwhere(np.diff(t_stamp_a) == 0)
            # Merge weighting (coeff) of same timestamps
            coeff_a[consecutive_idxs] += coeff_a[consecutive_idxs + 1]
            # Removing duplicate timestamps
            idx_to_keep = np.nonzero(np.diff(t_stamp_a))
            t_stamp_a = t_stamp_a[idx_to_keep]
            coeff_a = coeff_a[idx_to_keep]

            t_stamp_b /= 2
            # find the position of consecutive idx with same timestamp (i.e difference = 0)
            consecutive_idxs = np.argwhere(np.diff(t_stamp_b) == 0)
            # Merge weighting (coeff) of same timestamps
            coeff_b[consecutive_idxs] += coeff_b[consecutive_idxs + 1]
            # Removing duplicate timestamps
            idx_to_keep = np.nonzero(np.diff(t_stamp_b))
            t_stamp_b = t_stamp_b[idx_to_keep]
            coeff_b = coeff_b[idx_to_keep]

            coarsening_counter = 0
        else:
            coarsening_counter += 1

        # Pair calculation
        correlation_match = np.in1d(t_stamp_a, t_stamp_b + lag)
        G[lag] = np.sum(coeff_a[correlation_match])

    return G

@numba.jit(nopython=True)
def pnormalize(G, t, u, bins):
    r"""Normalize point-process cross-correlation function.
    This normalization is usually employed for fluorescence correlation
    spectroscopy (FCS) analysis.
    The normalization is performed according to
    `(Laurence 2006) <https://doi.org/10.1364/OL.31.000829>`__.
    Basically, the input argument `G` is multiplied by:
    .. math::
        \frac{T-\tau}{n(\{i \ni t_i \le T - \tau\})n(\{j \ni u_j \ge \tau\})}
    where `n({})` is the operator counting the elements in a set, *t* and *u*
    are the input arrays of the correlation, *τ* is the time lag and *T*
    is the measurement duration.
    Arguments:
        G (array): raw cross-correlation to be normalized.
        t (array): first input array of "points" used to compute `G`.
        u (array): second input array of "points" used to compute `G`.
        bins (array): array of bins used to compute `G`. Needs to have the
            same units as input arguments `t` and `u`.
    Returns:
        Array of normalized values for the cross-correlation function,
        same size as the input argument `G`.
    """
    duration = max((t.max(), u.max())) - min((t.min(), u.min()))
    Gn = G.copy()
    # for i, tau in enumerate(bins[1:]):
    #     Gn[i] *= ((duration - tau) /
    #               (float((t >= tau).sum()) *
    #                float((u <= (u.max() - tau)).sum())))
    for i, tau in enumerate(bins[1:]):
        Gn[i] *= ((duration - tau) /
                  (float((t - t[0] >= tau).sum()) *
                   float((u - u[0] <= (u.max() - u[0] - tau)).sum())))
    return Gn


@numba.jit(nopython=True)
def pcorrelate(t, u, bins, normalize=False):
    """Compute correlation of two arrays of discrete events (Point-process).
    The input arrays need to be values of a point process, such as
    photon arrival times or positions. The correlation is efficiently
    computed on an arbitrary array of lag-bins. As an example, bins can be
    uniformly spaced in log-space and span several orders of magnitudes.
    (you can use :func:`make_loglags` to creat log-spaced bins).
    This function implements the algorithm described in
    `(Laurence 2006) <https://doi.org/10.1364/OL.31.000829>`__.
    Arguments:
        t (array): first array of "points" to correlate. The array needs
            to be monothonically increasing.
        u (array): second array of "points" to correlate. The array needs
            to be monothonically increasing.
        bins (array): bin edges for lags where correlation is computed.
        normalize (bool): if True, normalize the correlation function
            as typically done in FCS using :func:`pnormalize`. If False,
            return the unnormalized correlation function.
    Returns:
        Array containing the correlation of `t` and `u`.
        The size is `len(bins) - 1`.
    See also:
        :func:`make_loglags` to genetate log-spaced lag bins.
    """
    nbins = len(bins) - 1

    # Array of counts (histogram)
    counts = np.zeros(nbins, dtype=np.int64)

    # For each bins, imin is the index of first `u` >= of each left bin edge
    imin = np.zeros(nbins, dtype=np.int64)
    # For each bins, imax is the index of first `u` >= of each right bin edge
    imax = np.zeros(nbins, dtype=np.int64)

    # For each ti, perform binning of (u - ti) and accumulate counts in Y
    for ti in t:
        for k, (tau_min, tau_max) in enumerate(zip(bins[:-1], bins[1:])):
            #print ('\nbin %d' % k)

            if k == 0:
                j = imin[k]
                # We start by finding the index of the first `u` element
                # which is >= of the first bin edge `tau_min`
                while j < len(u):
                    if u[j] - ti >= tau_min:
                        break
                    j += 1

            imin[k] = j
            if imax[k] > j:
                j = imax[k]
            while j < len(u):
                if u[j] - ti >= tau_max:
                    break
                j += 1
            imax[k] = j
            # Now j is the index of the first `u` element >= of
            # the next bin left edge
        counts += imax - imin
    G = counts / np.diff(bins)
    if normalize:
        G = pnormalize(G, t, u, bins)
    return G

@numba.jit(nopython=True)
def pnormalize_coeff(G, t, u, bins):
    r"""Normalize point-process cross-correlation function.
    This normalization is usually employed for fluorescence correlation
    spectroscopy (FCS) analysis.
    The normalization is performed according to
    `(Laurence 2006) <https://doi.org/10.1364/OL.31.000829>`__.
    Basically, the input argument `G` is multiplied by:
    .. math::
        \frac{T-\tau}{n(\{i \ni t_i \le T - \tau\})n(\{j \ni u_j \ge \tau\})}
    where `n({})` is the operator counting the elements in a set, *t* and *u*
    are the input arrays of the correlation, *τ* is the time lag and *T*
    is the measurement duration.
    Arguments:
        G (array): raw cross-correlation to be normalized.
        t (array): first input array of "points" used to compute `G`.
        u (array): second input array of "points" used to compute `G`.
        bins (array): array of bins used to compute `G`. Needs to have the
            same units as input arguments `t` and `u`.
    Returns:
        Array of normalized values for the cross-correlation function,
        same size as the input argument `G`.
    """
    duration = max((t.max(), u.max())) - min((t.min(), u.min()))
    Gn = G.copy()
    # for i, tau in enumerate(bins[1:]):
    #     Gn[i] *= ((duration - tau) /
    #               (float((t >= tau).sum()) *
    #                float((u <= (u.max() - tau)).sum())))
    for i, tau in enumerate(bins[1:]):
        Gn[i] *= ((duration - tau) /
                  (float((t - t[0] >= tau).sum()) *
                   float((u - u[0] <= (u.max() - u[0] - tau)).sum())))
    return Gn

@numba.jit(nopython=True)
def pcorrelate_coeff(t, u, bins, coeff, normalize=False):
    """Compute correlation of two arrays of discrete events (Point-process).
    The input arrays need to be values of a point process, such as
    photon arrival times or positions. The correlation is efficiently
    computed on an arbitrary array of lag-bins. As an example, bins can be
    uniformly spaced in log-space and span several orders of magnitudes.
    (you can use :func:`make_loglags` to creat log-spaced bins).
    This function implements the algorithm described in
    `(Laurence 2006) <https://doi.org/10.1364/OL.31.000829>`__.
    Arguments:
        t (array): first array of "points" to correlate. The array needs
            to be monothonically increasing.
        u (array): second array of "points" to correlate. The array needs
            to be monothonically increasing.
        bins (array): bin edges for lags where correlation is computed.
        coeff (array) : array of coefficient for the tau-FCS.
        normalize (bool): if True, normalize the correlation function
            as typically done in FCS using :func:`pnormalize`. If False,
            return the unnormalized correlation function.
    Returns:
        Array containing the correlation of `t` and `u`.
        The size is `len(bins) - 1`.
    See also:
        :func:`make_loglags` to genetate log-spaced lag bins.
    """
    nbins = len(bins) - 1

    # Array of counts (histogram)
    counts = np.zeros(nbins, dtype=np.int64)

    # For each bins, imin is the index of first `u` >= of each left bin edge
    imin = np.zeros(nbins, dtype=np.int64)
    # For each bins, imax is the index of first `u` >= of each right bin edge
    imax = np.zeros(nbins, dtype=np.int64)

    # For each ti, perform binning of (u - ti) and accumulate counts in Y
    for ti in t:
        for k, (tau_min, tau_max) in enumerate(zip(bins[:-1], bins[1:])):
            #print ('\nbin %d' % k)

            if k == 0:
                j = imin[k]
                # We start by finding the index of the first `u` element
                # which is >= of the first bin edge `tau_min`
                while j < len(u):
                    if u[j] - ti >= tau_min:
                        break
                    j += 1

            imin[k] = j
            if imax[k] > j:
                j = imax[k]
            while j < len(u):
                if u[j] - ti >= tau_max:
                    break
                j += 1
            imax[k] = j
            # Now j is the index of the first `u` element >= of
            # the next bin left edge
        counts += imax - imin
    G = counts / np.diff(bins)
    if normalize:
        G = pnormalize(G, t, u, bins)
    return G

@numba.jit('uint32[:](uint64[:], uint32[:], int64)', nopython=True)
def ucorrelate_coeff(t_stamps_a, coeff, max_lag=None):
    # if coeff is None:
    #     coeff = np.ones(t_stamps_a.size)

    if max_lag is None:
        max_lag = t_stamps_a.size

    G = np.zeros(max_lag, dtype=np.uint32)
    max_lag_minus_1 = max_lag - 1



    for n in range(t_stamps_a.size):
        j = n + 1   # We don't take inot account the 0 lag time.
        if j == t_stamps_a.size:
            break

        while True:
            lag = t_stamps_a[j] - t_stamps_a[n]
            if lag > max_lag_minus_1:
                break
            else:
                G[lag] += coeff[j]
                j += 1
                if j == t_stamps_a.size:
                    break
    return G

@numba.jit
def ucorrelate(t, u, maxlag=None):
    """Compute correlation of two signals defined at uniformly-spaced points.
    The correlation is defined only for positive lags (including zero).
    The input arrays represent signals defined at uniformily-spaced
    points. This function is equivalent to :func:`numpy.correlate`, but can
    efficiently compute correlations on a limited number of lags.
    Note that binning point-processes with uniform bins, provides
    signals that can be passed as argument to this function.
    Arguments:
        tx (array): first signal to be correlated
        ux (array): second signal to be correlated
        maxlag (int): number of lags where correlation is computed.
            If None, computes all the lags where signals overlap
            `min(tx.size, tu.size) - 1`.
    Returns:
        Array contained the correlation at different lags.
        The size of this array is equal to the input argument `maxlag`
        (if defined) or to `min(tx.size, tu.size) - 1`.
    Example:
        Correlation of two signals `t` and `u`::
            >>> t = np.array([1, 2, 0, 0])
            >>> u = np.array([0, 1, 1])
            >>> pycorrelate.ucorrelate(t, u)
            array([2, 3, 0])
        The same result can be obtained with numpy swapping `t` and `u` and
        restricting the results only to positive lags::
            >>> np.correlate(u, t, mode='full')[t.size - 1:]
            array([2, 3, 0])
    """
    if maxlag is None:
        maxlag = u.size
    maxlag = int(min(u.size, maxlag))
    C = np.zeros(maxlag, dtype=np.int64)
    for lag in range(C.size):
        tmax = min(u.size - lag, t.size)
        umax = min(u.size, t.size + lag)
        C[lag] = (t[:tmax] * u[lag:umax]).sum()
    return C


def make_loglags(exp_min, exp_max, points_per_base, base=10):
    """Make a log-spaced array useful as lag bins for cross-correlation.
    This function conveniently creates an arrays on lag-bins to be used
    with :func:`pcorrelate`.
    Arguments:
        exp_min (int): exponent of the minimum value
        exp_max (int): exponent of the maximum value
        points_per_base (int): number of points per base
            (i.e. in a decade when `base = 10`)
        base (int): base of the exponent. Default 10.
    Returns:
        Array of log-spaced values with specified range and spacing.
    Example:
        Compute log10-spaced bins with 2 bins per decade, starting
        from 10⁻¹ and stopping at 10³::
            >>> make_loglags(-1, 3, 2)
            array([  1.00000000e-01,   3.16227766e-01,   1.00000000e+00,
                     3.16227766e+00,   1.00000000e+01,   3.16227766e+01,
                     1.00000000e+02,   3.16227766e+02,   1.00000000e+03])
    See also:
        :func:`pcorrelate`
    """
    num_points = points_per_base * (exp_max - exp_min) + 1
    bins = np.logspace(exp_min, exp_max, num_points, base=base)
    return bins