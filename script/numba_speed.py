import numba
import numpy as np
import matplotlib.pyplot as plt
import time

from correlate import correlate

@numba.jit('uint32[:](uint64[:])', nopython=True)
def ucorrelate_coeff(t_stamps_a):
    # if coeff is None:
    #     coeff = np.ones(t_stamps_a.size)

    max_lag = int(1/(25E-9))

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
                G[lag] += 1
                j += 1
                if j == t_stamps_a.size:
                    break
    return G

# 'uint64[:](uint64[:], uint64[:])',
@numba.jit(nopython=True)
def ucorrelate(t, u):
    max_lag = int(1 / (25E-9))

    max_lag = int(min(u.size, max_lag))

    C = np.zeros(max_lag, dtype=np.uint64)
    for lag in range(C.size):
        tmax = min(u.size - lag, t.size)
        umax = min(u.size, t.size + lag)
        C[lag] = (t[:tmax] * u[lag:umax]).sum()
    return C


time_s = 10
mean_rate = 1000

tick_duration = 25E-9

t_end_click = time_s / tick_duration
mean_rate_in_tick = mean_rate * tick_duration

mean_rates = [1000, 2000, 5000, 10000, 20000, 50000, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7]
mean_rates = mean_rates[0:6]
timings_numba = []
timings_cython = []
for mean_rate in mean_rates:

    mean_rate_in_tick = mean_rate * tick_duration
    nb_of_tick_to_generate = int((t_end_click) * mean_rate_in_tick)
    print("mean_rate : ", mean_rate, " nb of photon : ", nb_of_tick_to_generate)
    arrival_times = np.cumsum(
        -(np.log(1.0 - np.random.random(nb_of_tick_to_generate)) / mean_rate_in_tick).astype(np.uint64))

    start = time.time()
    G = ucorrelate_coeff(arrival_times)
    end = time.time()
    timing = end - start
    print("time Numba :", timing)
    timings_numba.append(timing)

    # # max_lag = int(1 / (25E-9))
    # # G = np.zeros(max_lag, dtype=np.int)
    # start = time.time()
    # # correlate(arrival_times, G)
    # ucorrelate(arrival_times, arrival_times)
    # end = time.time()
    # timing = end - start
    # print("time Cython :", timing)
    # timings_cython.append(timing)


    # plt.semilogx(G)
    # plt.show()

plt.plot(mean_rates, timings_numba)
plt.plot(mean_rates, timings_cython)
plt.show()