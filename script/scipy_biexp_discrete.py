import numpy as np
from scipy.stats import rv_discrete, rv_continuous
import matplotlib.pyplot as plt


# class bi_exp_gen_tail_a1a2(rv_continuous):
#     # def _pdf(self, t, t0, a1, a2, tau1, tau2):
#     #         return (a1*np.exp(-(t-t0)/tau1) + a2*np.exp(-(t-t0)/tau2))
#     def _pdf(self, t, t0, a1, a2, tau1, tau2):
#         C = 1 / (a1 * tau1 + a2 * tau2)
#         pdf = C * (a1 * np.exp(-(t - t0) / tau1) + a2 * np.exp(-(t - t0) / tau2))
#         pdf[t < t0] = 0
#         return pdf
#
#     def _cdf(self, t, t0, a1, a2, tau1, tau2):
#         C = 1 / (a1 * tau1 + a2 * tau2)
#         cdf = C * (a1 * tau1 * (1 - np.exp((t0 - t) / tau1)) + a2 * tau2 * (1 - np.exp((t0 - t) / tau2)))
#         cdf[t < t0] = 0
#         return cdf


nbOfMicrotimeChannel = 4096
time_step_s = (60e-9 / nbOfMicrotimeChannel)  # time step in seconds (S.I.)
time_step_ns = time_step_s * 1e9  # time step in nano-seconds
time_nbins = nbOfMicrotimeChannel  # number of time bins

time_idx = np.arange(time_nbins)  # time axis in index units
time_ns = time_idx * time_step_ns  # time axis in nano-seconds


t0 = 12
a1 = 200
a2 = 100
tau1 = 2
tau2 = 9

C = 1 / (a1 * tau1 + a2 * tau2)
decay_ns = C*(a1 * np.exp(-(time_ns - t0) / tau1) + a2 * np.exp(-(time_ns - t0) / tau2))
decay_ns[time_ns < t0] = 0
# decay_microchannel = decay_ns*time_nbins
decay_ns /= decay_ns.sum()

decay_obj = rv_discrete(name='biexp', values=(time_idx, decay_ns))

num_samples = 10000

bi_exp_rvs = decay_obj.rvs(size=num_samples)
print(bi_exp_rvs)
print(np.mean(bi_exp_rvs))

decay_hist, bins = np.histogram(bi_exp_rvs, bins=time_nbins)

plt.semilogy(bins[0:-1], decay_hist)
plt.show()