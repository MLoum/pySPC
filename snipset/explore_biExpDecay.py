import numpy as np
import matplotlib.pyplot as plt


max_time_ns = 60
t = np.linspace(0, max_time_ns, 4096) # in ns

tau1 = 1.5
tau2 = 6.2
t0 = 3

bckg = 0

a1 = 0.992
decay_1 = a1*np.exp(-(t-t0)/tau1)
decay_1[t < t0] = bckg
decay_2 = (1-a1)*np.exp(-(t-t0)/tau2)
decay_2[t < t0] = bckg

t_ir = 0.4
irf = (t-t0)/t_ir * np.exp(-(t-t0)/t_ir)
irf[t < t0] = 0
irf /= irf.sum()

decay_1 = a1*np.exp(-(t-t0)/tau1)
decay_1[t < t0] = 0
decay_2 = (1-a1)*np.exp(-(t-t0)/tau2)
decay_2[t < t0] = 0
non_convoluted_decay = decay_1 + decay_2 + bckg

observed_count = (non_convoluted_decay - bckg).sum()
non_convoluted_decay /= non_convoluted_decay.sum()

a1_amp = 11884637.6
tau1 =1.66218020
tau2 = 10.22188106
a2_amp =154869.547
# a2_amp =0
bckgnd=144.655553


raw = np.loadtxt("data_biexp.txt")
raw = np.transpose(raw)
tps = raw[0]
data = raw[1]
t0 = 27
decay_sum = a1_amp*np.exp(-(t-t0)/tau1) + a2_amp*np.exp(-(t-t0)/tau2) + bckgnd
decay_sum[t < t0] = 0
decay_sum_convoluted = np.convolve(decay_sum, irf)[0:np.size(decay_sum)]


# plt.semilogy(t, decay)
# plt.title("a1 = " + str(a1) + ", tau1 =" + str(tau1) + ", tau2 =" + str(tau2))
# plt.semilogy(t, decay_1, alpha=0.5, label="short " + str(tau1))
# plt.semilogy(t, decay_2, alpha=0.5, label="long " + str(tau2))
# plt.legend()
# # plt.savefig("dbl_exp_a1_0v99.png")
# plt.ylim(bckg, 1)

decay1_convolved = np.convolve(decay_1, irf)[0:np.size(decay_1)]
decay2_convolved = np.convolve(decay_2, irf)[0:np.size(decay_2)]
decay_convolved = np.convolve(non_convoluted_decay, irf)[0:np.size(non_convoluted_decay)]
decay_convolved /= decay_convolved.sum()
fit = decay_convolved*observed_count + bckg

# plt.semilogy(t, irf, label="irf")
# plt.semilogy(t, decay1_convolved, label="delay1")
# plt.semilogy(t, decay2_convolved, label="delay2")
# plt.semilogy(t, fit, label="sum")
# # plt.semilogy(t, decay_1)
# if bckg == 0:
#     plt.ylim(0.001, 1)
# else:
#     plt.ylim(bckg, 1)
# plt.legend()
# plt.show()

# plt.semilogy(t, decay_sum)
# plt.semilogy(t, decay_sum_convoluted)
plt.semilogy(tps, data)
plt.semilogy(t, decay_sum)
# plt.semilogy(t, decay_1)

plt.show()