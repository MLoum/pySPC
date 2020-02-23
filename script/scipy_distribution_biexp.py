import numpy as np
from scipy.stats import rv_discrete, rv_continuous
import matplotlib.pyplot as plt




def bi_exp(t, t0, a1, a2, tau1, tau2):
    return a1*np.exp(-(t-t0)/tau1) + a2*np.exp(-(t-t0)/tau2)

# class bi_exp_gen(rv_continuous):
#     def _pmf(self, t, t0, a1, a2, tau1, tau2):
#         return a1*np.exp(-(t-t0)/tau1) + a2*np.exp(-(t-t0)/tau2)

class bi_exp_gen(rv_continuous):
    # def _pdf(self, t, t0, a1, a2, tau1, tau2):
    #         return (a1*np.exp(-(t-t0)/tau1) + a2*np.exp(-(t-t0)/tau2))
    def _pdf(self, t, t0, a1, a2, tau1, tau2):
        if np.isscalar(t):
            if t < t0:
                return 0
            else:
                C = 1 / (a1 * tau1 + a2 * tau2)
                return C * (a1 * np.exp(-(t - t0) / tau1) + a2 * np.exp(-(t - t0) / tau2))
        else:
            C = 1 / (a1 * tau1 + a2 * tau2)
            for i in range(t.size):
                if t[i] < t0:
                    return 0
                else:
                    return C*(a1*np.exp(-(t-t0)/tau1) + a2*np.exp(-(t-t0)/tau2))

    def _cdf(self, t, t0, a1, a2, tau1, tau2):
        if np.isscalar(t):
            if t < t0:
                return 0
            else:
                C = 1 / (a1 * tau1 + a2 * tau2)
                return C * (a1 * tau1*(1 - np.exp((t0 - t) / tau1)) + a2 * tau2 * (1 - np.exp((t0 - t) / tau2)))
        else:
            for i in range(t.size):
                if t[i] < t0:
                    return 0
                else:
                    C = 1 / (a1 * tau1 + a2 * tau2)
                    return C * (a1 * tau1*(1 - np.exp((t0 - t) / tau1)) + a2 * tau2 * (1 - np.exp((t0 - t) / tau2)))

def cdf_bi_exp(t, t0, a1, a2, tau1, tau2):
    C = 1 / (a1 * tau1 + a2 * tau2)
    cdf = C * (a1 * tau1*(1 - np.exp((t0 - t) / tau1)) + a2 * tau2 * (1 - np.exp((t0 - t) / tau2)))
    cdf[t < t0] = 0
    return cdf


t0 = 5
a1 = 200
a2 = 100
tau1 = 5
tau2 = 15

t = np.linspace(0, 50, 1000)
bi_exp_t0 = a1*np.exp(-(t-t0)/tau1) + a2*np.exp(-(t-t0)/tau2)
bi_exp_t0[t<t0]=0
plt.semilogy(t, bi_exp_t0)
plt.show()

# cdf = []
# for time in t:
#     cdf.append(cdf_bi_exp(time, t0=t0, a1=a1, a2=a2, tau1=tau1, tau2=tau2))

plt.plot(t, cdf_bi_exp(t, t0=t0, a1=a1, a2=a2, tau1=tau1, tau2=tau2))
plt.show()


# biexp_val = bi_exp(t, t0, a1, a2, tau1, tau2)
# biexp_val_proba = biexp_val / biexp_val.sum()
# bi_exp_rv = rv_discrete(name='bi_exp', values=(t, biexp_val_proba))
#
# num_samples = 3000
#
# sampled_bi_exp = bi_exp_rv.rvs(size=num_samples)

bi_exp_gen_obj = bi_exp_gen(a=0, b=50, name="bi_exp_gen")

print(bi_exp_gen_obj.cdf(40, t0=t0, a1=a1, a2=a2, tau1=tau1, tau2=tau2))

num_samples = 300
bi_exp_rvs = bi_exp_gen_obj.rvs(size=num_samples, t0=t0, a1=a1, a2=a2, tau1=tau1, tau2=tau2)
# bi_exp_rvs /= num_samples
print(bi_exp_rvs)

print(np.mean(bi_exp_rvs))
#
# plt.semilogy(t, bi_exp_gen_obj.pdf(t, t0=t0, a1=a1, a2=a2, tau1=tau1, tau2=tau2))
# plt.show()
decay_hist, bins = np.histogram(bi_exp_rvs)

plt.semilogy(bins[0:-1], decay_hist)
plt.show()