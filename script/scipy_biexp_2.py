import numpy as np
from scipy.stats import rv_discrete, rv_continuous
import matplotlib.pyplot as plt


class bi_exp_gen_tail_a1a2(rv_continuous):
    # def _pdf(self, t, t0, a1, a2, tau1, tau2):
    #         return (a1*np.exp(-(t-t0)/tau1) + a2*np.exp(-(t-t0)/tau2))
    def _pdf(self, t, t0, a1, a2, tau1, tau2):
        C = 1 / (a1 * tau1 + a2 * tau2)
        pdf = C * (a1 * np.exp(-(t - t0) / tau1) + a2 * np.exp(-(t - t0) / tau2))
        pdf[t < t0] = 0
        return pdf

    def _cdf(self, t, t0, a1, a2, tau1, tau2):
        C = 1 / (a1 * tau1 + a2 * tau2)
        cdf = C * (a1 * tau1 * (1 - np.exp((t0 - t) / tau1)) + a2 * tau2 * (1 - np.exp((t0 - t) / tau2)))
        cdf[t < t0] = 0
        return cdf

t0 = 5
a1 = 200
a2 = 100
tau1 = 5
tau2 = 15




# biexp_val = bi_exp(t, t0, a1, a2, tau1, tau2)
# biexp_val_proba = biexp_val / biexp_val.sum()
# bi_exp_rv = rv_discrete(name='bi_exp', values=(t, biexp_val_proba))
#
# num_samples = 3000
#
# sampled_bi_exp = bi_exp_rv.rvs(size=num_samples)

bi_exp_gen_obj = bi_exp_gen_tail_a1a2(a=0, b=50, name="bi_exp_gen")

print(bi_exp_gen_obj.cdf(40, t0=t0, a1=a1, a2=a2, tau1=tau1, tau2=tau2))

num_samples = 1000
bi_exp_rvs = bi_exp_gen_obj.rvs(size=num_samples,  t0=t0, a1=a1, a2=a2, tau1=tau1, tau2=tau2)
print(bi_exp_rvs)
