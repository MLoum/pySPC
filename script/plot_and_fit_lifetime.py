from core import Experiment
import matplotlib.pyplot as plt
import numpy as np
import os

# Relative path
filepath = "../test/spc/fluo2.spc"

# Absolute path
filepath = os.path.join(os.getcwd(), filepath)
filepath = os.path.normpath(filepath)
# print(filepath)

# Create the main (model) object and fill it with the spc file
exp = Experiment.Experiment(filepath)

# Calculate the microtime histogramm and fill the corresponding result object
exp.micro_time_life_time()

# Alias variables from the results object.
# There are as many lifetime results as there are channels.
lifetime_histogram = exp.results.lifetimes[0].data
lifetime_time_axis = exp.results.lifetimes[0].time_axis

# fig = exp.results.lifetimes[0].create_canonic_graph()
# fig.savefig("test.png")
# (self, is_plot_error_bar=False, is_plot_text=False):

# Plot
# plt.plot(lifetime_time_axis, lifetime_histogram)
# plt.xlabel("time (ns)")
# plt.ylabel("Occurence")
# plt.show()

# Fit
t0_ini = 7
amp_ini = 457
tau_ini = 3
cst_ini = 150


exp.results.lifetimes[0].set_model("One Decay")
# exp.results.lifetimes[0].guess(idx_start=idx_decay_curve_start, idx_end=-1)
exp.results.lifetimes[0].set_params([t0_ini, amp_ini, tau_ini, tau_ini, cst_ini])
fit_results = exp.results.lifetimes[0].fit(idx_start=t0_ini, idx_end=-1)
print(fit_results)


print(fit_results.best_values)

fig, GridSpec  = fit_results.plot()
fig.show()
# fig = exp.results.lifetimes[0].create_canonic_graph()
fig.savefig("test.png")

# # Plot
# plt.plot(lifetime_time_axis, lifetime_histogram)
# plt.plot(lifetime_time_axis[], fit_results.best_fit)
# plt.xlabel("time (ns)")
# plt.ylabel("Occurence")
# plt.show()