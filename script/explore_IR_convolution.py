import numpy as np
import matplotlib.pyplot as plt

import os
import sys

py_SPC_path = os.path.normpath(r"C:\TRAVAIL\recherche\code\pySPC")
print(py_SPC_path)
sys.path.insert(0, py_SPC_path)

from core import Experiment

# Relative path
datapath = r"C:\Users\MatthieuL\Documents\data\2018_10_23_TpsDeVieOlia_AIE"
filepath_IR =  r"IR_relfexion_coverslip.spc"
filepath_Fluo =  r"Test_perylene_etOh_gly.spc"

filepath_IR = os.path.normpath(os.path.join(datapath, filepath_IR))
exp_IR = Experiment.Experiment(filepath_IR)
lifetime_measurement_IR = exp_IR.micro_time_life_time()
# fig_lifteime_IR = lifetime_measurement.create_canonic_graph()
# fig_lifteime_IR.show()
# fig_lifteime_IR.savefig("test.png")

# filepath_Fluo = os.path.normpath(os.path.join(datapath, filepath_Fluo))
# exp_fluo = Experiment.Experiment(filepath_Fluo)
# lifetime_measurement_Fluo = exp_fluo.micro_time_life_time()
# fig_lifteime_Fluo = lifetime_measurement_Fluo.create_canonic_graph()
# fig_lifteime_Fluo.show()
# fig_lifteime_Fluo.savefig("test.png")


def one_exp_decay(t, t0, amp, tau, cst):
    return cst + amp * np.exp(-(t - t0) / tau)


def convoluted_one_exp_decay(t, t0, amp, tau, cst):
    return cst + amp * np.exp(-(t - t0) / tau)

exp_decay = np.copy(lifetime_measurement_IR.time_axis)


lifetime_measurement_IR.data = lifetime_measurement_IR.data.astype(np.float64) / np.max(lifetime_measurement_IR.data)

nb_of_point = exp_decay.size

t0 = 0
cst = 0
idx_t0 = np.searchsorted(exp_decay, t0)
exp_decay = one_exp_decay(exp_decay, t0, 1, 5, cst)
exp_decay[0:idx_t0] = cst

exp_decay_convolved = np.convolve(exp_decay, lifetime_measurement_IR.data)
exp_decay_convolved = exp_decay_convolved[0:nb_of_point]

test_conv_single_point = np.convolve(1, lifetime_measurement_IR.data)
# plt.plot(test_conv_single_point)



# plt.plot(lifetime_measurement_IR.time_axis, exp_decay)
# plt.plot(lifetime_measurement_IR.time_axis, exp_decay_convolved)
plt.show()
plt.savefig("test2.png")



print("End")