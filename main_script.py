import numpy as np
import matplotlib.pyplot as plt

import os
import sys

# print(os.path.abspath('.'))
# print(os.path.abspath('../core'))

py_SPC_path = os.path.normpath(r"C:\TRAVAIL\recherche\code\pySPC")
print(py_SPC_path)
sys.path.insert(0, py_SPC_path)

from core import Experiment


1# Relative path
datapath = r"C:\Users\MatthieuL\Documents\2018_06_20 Tps de Vie Nanotube 1231"
filepath =  r"Nanotube 1231 x40 BP405LP405Dicrho405 fibre sur laser nanotube Random pmax.spc"

filepath = os.path.normpath(os.path.join(datapath, filepath))
print(filepath)

# Create the main (model) object and fill it with the spc file
exp = Experiment.Experiment(filepath)

chronogram = exp.chronogram()
plt.plot(chronogram.xAxis, chronogram.data)
plt.show()

idx_t1 = exp.convert_seconds_in_ticks(15)
idx_t2 = exp.convert_seconds_in_ticks(35)
exp.data.filter_bin_and_threshold(num_channel=0, threshold=60, bin_in_tick=1E5, replacement_mode="poissonian_noise")

chronogram = exp.chronogram()
plt.plot(chronogram.xAxis, chronogram.data, "ro")
plt.show()
dummy = 1