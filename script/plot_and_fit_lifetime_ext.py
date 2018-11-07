import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import os
import sys

py_SPC_path = os.path.normpath(r"C:\TRAVAIL\recherche\code\pySPC")
print(py_SPC_path)
sys.path.insert(0, py_SPC_path)

from core import Experiment


# Relative path
datapath = r"C:\Users\MatthieuL\Documents\data\Olia"
filepath = r"OF1430_405nm_Pnc_405Dichro_405BP_80-20-DMSO-Water_1mM.spc"

filepath = os.path.normpath(os.path.join(datapath, filepath))

# Create the main (model) object and fill it with the spc file
exp = Experiment.Experiment(filepath)

# Calculate the microtime histogramm and fill the corresponding result object
exp.micro_time_life_time()

# Alias variables from the results object.
# There are as many lifetime results as there are channels.
lifetime_histogram = exp.results.lifetimes[0].data
lifetime_time_axis = exp.results.lifetimes[0].timeAxis

# Plot
plt.plot(lifetime_time_axis, lifetime_histogram)
plt.xlabel("time (ns)")
plt.ylabel("Occurence")
plt.show()
