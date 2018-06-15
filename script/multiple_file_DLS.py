# -*- coding: utf-8 -*-
from core import ExpParam
from core import Results
from core import Data
from core import Experiment

import glob, os
import re

# Put directory of the spc files :
# os.chdir(r'C:\Users\MatthieuL\Documents\data\2018_05_24 deuxieme Calibration DLS\Or')
# os.chdir(r'C:\Users\MatthieuL\Documents\data\2018_05_24 deuxieme Calibration DLS\Arg')
os.chdir(r'C:\Users\MatthieuL\Documents\data\2018_05_24 deuxieme Calibration DLS\Fluo_3xdilue')
# os.chdir(r'C:\Users\MatthieuL\Documents\data\DLS')

# Correlation parameters
max_correlTime_ms = 100
start_time_mu_s = 8
precision = 10

angle_list = []
CPS_list = []
tau_list = []
beta_list = []

B_ini = 1
Beta_ini = 0.8
tau_ini = 350
mu_2_ini=0
mu_3_ini=0
mu_4_ini=0

for spc_file in glob.glob("*.spc"):
    print(spc_file)

    # print(int(list(filter(str.isdigit, spc_file))[0]))
    # Extract angle in filename
    angle_list.append(int(re.findall(r'\d+', spc_file)[0]))


    exp = Experiment.Experiment()
    exp.new_exp("file", [spc_file])

    CPS = exp.data.channels[0].CPS
    CPS_list.append(CPS)

    print("CPS : %f" % CPS)

    t1_tick = exp.data.channels[0].startTick
    t2_tick = exp.data.channels[0].endTick
    print("Calculating correlation")
    exp.data.DLS(0, 0, t1_tick, t2_tick, max_correlTime_ms, start_time_mu_s, precision)

    print("Fitting")
    exp.results.DLS_Measurements[0].set_model("Cumulant")
    exp.results.DLS_Measurements[0].guess(idx_start=0, idx_end=-1)
    exp.results.DLS_Measurements[0].set_params([B_ini, Beta_ini, tau_ini, mu_2_ini, mu_3_ini, mu_4_ini])



    fitResults = exp.results.DLS_Measurements[0].fit(idx_start=0, idx_end=-1)
    print(fitResults.best_values)
    tau_list.append(fitResults.best_values['tau'])
    beta_list.append(fitResults.best_values['beta'])

    print(fitResults.fit_report())
    # print(tau_list)

f = open("result.txt", "w")

i = 0
for angle in angle_list:
    f.write("%d %d %f %f\n" % (angle, CPS_list[i], tau_list[i], beta_list[i]))
    i+=1

f.close()

