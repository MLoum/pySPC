# -*- coding: utf-8 -*-
from core import ExpParam
from core import Results
from core import Data
from core import Experiments, Experiment

import matplotlib.pyplot as plt
import glob, os
import re

# Put directory of the spc files :
# os.chdir(r'C:\Users\MatthieuL\Documents\data\2018_05_24 deuxieme Calibration DLS\Or')
# os.chdir(r'C:\Users\MatthieuL\Documents\data\2018_05_24 deuxieme Calibration DLS\Arg')
# os.chdir(r'C:\Users\MatthieuL\Documents\data\2018_05_24 deuxieme Calibration DLS\Fluo_3xdilue')
# os.chdir(r'C:\Users\MatthieuL\Documents\data\DLS')
os.chdir(r'C:\Users\MatthieuL\Documents\data\2019_60_20_DLS')

# Correlation parameters
max_correlTime_ms = 100
start_time_mu_s = 15
precision = 10

angle_list = []
CPS_list = []
tau_list = []
beta_list = []
mu2_list = []
filename_list = []

B_ini = 1
Beta_ini = 0.8
tau_ini = 350
mu_2_ini=0
mu_3_ini=0
mu_4_ini=0

expS = Experiments.Experiments()

file_name = ["2019_06_20_DLS_KTP_polyD_633nm_300dg.spc", "2019_06_20_DLS_Au_50nm_633nm_300dg.spc", "2019_06_20_DLS_KTP_polyD_633nm_225dg.spc", "2019_06_20_DLS_Au_50nm_633nm_225dg.spc"]


for spc_file in glob.glob("*.spc"):
    print(spc_file)

    if spc_file not in file_name:
        continue
    # print(int(list(filter(str.isdigit, spc_file))[0]))
    # Extract angle in filename
    # angle_list.append(int(re.findall(r'\d+', spc_file)[0]))
    pos = spc_file.find("dg")
    if pos == -1:
        continue
    angle = spc_file[pos-3:pos]

    angle_list.append(angle)
    filename_list.append(spc_file)
    exp = expS.add_new_exp(mode="file", params=[spc_file])

    CPS = exp.data.channels[0].CPS
    CPS_list.append(CPS)

    print("CPS : %f" % CPS)

    num_channel = 0
    t1_tick = exp.data.channels[0].start_tick
    t2_tick = exp.data.channels[0].end_tick
    print("Calculating correlation")

    measurement = exp.create_measurement(num_channel, t1_tick, t2_tick, type="DLS", name="dls", comment="")
    measurement.start_cor_time_micros = 8
    measurement.num_c1 = 0
    measurement.num_c2 = 0
    measurement.max_cor_time_ms = 500
    measurement.precision = 10
    expS.calculate_measurement(exp.file_name, measurement.name, params=None)


    print("Fitting")
    measurement.set_model("Cumulant")
    measurement.guess(idx_start=0, idx_end=-1)
    measurement.set_params([B_ini, Beta_ini, tau_ini, mu_2_ini, mu_3_ini, mu_4_ini])

    fitResults = measurement.fit(idx_start=0, idx_end=-1)
    print(fitResults.best_values)
    tau_list.append(fitResults.best_values['tau'])
    beta_list.append(fitResults.best_values['beta'])
    # mu2_list.append(fitResults.best_values['mu2'])

    # measurement.create_canonic_graph(is_plot_error_bar=False, is_plot_text=True, save_file_name=spc_file + ".png")
    print(fitResults.fit_report())
    # print(tau_list)


# expS.save_state("dls.save")

exp_300_KTP = expS.get_exp("2019_06_20_DLS_KTP_polyD_633nm_300dg.spc")
exp_300_Au = expS.get_exp("2019_06_20_DLS_Au_50nm_633nm_300dg.spc")
exp_225_KTP = expS.get_exp("2019_06_20_DLS_KTP_polyD_633nm_225dg.spc")
exp_225_Au = expS.get_exp("2019_06_20_DLS_Au_50nm_633nm_225dg.spc")

measurement_dls_300_ktp = exp_300_KTP.get_measurement("dls")
measurement_dls_300_Au = exp_300_Au.get_measurement("dls")
measurement_dls_225_ktp = exp_225_KTP.get_measurement("dls")
measurement_dls_225_Au = exp_225_Au.get_measurement("dls")

ax = plt.gca()
ax.grid(True)
ax.grid(True, which='minor', lw=0.3)



plt.semilogx(measurement_dls_300_ktp.time_axis, measurement_dls_300_ktp.data, "ro", alpha=0.5, label="Ktp")
plt.semilogx(measurement_dls_300_ktp.time_axis[:-1], measurement_dls_300_ktp.fit_results.best_fit, "k")
plt.semilogx(measurement_dls_300_Au.time_axis, measurement_dls_300_Au.data, "bo", alpha=0.5, label="Au 50nm")
plt.semilogx(measurement_dls_300_Au.time_axis[:-1], measurement_dls_300_Au.fit_results.best_fit, "k")
plt.xlim(12, 1E4)
plt.ylim(0.9, 2)
plt.legend()
plt.ylabel('G(τ)', fontsize=20)
plt.xlabel('Time Lag, τ (µs)', fontsize=20)
plt.savefig("export.png", dpi=300)
plt.savefig("C:\Intel\dls_ktp.png", dpi=300)
plt.show()

# f = open("result.txt", "w")
#
# i = 0
# for angle in angle_list:
#     string = filename_list[i]
#     string += ' {} {} {} {}\n'.format(angle, CPS_list[i], tau_list[i], beta_list[i])
#     # f.write("%d %d %f %f %f\n" % (angle, CPS_list[i], tau_list[i], beta_list[i], mu2_list[i]))
#     f.write(string)
#     i += 1
#
# f.close()

