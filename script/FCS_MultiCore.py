import numpy as np
import matplotlib.pyplot as plt

import os
import sys

if __name__ == '__main__':

    py_SPC_path = os.path.normpath(r"C:\TRAVAIL\recherche\code\pySPC")
    print(py_SPC_path)
    sys.path.insert(0, py_SPC_path)

    from core import Experiment

    datapath = r"C:\Users\MatthieuL\Documents\data\2018_10_24_TestFCS_BillesFluo"
    filepath = r"OF1440_H2oPure_c0sur10000_x20_DichroBPLP405_Pmax_Fibre25.spc"
    filepath = r"BilleFluo30nano_c0sur10000_x20_DichroBPLP405_Pmax_Fibre25.spc"

    filepath = os.path.normpath(os.path.join(datapath, filepath))
    # Create the main (model) object and fill it with the spc file
    exp_100nm = Experiment.Experiment(filepath)
    # exp_poisson = Experiment.Experiment()
    # exp_poisson.new_exp(mode="generate", params=["Poisson", 10, 10000])
    # chronogram = exp_100nm.chronogram()
    FCS_measurement = exp_100nm.FCS(start_cor_time_micros=10)

    # plt.plot(chronogram.xAxis, chronogram.data)
    # plt.show()


    FCS_measurement.set_model("1 Diff")
    G0_ini = 0.01
    tdiff_ini = 10000
    cst_ini = 1

    FCS_measurement.guess(0)
    # FCS_measurement.set_params([G0_ini, tdiff_ini, cst_ini])
    fit_results = FCS_measurement.fit()
    print(fit_results.best_values)

    FCS_fig = FCS_measurement.create_canonic_graph(is_plot_error_bar=True, is_plot_text=False)
    FCS_fig.savefig("test_FCS.png")
    FCS_fig.show()
    print("end")