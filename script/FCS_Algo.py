import numpy as np
import matplotlib.pyplot as plt

import os
import sys

if __name__ == '__main__':

    py_SPC_path = os.path.normpath(r"C:\TRAVAIL\recherche\code\pySPC")
    print(py_SPC_path)
    sys.path.insert(0, py_SPC_path)

    from core import Experiment, Experiments

    datapath = r"C:\Users\MatthieuL\Documents\data\2018_10_24_TestFCS_BillesFluo"
    filepath = r"OF1440_H2oPure_c0sur10000_x20_DichroBPLP405_Pmax_Fibre25.spc"
    filepath = r"BilleFluo30nano_c0sur10000_x20_DichroBPLP405_Pmax_Fibre25.spc"

    filepath = os.path.normpath(os.path.join(datapath, filepath))
    # Create the main (model) object and fill it with the spc file
    exps = Experiments.Experiments()
    # exp_100nm = exps.add_new_exp("file", [filepath])
    # exp_poisson = Experiment.Experiment()
    exp_poisson = exps.add_new_exp(mode="generate", params=["Poisson", 2, 5000])
    FCS_measurement = exp_poisson.create_measurement(num_channel=0, start_tick=0, end_tick=-1, type_="FCS", name="FCS_poisson", comment="", is_store=True)

    exp_poisson.calculate_FCS(measurement=FCS_measurement, num_c1=0, num_c2=0, start_cor_time_micros=0.5, max_cor_time_ms=100)

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