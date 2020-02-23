from core import Experiment
# import shelve
import os
import numpy as np

class Experiments(object):
    """
    Root class of the model (named here "core") of the MVC pattern.

    main members :
    - experiments : a list of experiments which whom perfom *global* analysis

    main methods :
    - new_exp
    - save_state/load_state
    """

    def __init__(self):
        self.experiments = {}
        self.irf = {}
        self.logger = None

    def add_new_exp(self, mode, params_dict):
        #TODO test if creation of exp is successfull

        exp = Experiment.Experiment(mode, params_dict, exps=self)
        self.experiments[exp.file_name] = exp
        return exp

    def add_exp(self, exp):
        self.experiments[exp.file_name] = exp

    def get_exp(self, exp_name):
        if exp_name in self.experiments:
            return self.experiments[exp_name]
        else:
            return None

    def save_state(self, shelf):
        # self.shelf = shelve.open(savefile_path, 'n') #n for new
        # self.shelf['experiments'] = self.experiments

        # shelf['irf'] = self.irf
        #
        # for key in self.experiments.keys():
        #     exp = self.experiments[key]
        #     for key_m in exp.measurements.keys():
        #         mes = exp.measurements[key_m]
        #         # shelf[mes.name] = mes
        #         for item in dir(mes):
        #             print(item)
        #             shelf[str(item)] = item
        #
        #
        # shelf[exp.file_name] = exp

        # FIXME cause a bug ???
        shelf['experiments'] = self.experiments


    def load_state(self, shelf):
        # klist = list(shelf.keys())
        #
        # for key in klist:
        #     if key != "irf":
        #         self.add_exp(shelf[key])

        # FIXME
        self.experiments = shelf['experiments']

    def global_tau_FCS(self):
        pass

    def calculate_measurement(self, exp_name, measurement_name, params):
        if exp_name in self.experiments:
            exp = self.experiments[exp_name]
            if measurement_name in exp.measurements:
                measurement = exp.measurements[measurement_name]
                measurement.set_additional_param_for_calculation(params)
                measurement.calculate()

    def get_IRF_from_file(self, file_path):
        filename, file_extension = os.path.splitext(file_path)
        if file_extension in ["txt", "dat"]:
            # TODO IRF from ascii FIle
            raw = np.loadtxt(file_path)
            pass
        ir_exp = Experiment.Experiment("file", [file_path])
        measurement_ir = ir_exp.create_measurement(0, 0, -1,"lifetime", "", "")
        measurement_ir.calculate()
        # ir_exp.calculate_life_time(measurement_ir)
        return ir_exp.file_name, measurement_ir.data, measurement_ir.time_axis

    def add_irf(self, irf):
        self.irf[irf.name] = irf

    def get_irf_name_list(self):
        return list(self.irf.keys())

    def get_raw_data(self, exp, num_channel=0, start_tick=0, end_tick=-1, type="timestamp", mode="data"):
        return exp.get_raw_data(num_channel, start_tick, end_tick, type, mode)

    def export_raw_data(self, exp, nb_of_photon=1000, num_channel=0, type="timestamp", mode="data"):
        raw = self.get_raw_data(exp, mode="full")
        raw = raw[0:nb_of_photon]
        np.savetxt("export.txt", raw)

