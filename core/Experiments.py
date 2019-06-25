from core import Experiment
import shelve

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

    def add_new_exp(self, mode, params):
        #TODO test if creation of exp is successfull

        exp = Experiment.Experiment(mode, params)
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
        shelf['experiments'] = self.experiments

        # shelf.close()

    def load_state(self, shelf):
        self.experiments = shelf['experiments']

    def global_tau_FCS(self):
        pass

    def calculate_measurement(self, exp_name, measurement_name, params):
        if exp_name in self.experiments:
            exp = self.experiments[exp_name]
            if measurement_name in exp.measurements:
                measurement = exp.measurements[measurement_name]
                type_ = measurement.type
                if type_ == "FCS":
                    num_c1, num_c2, start_cor_time_micros, max_cor_time_ms = params
                    exp.calculate_FCS(measurement, num_c1, num_c2, start_cor_time_micros, max_cor_time_ms)
                elif type_ == "lifetime":
                    exp.calculate_life_time(measurement)
                elif type_ == "DLS":
                    exp.calculate_DLS(measurement)

