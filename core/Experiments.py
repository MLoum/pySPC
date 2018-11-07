from core import Experiment

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
        self.experiments = []

    def add_new_exp(self, filepath=None):
        #TODO test if creation of exp is successfull
        self.experiments(Experiment.Experiment(filepath))

    def add_exp(self, exp):
        self.experiments.append(exp)

    def save_state(self):
        pass

    def load_state(self):
        pass

    def global_tau_FCS(self):
        pass
