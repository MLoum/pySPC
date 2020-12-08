import numpy as np
from lmfit import minimize


class FitMethod:
    def __init__(self, decay_generator):
        self.decay_generator = decay_generator

    def fit(self):
        """
        virtual method
        :return:
        """
        pass


class SimulatedAnnealing(FitMethod):

    def __init__(self, decay_generator):
        super().__init__(decay_generator)
        # Timing
        # Resultats vs vraie valeur pour faire des tests (comme la corrélation).

    def fit(self):
        # Model parmi la liste.
        self.model = None

        # Axe temporel
        self.time_axis = self.decay_generator.time_ns

        # IRF


        fitting_method = "dual_annealing"

        # Trouver les indices où faire le fit
        # FIXME
        self.idx_start = 0
        self.idx_end = 0
        y_eval_range = self.data[self.idx_start:self.idx_end]
        x_eval_range = self.time_axis[self.idx_start:self.idx_end]
        self.model.x_range = (self.idx_start, self.idx_end)


        # MLE
        def maximum_likelihood_method(params, x, ydata):
            ymodel = self.model.eval(x, params)
            ymodel = ymodel[self.idx_start:self.idx_end]
            ydata_reduc = ydata[self.idx_start:self.idx_end]
            likelyhood = 2 * (ydata_reduc * np.log(ydata_reduc / ymodel)).sum()
            # likelyhood = -2*(ydata_reduc * np.log(ymodel)).sum()
            return likelyhood

        def callback_iteration(params, iter, resid, *args, **kws):
            # TODO draw and report progress
            pass

        self.fit_results = minimize(maximum_likelihood_method, self.params, args=(self.time_axis, self.data),
                                    method=fitting_method,
                                    iter_cb=callback_iteration, nan_policy='propagate')

        #TODO multi-core
        for decay in self.decay_generator.decay_data:
            pass