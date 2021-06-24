from core.liftetimeBenchmark.DecayGenerator import DecayGenerator
from core.analyze.lifetime import lifeTimeMeasurements, OneExpDecay, OneExpDecay_tail
import numpy as np
import matplotlib.pyplot as plt

class SingleBench:
    """
    Very simple and basic object to store the data of only one bench (the real paremeters of the decay and the fitted data).
    """
    def __init__(self, measurement, num, fitted_params_dict, aux_params_dict):
        # fit data and much more
        self.measurement = measurement
        self.num = num
        # real parameters
        self.fitted_params_dict = fitted_params_dict
        self.other_fitted_params_dict = {}
        # Non fitted auxiliaray parameters like noise, etc
        self.aux_params_dict = aux_params_dict
        # ?
        self.errors = {}
        self.score = 0



class LifeTimeBenchmark:
    def __init__(self):
        self.benchs = []
        self.results_dict = {}
        self.is_ini_guess = True
        self.is_fit = True
        self.measurement_dummy = lifeTimeMeasurements()

        self.is_working = False
        self.is_generating = False
        self.is_fitting = False
        self.is_ploting = False

        self.mean_score = None
        self.std_score = None
        self.hist_score = None
        self.x_hist_score = None

    def launch_benchmark(self):
        pass

    def set_model(self, model_name):
        """
        Only for parameters display in the GUI
        :return:
        """
        self.measurement_dummy.set_model(model_name)

    def getting_feedback_during_processing(self):

        if self.is_working:
            if self.is_generating:
                return "generating Decays"
            elif self.is_fitting:
                return "Fitting Decays"
            elif self.is_ploting:
                return "Ploting results"
        else:
            return 'Idle'
        pass

    def create_data(self, params):
        self.benchs = []
        self.is_working = True
        # Generate the decay
        self.set_work_state("generating")
        # decay_generator = DecayGenerator(nb_decay=nb_decay, time_nbins=4096, model="single_exp", params_dict=decay_params_dict_)
        # decay_generator = DecayGenerator(nb_decay=nb_decay, time_nbins=4096, model="single_exp_simple", params_dict=decay_params_dict_)
        decay_generator = DecayGenerator(params)
        decay_generator.generate_data(params)

        # Transfer the decays to "measurement" objects
        i = 0
        for decay in decay_generator.decays:
            measurement = lifeTimeMeasurements()
            measurement.data = decay.data
            measurement.time_axis = decay_generator.time_ns
            measurement.type = "lifetime"
            measurement.error_bar = np.sqrt(measurement.data + 1)
            measurement.set_model(params["model_name"])

            measurement.model.IRF = decay_generator.irf
            measurement.model.data = decay.data

            # MLE ou chi square
            measurement.set_params(params)
            self.benchs.append(SingleBench(measurement, i, decay.fitted_params_dict, decay.aux_params_dict))
            i += 1

        self.set_work_state("fitting")

        # Fit all the bench
        self.is_ini_guess = params["is_ini_guess"]
        self.is_ini_guess = params["is_fit"]

        for bench in self.benchs:

            if self.is_ini_guess:
                bench.measurement.model.guess(decay_generator.time_ns, bench.measurement.params)
            if self.is_fit:
                bench.measurement.fit()
            for param_key in bench.measurement.params:
                if param_key in bench.fitted_params_dict:
                    # if bench.measurement.params[param_key].user_data is not None:
                    #     if "dontGenerate" in bench.measurement.params[param_key].user_data:
                    #         continue

                    # FIXME how do you define error ?
                    # bench.errors[param_key] = (bench.measurement.params[param_key] - bench.fitted_params_dict[param_key])**2
                    bench.errors[param_key] = abs(bench.measurement.fit_results.params[param_key].value - bench.fitted_params_dict[
                        param_key])/bench.measurement.params[param_key].value*100
                else:
                    # param_key is in the parameters of the fit but not on the generated value
                    # e.g background that is a consequence of noise.
                    bench.other_fitted_params_dict[param_key] = bench.measurement.params[param_key].value


        scores = []
        for bench in self.benchs:
            for param_key in bench.errors:
                bench.score += bench.errors[param_key]
            scores.append(bench.score)

        self.mean_score = np.mean(scores)
        self.std_score = np.std(scores)
        self.median_score = np.median(scores)
        self.hist_score, self.x_hist_score = np.histogram(scores)


        # Gather the results and rearange them to display them later
        for key in decay_generator.decays[0].fitted_params_dict:
            ini_vals, errors, ini_guess = [], [], []
            for bench in self.benchs:
                ini_vals.append(bench.fitted_params_dict[key])
                errors.append(bench.errors[key])
                ini_guess.append(bench.measurement.model.ini_params[key])

            self.results_dict[key] = [ini_vals, errors, ini_guess]




        # print(self.results_dict)
        # Compress the data size

        # Exploit data

        # Give a score

        # Plots
        # Scatter or histogramm ?
        self.set_work_state("ploting")
        # self.plot_results(output="graph.png")

        self.is_working = False

    def set_work_state(self, state):
        self.is_generating = False
        self.is_fitting = False
        self.is_ploting = False
        if state == "generating":
            self.is_generating = True
        elif state == "fitting":
            self.is_fitting = True
        elif state == "ploting":
            self.is_ploting = True


    def saveState(self):
        # TODO with shelves
        pass

    def loadState(self):
        # TODO with shelves
        pass

    def plot_results(self, output=None):
        """Visualize the result of benchmark

        The output file will display the error value per parameter and contour
        plots for all combination of two parameters.

        Inspired by the `corner` package (https://github.com/dfm/corner.py).

        Parameters
        ----------
        result : :class:`~lmfit.minimizer.MinimizerResult`
            Contains the results from the :meth:`brute` method.

        output : str, optional
            Name of the output PDF file (default is 'None')
        """


        # self.top_level = tk.Toplevel(self.master_frame)
        # self.top_level.title("Explore chi2")

        npars_fitted = len(self.benchs[0].fitted_params_dict)
        npars_aux = len(self.benchs[0].aux_params_dict)
        #TODO dpi is function of the nb of graph
        self.fig, self.axes = plt.subplots(npars_fitted, npars_fitted, dpi=300)

        # TODO gerer les variables auxiliaire
        # self.fig, self.axes = plt.subplots(npars_fitted, npars_fitted + npars_aux, dpi=100)
        fig = self.fig
        axes = self.axes


        # self.canvas = FigureCanvasTkAgg(self.fig, master=self.top_level)
        # self.toolbar = NavigationToolbar2Tk(self.canvas, self.top_level)
        # self.canvas._tkcanvas.pack(side='top', fill='both', expand=1)

        varlabels_fitted = list(self.benchs[0].fitted_params_dict.keys())
        varlabels_aux = list(self.benchs[0].aux_params_dict.keys())

        for i, par1 in enumerate(varlabels_fitted):
            for j, par2 in enumerate(varlabels_fitted):

                # # parameter vs error in case of only one parameter
                # if npars_fitted == 1:
                #     key = varlabels_fitted[i]
                #     ax = axes[0]
                #     ax.plot(self.results_dict[key][0], self.results_dict[key][1], 'o', ms=3)
                #
                #     ax.set_ylabel(r'Error')
                #     ax.set_xlabel(key)

                # parameter vs errors profile on top
                # if i == j and j < npars_fitted - 1:

                # i horizontal, j vertical
                if i == j:
                    ax = axes[i, j]
                    # if i == 0:
                    #     # Pour ne pas dessiner un graphe ?
                    #     # axes[0, 0].axis('off')
                    #     ax.set_xlabel(par1)

                #     ax = axes[i, j]

                    # red_axis = tuple([a for a in range(npars_fitted) if a != i])
                    key = varlabels_fitted[i]
                    # index 0 of self.results_dict[key] is the generated "true" value, index 1 of  self.results_dict[key] is the relative error obtained via the fit in %
                    ax.plot(self.results_dict[key][0], self.results_dict[key][1], 'o', ms=3)
                    ax.set_ylabel(r'Error')
                    ax.yaxis.set_label_position("right")
                    ax.yaxis.set_ticks_position('right')
                    if j != npars_fitted - 1:
                        ax.set_xticks([])
                    else:
                        ax.set_xlabel(key)


                elif i < j:
                    axes[i, j].axis('off')


                # # parameter vs errors profile on the left
                # elif j == 0 and i > 0:
                #     ax = axes[i, j]
                #     # red_axis = tuple([a for a in range(npars_fitted) if a != i])
                #
                #     key = varlabels_fitted[i]
                #     ax.plot(self.results_dict[key][0], self.results_dict[key][1], 'o', ms=3)
                #     ax.invert_xaxis()
                #     ax.set_ylabel(key)
                #     if i != npars_fitted - 1:
                #         ax.set_xticks([])
                #     elif i == npars_fitted - 1:
                #         ax.set_xlabel(par1)

                # contour plots for all combinations of two parameters
                elif j < i:
                    ax = axes[i, j]
                    key_i = varlabels_fitted[i]
                    key_j = varlabels_fitted[j]

                    # red_axis = tuple([a for a in range(npars_fitted) if a != i and a != j])
                    # X, Y = np.meshgrid(self.results_dict[key_i][0],
                    #                    self.results_dict[key_j][0])

                    X, Y = np.meshgrid(self.results_dict[key_j][0],
                                       self.results_dict[key_i][0])

                    # lvls1 = np.linspace(result.brute_Jout.min(),
                    #                     np.median(result.brute_Jout) / 2.0, 7, dtype='int')
                    # lvls2 = np.linspace(np.median(result.brute_Jout) / 2.0,
                    #                     np.median(result.brute_Jout), 3, dtype='int')
                    # lvls = np.unique(np.concatenate((lvls1, lvls2)))
                    size = len(self.results_dict[key_i][0])
                    Z = np.zeros((size, size))

                    for k in range(size):
                        for m in range(size):
                            #TODO we sum the two relative errors ?
                            Z[k, m] = self.results_dict[key_i][1][k] + self.results_dict[key_j][1][m]

                    ax.contourf(X.T, Y.T, Z)
                    ax.set_yticks([])
                    if i != 0 and j==0:
                        ax.set_ylabel(key_i)
                    if i != npars_fitted - 1:
                        ax.set_xticks([])
                    elif i == npars_fitted - 1:
                        ax.set_xlabel(key_j)
                    # if j - i >= 2:
                    #     axes[i, j].axis('off')


        # TODO
        for i, par1 in enumerate(varlabels_aux):
            for j, par2 in enumerate(varlabels_fitted):
                pass
                # ax = axes[i, j]
                # key_i = varlabels_fitted[i]
                # key_j = varlabels_fitted[j]
                #
                # X, Y = np.meshgrid(self.results_dict[key_i][0],
                #                    self.results_dict[key_j][0])


        self.fig.set_tight_layout(True)
        self.fig.canvas.draw()

        if output is not None:
            self.fig.savefig(output)

        self.fig.show()


if __name__ == "__main__":

    life_time_benchmark = LifeTimeBenchmark()
    life_time_benchmark.create_data()

    print("OK")

