import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from matplotlib.colors import LogNorm

class guiForFitOperation():

    #NB class variable shared by all instance.
    # idx_lim_for_fit_min = 0
    # idx_lim_for_fit_max = -1
    # idx_lim_for_fit_min_sv = tk.StringVar()
    # idx_lim_for_fit_max_sv = tk.StringVar()

    def __init__(self, master_frame, controller, model_names, nb_param_fit, fitModeName="", is_burst_analysis=False):
        self.master_frame = master_frame
        self.model_names = model_names
        self.controller = controller
        self.view = self.controller.view

        self.measurement = controller.current_measurement

        self.list_label_param_fit = []
        self.list_label_string_variable_fit = []
        self.list_entry_param_fit = []
        self.list_entry_string_variable_fit = []
        self.list_entry_param_fit_min = []
        self.list_entry_string_variable_fit_min = []
        self.list_entry_param_fit_max = []
        self.list_entry_string_variable_fit_max = []
        self.list_checkbox_int_variable_is_fixed = []
        self.list_param_is_fixed = []
        self.list_button_plus =[]
        self.list_button_minus = []
        self.list_brute_step_sv = []
        self.list_entry_brute_step = []

        self.is_burst_analysis = is_burst_analysis

        self.fit_mode_name = fitModeName

        self.nb_param_fit = nb_param_fit

    def populate(self):
        # cmd

        self.cmd_frame = ttk.Frame(self.master_frame)

        label = ttk.Label(self.cmd_frame, text='Model')
        label.grid(row=0, column=0)

        self.cb_model_sv = tk.StringVar()
        cb = ttk.Combobox(self.cmd_frame, width=15, justify=tk.CENTER, textvariable=self.cb_model_sv,
                          values='', state='readonly')
        cb.bind('<<ComboboxSelected>>', self.changeModel)
        cb['values'] = self.model_names
        self.cb_model_sv.set(self.model_names[0])
        cb.set(self.model_names[0])
        cb.grid(row=0, column=1)

        ttk.Button(self.cmd_frame, text="IniGuess", command=self.ini_guess_fit).grid(row=1, column=0)
        ttk.Button(self.cmd_frame, text="Eval", command=self.eval_fit).grid(row=1, column=1)
        ttk.Button(self.cmd_frame, text="Fit", command=self.fit).grid(row=1, column=2)

        # Additional Params that are non fitted but needs to be known
        # i = 0
        # for param in self.list_additional_param:
        #     ttk.Label(self.cmd_frame, text='list_additional_param').grid(row=2, column=i)
        #
        # self.list_additional_param_sv.append(tk.StringVar())
        # ttk.Entry(self.cmd_frame, textvariable=self.list_additional_param_sv[i], justify=tk.CENTER, width=8).grid(row=2, column=i+1)
        # ttk.Button(self.cmd_frame, text="Set with cursor", command=self.set_bckgnd_cursor).grid(row=2, column=2)


        # Formula
        self.formulaFrame = tk.Frame(master=self.cmd_frame)
        self.formulaFrame.grid(row=3, column=0, columnspan=3)

        self.figTex = plt.Figure(figsize=(13, 2), dpi=28, frameon=False)
        self.axTex = self.figTex.add_axes([0, 0, 1, 1])

        self.axTex.axis('off')

        # self.axTex.get_xaxis().set_visible(False)
        # self.axTex.get_yaxis().set_visible(False)

        self.canvasTk = FigureCanvasTkAgg(self.figTex, master=self.formulaFrame)
        self.canvasTk.get_tk_widget().pack(side='top', fill='both', expand=1)

        # fit boundaries
        ttk.Label(self.cmd_frame, text='fit boundaries (x1, x2)').grid(row=4, column=0)


        self.idx_lim_for_fit_min_sv = tk.StringVar()
        ttk.Entry(self.cmd_frame, textvariable=self.idx_lim_for_fit_min_sv, justify=tk.CENTER, width=12).grid(row=4, column=1)

        self.idx_lim_for_fit_max_sv = tk.StringVar()
        ttk.Entry(self.cmd_frame, textvariable=self.idx_lim_for_fit_max_sv, justify=tk.CENTER, width=12).grid(row=4, column=2)


        # Explore chi square
        ttk.Button(self.cmd_frame, text="Explore χ²", command=self.explore_chi_square).grid(row=5, column=0)

        self.is_take_account_error_bar = tk.IntVar(value=1)
        ttk.Checkbutton(self.cmd_frame, text="Use error bars", variable=self.is_take_account_error_bar, command=self.use_error_bar).grid(row=6, column=0)

        self.cmd_frame.pack(side="left", fill="both", expand=True)

        # Parameters
        self.param_frame = ttk.Frame(self.master_frame)

        # column header
        ttk.Label(self.param_frame, text='').grid(row=0, column=0)
        ttk.Label(self.param_frame, text='value').grid(row=0, column=1)
        ttk.Label(self.param_frame, text='').grid(row=0, column=2)
        ttk.Label(self.param_frame, text='').grid(row=0, column=3)
        ttk.Label(self.param_frame, text='min').grid(row=0, column=4)
        ttk.Label(self.param_frame, text='max').grid(row=0, column=5)
        ttk.Label(self.param_frame, text='b step').grid(row=0, column=6)
        ttk.Label(self.param_frame, text='hold').grid(row=0, column=7)

        for i in range(self.nb_param_fit):
            # TODO validate that entries are numeric value (cf method in GUI_root)

            # param name (e.g. tau1)
            self.list_label_string_variable_fit.append(tk.StringVar())
            txt = "p" + str(i)
            self.list_label_param_fit.append(
                ttk.Label(self.param_frame, text=txt, textvariable=self.list_label_string_variable_fit[i]))
            self.list_label_param_fit[i].grid(row=1+i, column=0)

            # value
            self.list_entry_string_variable_fit.append(tk.StringVar())
            self.list_entry_param_fit.append(
                ttk.Entry(self.param_frame, textvariable=self.list_entry_string_variable_fit[i], justify=tk.CENTER,
                          width=7, state=tk.DISABLED))
            self.list_entry_param_fit[i].grid(row=1+i, column=1)

            # + button
            self.list_button_plus.append(tk.Button(master=self.param_frame, text='+', command=lambda: self.value_plus(self.list_entry_string_variable_fit[i])))
            self.list_button_plus[i].grid(row=1+i, column=2)

            # - button
            self.list_button_minus.append(tk.Button(master=self.param_frame, text='-', command=lambda: self.value_minus(self.list_entry_string_variable_fit[i])))
            self.list_button_minus[i].grid(row=1+i, column=3)

            # min fit value constraint
            self.list_entry_string_variable_fit_min.append(tk.StringVar())
            self.list_entry_param_fit_min.append(
                ttk.Entry(self.param_frame, textvariable=self.list_entry_string_variable_fit_min[i], justify=tk.CENTER,
                          width=7, state=tk.DISABLED))
            self.list_entry_param_fit_min[i].grid(row=1+i, column=4)

            # max fit value constraint
            self.list_entry_string_variable_fit_max.append(tk.StringVar())
            self.list_entry_param_fit_max.append(
                ttk.Entry(self.param_frame, textvariable=self.list_entry_string_variable_fit_max[i], justify=tk.CENTER,
                          width=7, state=tk.DISABLED))
            self.list_entry_param_fit_max[i].grid(row=1+i, column=5)

            # hold check button
            self.list_brute_step_sv.append(tk.StringVar())
            self.list_entry_brute_step.append(ttk.Entry(self.param_frame, textvariable=self.list_brute_step_sv[i], justify=tk.CENTER,
                          width=7, state=tk.DISABLED))
            self.list_entry_brute_step[i].grid(row=1+i, column=6)


            # hold check button
            self.list_checkbox_int_variable_is_fixed.append(tk.IntVar())
            self.list_param_is_fixed.append(
                ttk.Checkbutton(self.param_frame, variable=self.list_checkbox_int_variable_is_fixed[i], state=tk.DISABLED))
            self.list_param_is_fixed[i].grid(row=1+i, column=7)

        self.changeModel(None)

        self.param_frame.pack(side="left", fill="both", expand=True)

    def enable_disable_ui(self, num_end):
        for i in range(num_end):
            self.list_entry_param_fit[i].state(['!disabled'])
            self.list_entry_param_fit_min[i].state(['!disabled'])
            self.list_entry_param_fit_max[i].state(['!disabled'])
            self.list_button_plus[i].config(state="normal")
            self.list_button_minus[i].config(state="normal")
            self.list_entry_brute_step[i].config(state="normal")
            self.list_param_is_fixed[i].config(state="normal")

        for i in range(num_end, self.nb_param_fit):
            self.list_label_string_variable_fit[i].set("")
            self.list_entry_param_fit[i].state(['disabled'])
            self.list_entry_param_fit_min[i].state(['disabled'])
            self.list_entry_param_fit_max[i].state(['disabled'])
            self.list_button_plus[i].config(state=tk.DISABLED)
            self.list_button_minus[i].config(state=tk.DISABLED)
            self.list_entry_brute_step[i].state(['disabled'])
            self.list_param_is_fixed[i].config(state=tk.DISABLED)

    def setFitFormula(self, formula, fontsize=40):
        formula = "$" + formula + "$"

        self.axTex.clear()
        self.axTex.text(0, 0.2, formula, fontsize=fontsize)
        self.canvasTk.draw()

    def changeModel(self, event):
        # Methode virtuelle, voir les classes dérivées.
        raise NotImplementedError()

    def explore_chi_square(self):
        params = self.get_fit_params()
        result = self.measurement.explore_chi2_surf(params)
        self.plot_results_brute(result)

    def set_bckgnd_cursor(self):
        pass

    def use_error_bar(self):
        is_use = self.is_take_account_error_bar.get()
        self.controller.set_use_error_bar(is_use)
        return is_use

    def ask_controller_eval_guess_fit(self, mode):
        params = self.get_fit_params()
        self.controller.guess_eval_fit(mode, params,
                                       is_burst_analysis=self.is_burst_analysis)

    def eval_fit(self):
        self.ask_controller_eval_guess_fit("eval")

    def ini_guess_fit(self):
        self.ask_controller_eval_guess_fit("guess")

    def fit(self):
        self.ask_controller_eval_guess_fit("fit")

    def value_plus(self, tk_string_var):
        # FIXME
        #TODO try execpt
        value = float(tk_string_var.get())
        value *= 1.1
        tk_string_var.set(str(value))

    def value_minus(self, tk_string_var):
        #FIXME
        tk_string_var.set(str(float(tk_string_var.get())*0.9))


    def setParamsFromFit(self, params):
        i = 0
        for paramName, param in params.items():
            self.list_entry_string_variable_fit[i].set(str(param.value))
            i += 1

    def get_lim_for_fit(self):
        if self.idx_lim_for_fit_min_sv.get() == "":
            xlim_min_fit = 0
        else:
            xlim_min_fit = float(self.idx_lim_for_fit_min_sv.get())

        if self.idx_lim_for_fit_max_sv.get() == "":
            xlim_max_fit = -1
        else:
            xlim_max_fit = float(self.idx_lim_for_fit_max_sv.get())
        return (xlim_min_fit, xlim_max_fit)

    def get_params_values(self):
        params_value = []
        for sv in self.list_entry_string_variable_fit:
            strValue = sv.get()
            if strValue == "":
                params_value.append(0)
            else:
                params_value.append(float(sv.get()))
        return params_value

    def get_params_min(self):
        params_min = []
        for sv in self.list_entry_string_variable_fit_min:
            strValue = sv.get()
            if strValue == "":
                # FIXME np.inf cause problems
                # params_min.append(-np.inf)
                params_min.append(-1E12)
            else:
                params_min.append(float(sv.get()))
        return params_min

    def get_params_max(self):
        params_max = []
        for sv in self.list_entry_string_variable_fit_max:
            strValue = sv.get()
            if strValue == "":
                # FIXME np.inf cause problems
                # params_max.append(np.inf)
                params_max.append(1E12)
            else:
                params_max.append(float(sv.get()))
        return params_max

    def get_params_hold(self):
        params_hold = []
        for iv in self.list_checkbox_int_variable_is_fixed:
            int_value = iv.get()
            if int_value == 0:
                params_hold.append(True)
            else:
                params_hold.append(False)
        return params_hold

    def get_brute_step(self):
        params_brute_step = []
        for sv in self.list_brute_step_sv:
            strValue = sv.get()
            if strValue == "":
                # FIXME np.inf cause problems
                # params_max.append(np.inf)
                params_brute_step.append(None)
            else:
                params_brute_step.append(float(sv.get()))
        return params_brute_step

    def get_fit_params(self):
        params = {}
        params["model_name"] = self.cb_model_sv.get()
        params["val"] = self.get_params_values()
        params["min"] = self.get_params_min()
        params["max"] = self.get_params_max()
        params["hold"] = self.get_params_hold()
        params["lim_fit"] = self.get_lim_for_fit()
        params["brute_step"] = self.get_brute_step()
        params["use_error_bar"] = self.use_error_bar()
        return params



    def plot_results_brute(self, result, best_vals=True, varlabels=None,
                           output=None):
        """Visualize the result of the brute force grid search.

        The output file will display the chi-square value per parameter and contour
        plots for all combination of two parameters.

        Inspired by the `corner` package (https://github.com/dfm/corner.py).

        Parameters
        ----------
        result : :class:`~lmfit.minimizer.MinimizerResult`
            Contains the results from the :meth:`brute` method.

        best_vals : bool, optional
            Whether to show the best values from the grid search (default is True).

        varlabels : list, optional
            If None (default), use `result.var_names` as axis labels, otherwise
            use the names specified in `varlabels`.

        output : str, optional
            Name of the output PDF file (default is 'None')
        """


        self.top_level = tk.Toplevel(self.master_frame)
        self.top_level.title("Explore chi2")

        npars = len(result.var_names)
        #TODO dpi is funciton of the nb of graph
        self.fig, self.axes = plt.subplots(npars, npars, dpi=100)
        fig = self.fig
        axes = self.axes



        self.canvas = FigureCanvasTkAgg(self.fig, master=self.top_level)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.top_level)
        self.canvas._tkcanvas.pack(side='top', fill='both', expand=1)

        if not varlabels:
            varlabels = result.var_names
        if best_vals and isinstance(best_vals, bool):
            best_vals = result.params

        for i, par1 in enumerate(result.var_names):
            for j, par2 in enumerate(result.var_names):

                # parameter vs chi2 in case of only one parameter
                if npars == 1:
                    axes.plot(result.brute_grid, result.brute_Jout, 'o', ms=3)
                    axes.set_ylabel(r'$\chi^{2}$')
                    axes.set_xlabel(varlabels[i])
                    if best_vals:
                        axes.axvline(best_vals[par1].value, ls='dashed', color='r')

                # parameter vs chi2 profile on top
                elif i == j and j < npars - 1:
                    if i == 0:
                        axes[0, 0].axis('off')
                    ax = axes[i, j + 1]
                    red_axis = tuple([a for a in range(npars) if a != i])
                    ax.plot(np.unique(result.brute_grid[i]),
                            np.minimum.reduce(result.brute_Jout, axis=red_axis),
                            'o', ms=3)
                    ax.set_ylabel(r'$\chi^{2}$')
                    ax.yaxis.set_label_position("right")
                    ax.yaxis.set_ticks_position('right')
                    ax.set_xticks([])
                    if best_vals:
                        ax.axvline(best_vals[par1].value, ls='dashed', color='r')

                # parameter vs chi2 profile on the left
                elif j == 0 and i > 0:
                    ax = axes[i, j]
                    red_axis = tuple([a for a in range(npars) if a != i])
                    ax.plot(np.minimum.reduce(result.brute_Jout, axis=red_axis),
                            np.unique(result.brute_grid[i]), 'o', ms=3)
                    ax.invert_xaxis()
                    ax.set_ylabel(varlabels[i])
                    if i != npars - 1:
                        ax.set_xticks([])
                    elif i == npars - 1:
                        ax.set_xlabel(r'$\chi^{2}$')
                    if best_vals:
                        ax.axhline(best_vals[par1].value, ls='dashed', color='r')

                # contour plots for all combinations of two parameters
                elif j > i:
                    ax = axes[j, i + 1]
                    red_axis = tuple([a for a in range(npars) if a != i and a != j])
                    X, Y = np.meshgrid(np.unique(result.brute_grid[i]),
                                       np.unique(result.brute_grid[j]))
                    lvls1 = np.linspace(result.brute_Jout.min(),
                                        np.median(result.brute_Jout) / 2.0, 7, dtype='int')
                    lvls2 = np.linspace(np.median(result.brute_Jout) / 2.0,
                                        np.median(result.brute_Jout), 3, dtype='int')
                    lvls = np.unique(np.concatenate((lvls1, lvls2)))
                    ax.contourf(X.T, Y.T, np.minimum.reduce(result.brute_Jout, axis=red_axis),
                                lvls, norm=LogNorm())
                    ax.set_yticks([])
                    if best_vals:
                        ax.axvline(best_vals[par1].value, ls='dashed', color='r')
                        ax.axhline(best_vals[par2].value, ls='dashed', color='r')
                        ax.plot(best_vals[par1].value, best_vals[par2].value, 'rs', ms=3)
                    if j != npars - 1:
                        ax.set_xticks([])
                    elif j == npars - 1:
                        ax.set_xlabel(varlabels[i])
                    if j - i >= 2:
                        axes[i, j].axis('off')

        self.fig.set_tight_layout(True)
        self.fig.canvas.draw()

        if output is not None:
            self.fig.savefig(output)

