import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

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

        self.list_additional_param = []
        self.list_additional_param_sv = []

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

        self.figTex = plt.Figure(figsize=(13, 1.5), dpi=30, frameon=False)
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
        ttk.Label(self.param_frame, text='hold').grid(row=0, column=6)

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
            self.list_checkbox_int_variable_is_fixed.append(tk.IntVar())
            self.list_param_is_fixed.append(
                ttk.Checkbutton(self.param_frame, variable=self.list_checkbox_int_variable_is_fixed[i], state=tk.DISABLED))
            self.list_param_is_fixed[i].grid(row=1+i, column=6)

        self.changeModel(None)

        self.param_frame.pack(side="left", fill="both", expand=True)

    def enable_disable_ui(self, num_end):
        for i in range(num_end):
            self.list_entry_param_fit[i].state(['!disabled'])
            self.list_entry_param_fit_min[i].state(['!disabled'])
            self.list_entry_param_fit_max[i].state(['!disabled'])
            self.list_button_plus[i].config(state="normal")
            self.list_button_minus[i].config(state="normal")
            self.list_param_is_fixed[i].config(state="normal")

        for i in range(num_end, self.nb_param_fit):
            self.list_label_string_variable_fit[i].set("")
            self.list_entry_param_fit[i].state(['disabled'])
            self.list_entry_param_fit_min[i].state(['disabled'])
            self.list_entry_param_fit_max[i].state(['disabled'])
            self.list_button_plus[i].config(state=tk.DISABLED)
            self.list_button_minus[i].config(state=tk.DISABLED)
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
        pass

    def set_bckgnd_cursor(self):
        pass

    def ask_controller_eval_guess_fit(self, mode):
        model_name, params, xlim_min_fit, xlim_max_fit, params_min, params_max, params_hold = self.get_fit_params()
        self.controller.guess_eval_fit(mode, model_name=model_name,
                                       params=params, params_min=params_min, params_max=params_max, params_hold=params_hold, idx_start=xlim_min_fit, idx_end=xlim_max_fit,
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

    def get_fit_params(self):
        model_name = self.cb_model_sv.get()
        params = []
        params_min = []
        params_max = []
        params_hold = []

        for sv in self.list_entry_string_variable_fit:
            strValue = sv.get()
            if strValue == "":
                params.append(0)
            else:
                params.append(float(sv.get()))

        for sv in self.list_entry_string_variable_fit_min:
            strValue = sv.get()
            if strValue == "":
                # FIXME np.inf cause problems
                # params_min.append(-np.inf)
                params_min.append(-1E12)
            else:
                params_min.append(float(sv.get()))

        for sv in self.list_entry_string_variable_fit_max:
            strValue = sv.get()
            if strValue == "":
                # FIXME np.inf cause problems
                # params_max.append(np.inf)
                params_max.append(1E12)
            else:
                params_max.append(float(sv.get()))

        for iv in self.list_checkbox_int_variable_is_fixed:
            int_value = iv.get()
            if int_value == 0:
                params_hold.append(True)
            else:
                params_hold.append(False)

        if self.idx_lim_for_fit_min_sv.get() == "":
            xlim_min_fit = 0
        else:
            xlim_min_fit = float(self.idx_lim_for_fit_min_sv.get())

        if self.idx_lim_for_fit_max_sv.get() == "":
            xlim_max_fit = -1
        else:
            xlim_max_fit = float(self.idx_lim_for_fit_max_sv.get())

        return model_name, params, xlim_min_fit, xlim_max_fit, params_min, params_max, params_hold
