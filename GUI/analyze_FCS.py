import tkinter as tk
from tkinter import ttk
from .guiForFitOperation import guiForFitOperation


class guiForFitOperation_FCS(guiForFitOperation):

    def __init__(self, master_frame, controller, model_names, nb_param_fit):
        super().__init__(master_frame, controller, model_names, nb_param_fit, fitModeName="FCS")

    def changeModel(self, event):
        # Methode virtuelle, voir les classes dérivées.
        nbFitParam = self.nb_param_fit
        if self.cb_model_sv.get() == "1 Diff":
            self.list_label_string_variable_fit[0].set("G0")
            self.list_label_string_variable_fit[1].set("tdiff")
            self.list_label_string_variable_fit[2].set("r")
            self.list_label_string_variable_fit[3].set("cst")
            self.enable_disable_ui(4)
            self.setFitFormula(r"G_0 \frac{1}{1+t/tdiff}*\frac{1}{\sqrt{1+r*t/tdiff}} + cst")

        elif self.cb_model_sv.get() == "2 Diff":
            self.list_label_string_variable_fit[0].set("G0a")
            self.list_label_string_variable_fit[1].set("tdiffa")
            self.list_label_string_variable_fit[2].set("r")
            self.list_label_string_variable_fit[3].set("cst")
            self.list_label_string_variable_fit[4].set("G0b")
            self.list_label_string_variable_fit[5].set("tdiffb")
            self.enable_disable_ui(6)
            self.setFitFormula(r"G_0a \frac{1}{1+t/tdiffa}*\frac{1}{\sqrt{1+r*t/tdiff}} + G_0b \frac{1}{1+t/tdiffb}*\frac{1}{\sqrt{1+r*t/tdiff}} + cst")

        else:
            for i in range(nbFitParam) :
                self.list_label_string_variable_fit[i].set("")
                self.list_entry_param_fit[i].state(['disabled'])



class FCS_Analyze_gui():
    def __init__(self, masterFrame, controller, appearenceParam, measurement=None):
        self.masterFrame = masterFrame
        self.controller = controller
        self.appearenceParam = appearenceParam
        self.measurement = measurement


    def populate(self):
        self.frame_Correlate = tk.LabelFrame(self.masterFrame, text="Correlate",
                                                borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frame_fit = tk.LabelFrame(self.masterFrame, text="Fit",
                                          borderwidth=self.appearenceParam.frameLabelBorderWidth)

        self.frame_Correlate.pack(side="left", fill="both", expand=True)
        self.frame_fit.pack(side="left", fill="both", expand=True)

        label = ttk.Label(self.frame_Correlate, text='Max Correl Time (ms)')
        label.grid(row=0, column=0)

        self.maxCorrelTime_sv = tk.StringVar()
        e = ttk.Entry(self.frame_Correlate, textvariable=self.maxCorrelTime_sv, justify=tk.CENTER, width=7)
        e.grid(row=0, column=1)
        self.maxCorrelTime_sv.set('1000')

        label = ttk.Label(self.frame_Correlate, text='Max Correl Time (µs)')
        label.grid(row=0, column=2)
        self.startCorrelTime_sv = tk.StringVar()
        e = ttk.Entry(self.frame_Correlate, textvariable=self.startCorrelTime_sv, justify=tk.CENTER, width=7)
        e.grid(row=0, column=3)
        self.startCorrelTime_sv.set('10')


        label = ttk.Label(self.frame_Correlate, text='channel A')
        label.grid(row=0, column=4)
        self.num_c1_sv = tk.StringVar()
        e = ttk.Entry(self.frame_Correlate, textvariable=self.num_c1_sv, justify=tk.CENTER, width=7)
        e.grid(row=0, column=5)
        self.num_c1_sv.set('1')

        label = ttk.Label(self.frame_Correlate, text='channel B')
        label.grid(row=0, column=6)
        self.num_c2_sv = tk.StringVar()
        e = ttk.Entry(self.frame_Correlate, textvariable=self.num_c2_sv, justify=tk.CENTER, width=7)
        e.grid(row=0, column=7)
        self.num_c2_sv.set('1')

        ttk.Label(self.frame_Correlate, text='Algo').grid(row=0, column=8)

        self.algo_combo_box_sv = tk.StringVar()
        cb = ttk.Combobox(self.frame_Correlate, width=25, justify=tk.CENTER, textvariable=self.algo_combo_box_sv, values='')
        # cb.bind('<<ComboboxSelected>>', self.change_algo)
        cb['values'] = ('Whal', 'Laurence', 'F2Cor')
        self.algo_combo_box_sv.set('Whal')
        cb.grid(row=0, column=9)


        b = ttk.Button(self.frame_Correlate, text="AutoCorrelation", width=12, command=self.launchAutoCorrelationFCS)
        b.grid(row=1, column=0)

        b = ttk.Button(self.frame_Correlate, text="CrossCorrelation", width=12, command=self.launchCrossCorrelationFCS)
        b.grid(row=1, column=1)

        self.is_multiproc_iv = tk.IntVar()
        self.is_multiproc_check_box = ttk.Checkbutton(self.frame_Correlate, text="Multi Proc", variable=self.is_multiproc_iv)
        self.is_multiproc_check_box.grid(row=2, column=0)
        self.is_multiproc_iv.set(1)

        self.is_show_all_curve = tk.IntVar()
        self.is_show_all_curve_check_box = ttk.Checkbutton(self.frame_Correlate, text="Show all curves", variable=self.is_show_all_curve, command=self.show_all_curve)
        self.is_show_all_curve_check_box.grid(row=2, column=1)



        #FIT
        self.gui_for_fit_operation = guiForFitOperation_FCS(self.frame_fit, self.controller, ('1 Diff',  '2 Diff', 'Rotation'), nb_param_fit=8)
        self.gui_for_fit_operation.populate()


    def launchAutoCorrelationFCS(self):
        self.controller.calculate_measurement()
        self.controller.graph_measurement()

    def launchCrossCorrelationFCS(self):
        # TODO askSimpleDialog for Cross
        pass
        #self.controller.correlateFCS()

    def show_all_curve(self):
        pass

