import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog
from .guiForFitOperation import guiForFitOperation
from core import Experiment

from tkinter import filedialog, messagebox, simpledialog

from .Dialog import fitIRFDialog


class guiForFitOperation_Lifetime(guiForFitOperation):

    def __init__(self, master_frame, controller, model_names, nb_param_fit, is_burst_analysis=False):
        super().__init__(master_frame, controller, model_names, nb_param_fit, fitModeName="lifetime", is_burst_analysis=is_burst_analysis)

    def changeModel(self, event):
        # Methode virtuelle, voir les classes dérivées.
        nbFitParam = self.nb_param_fit

        if self.cb_model_sv.get() == "One Decay IRF":
            self.list_label_string_variable_fit[0].set("tau")
            self.list_label_string_variable_fit[1].set("IRF shift")
            self.list_label_string_variable_fit[2].set("bckgnd")
            self.enable_disable_ui(3)
            self.setFitFormula(r"f(t; \tau, shift) = IRF(shift) \times (e^{-t/\tau}) + bckgnd")

            self.list_entry_string_variable_fit_min[0].set("0")
            # self.list_entry_string_variable_fit_min[1].set("0")
            self.list_entry_string_variable_fit_min[2].set("0")

        elif self.cb_model_sv.get() == "One Decay Tail":
            self.list_label_string_variable_fit[0].set("t0")
            self.list_label_string_variable_fit[1].set("tau")
            self.enable_disable_ui(2)
            self.setFitFormula(r"cst + e^{-(t-t_0)/\tau}")

        elif self.cb_model_sv.get() == "Two Decays IRF":
            self.list_label_string_variable_fit[0].set("tau1")
            self.list_label_string_variable_fit[1].set("a1")
            self.list_label_string_variable_fit[2].set("tau2")
            self.list_label_string_variable_fit[3].set("IRF shift")
            self.list_label_string_variable_fit[4].set("bckgnd")
            self.enable_disable_ui(5)
            self.setFitFormula(r"IRF(shift) \times  (a1 . e^{-t/\tau_1} + (1-a_1) \times e^{-t/\tau_2}) + bckgnd")

            self.list_entry_string_variable_fit_min[0].set("0")
            self.list_entry_string_variable_fit_min[1].set("0")
            self.list_entry_string_variable_fit_min[2].set("0")
            self.list_entry_string_variable_fit_min[4].set("0")

        elif self.cb_model_sv.get() == "IRF":
            self.list_label_string_variable_fit[0].set("mu")
            self.list_label_string_variable_fit[1].set("sigma")
            self.list_label_string_variable_fit[2].set("tau")
            self.list_label_string_variable_fit[3].set("shift")
            self.list_label_string_variable_fit[4].set("cst")
            self.enable_disable_ui(5)
            self.setFitFormula(r"\frac{\lambda}{2} e^{\frac{\lambda}{2} (2 \mu + \lambda \sigma^2 - 2 x)} \operatorname{erfc} (\frac{\mu + \lambda \sigma^2 - x}{ \sqrt{2} \sigma})")

        else:
            for i in range(nbFitParam) :
                self.list_label_string_variable_fit[i].set("")
                self.enable_disable_ui(0)



class lifeTimeAnalyze_gui():
    def __init__(self, masterFrame, controller, appearenceParam, measurement, is_burst_analysis=False):
        self.masterFrame = masterFrame
        self.controller = controller
        self.appearenceParam = appearenceParam
        self.measurement = measurement
        self.is_keep_selection_for_filter = True
        self.is_burst_analysis = is_burst_analysis
        self.type = "lifetime"
        # self.is_graph_x_ns = True

    def populate(self):
        self.frameMicro_graph = tk.LabelFrame(self.masterFrame, text="Graph",
                                         borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frameMicro_filter = tk.LabelFrame(self.masterFrame, text="Filter",
                                         borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frameMicro_IR_signal = tk.LabelFrame(self.masterFrame, text="Static param fit",
                                         borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frameMicro_IR = tk.LabelFrame(self.frameMicro_IR_signal, text="IR",
                                         borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frameMicro_Signal = tk.LabelFrame(self.frameMicro_IR_signal, text="Signal",
                                         borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frameMicro_fit = tk.LabelFrame(self.masterFrame, text="Fit",
                                         borderwidth=self.appearenceParam.frameLabelBorderWidth)

        self.frameMicro_graph.pack(side="left", fill="both", expand=True)
        self.frameMicro_filter.pack(side="left", fill="both", expand=True)
        self.frameMicro_IR_signal.pack(side="left", fill="both", expand=True)
        self.frameMicro_IR.pack(side="top", fill="both", expand=True)
        self.frameMicro_Signal.pack(side="top", fill="both", expand=True)
        self.frameMicro_fit.pack(side="left", fill="both", expand=True)


        #Graph

        b = ttk.Button(self.frameMicro_graph, text="Graph", width=6, command=self.launch_micro_time_histo)
        b.grid(row=0, column=0)

        self.is_semi_log = tk.IntVar()
        ttk.Checkbutton(self.frameMicro_graph, text="SemiLog ?", variable=self.is_semi_log).grid(row=0, column=1)

        # self.toggle_button_graph = ttk.Button(self.frameMicro_graph, text="ns", width=15, command=self.toggle_graph_x)
        # self.toggle_button_graph.grid(row=0, column=0)

        label = ttk.Label(self.frameMicro_graph, text='channel :')
        label.grid(row=1, column=0)
        self.num_channel_sv = tk.StringVar()
        e = ttk.Entry(self.frameMicro_graph, textvariable=self.num_channel_sv, justify=tk.CENTER, width=7)
        e.grid(row=1, column=1)
        self.num_channel_sv.set('0')

        self.is_overlay_on_time_zoom = tk.IntVar()
        ttk.Checkbutton(self.frameMicro_graph, text="Overlay on time zoom", variable=self.is_overlay_on_time_zoom, command=self.is_draw_overlay).grid(row=2, column=0, columnspan=2)

        #Filter
        self.toggle_button = ttk.Button(self.frameMicro_filter, text="Keep selection", width=15, command=self.toggle_filter_mode)
        self.toggle_button.grid(row=0, column=0)

        ttk.Button(self.frameMicro_filter, text="Filter", width=6, command=self.microtime_filter).grid(row=1, column=0)



        #IR
        self.isDraw_IR = tk.IntVar()
        self.is_draw_IR_check_box = ttk.Checkbutton(self.frameMicro_IR, text="Use IR ?", variable=self.isDraw_IR, command=self.is_use_IR)
        self.is_draw_IR_check_box.grid(row=0, column=0)

        ttk.Button(self.frameMicro_IR, text="Open file",  command=self.openIR_file).grid(row=0, column=1)


        ttk.Button(self.frameMicro_IR, text="Generate",  command=self.generateIR_file).grid(row=0, column=2)


        tk.Label(self.frameMicro_IR, text="IR Name :").grid(row=1, column=0)

        self.ir_name_sv = tk.StringVar()
        ttk.Entry(self.frameMicro_IR, textvariable=self.ir_name_sv, justify=tk.CENTER, width=50).grid(row=1, column=1, columnspan=2)
        self.ir_name_sv.set("None loaded yet")

        tk.Label(self.frameMicro_IR, text="Beginning % :").grid(row=2, column=0)

        self.ir_start_sv = tk.StringVar()
        w = ttk.Entry(self.frameMicro_IR, textvariable=self.ir_start_sv, justify=tk.CENTER, width=15)
        w.bind('<Return>', self.change_IR)
        self.ir_start_sv.set(0)
        w.grid(row=2, column=1)

        tk.Label(self.frameMicro_IR, text="End % :").grid(row=3, column=0)

        self.ir_end_sv = tk.StringVar()
        w = ttk.Entry(self.frameMicro_IR, textvariable=self.ir_end_sv, justify=tk.CENTER, width=15)
        w.bind('<Return>', self.change_IR)
        self.ir_end_sv.set(100)
        w.grid(row=3, column=1)

        tk.Label(self.frameMicro_IR, text="Shift (µchannel) :").grid(row=4, column=0)
        self.shiftIR_amount_sv = tk.StringVar()
        w = ttk.Entry(self.frameMicro_IR, textvariable=self.shiftIR_amount_sv, justify=tk.CENTER, width=15)
        w.bind('<Return>', self.change_IR)
        w.grid(row=4, column=1)
        self.shiftIR_amount_sv.set(0)

        tk.Label(self.frameMicro_IR, text="Background").grid(row=5, column=0)
        self.bckg_IR_sv = tk.StringVar()
        w = ttk.Entry(self.frameMicro_IR, textvariable=self.bckg_IR_sv, justify=tk.CENTER, width=15)
        w.bind('<Return>', self.change_IR)
        w.grid(row=5, column=1)
        self.bckg_IR_sv.set(0)

        ttk.Button(self.frameMicro_IR, text="Auto",  command=self.autoShiftIR).grid(row=3, column=2)

        # b = ttk.Button(self.frameMicro_IR, text="Fit IR",  command=self.fit_IR).grid(row=4, column=0)

        #Signal
        # tk.Label(self.frameMicro_Signal, text="Signal bckgnd :").grid(row=0, column=0)
        # self.signal_bckgnd_sv = tk.StringVar()
        # self.signal_bckgnd_sv.set("0")
        # ttk.Entry(self.frameMicro_Signal, textvariable=self.signal_bckgnd_sv, justify=tk.CENTER, width=15).grid(row=0, column=1)
        # ttk.Button(self.frameMicro_Signal, text="use cursor", command=self.select_bckgnd_w_cursor).grid(row=0, column=2)

        #FIT
        # model_names -> cf core/lifetime.py
        self.gui_for_fit_operation = guiForFitOperation_Lifetime(self.frameMicro_fit, self.controller,
                                                                 model_names=('One Decay IRF', 'One Decay Tail', 'Two Decays IRF', 'Two Decays Tail', 'IRF'), nb_param_fit=8,
                                                                 is_burst_analysis=self.is_burst_analysis)
        self.gui_for_fit_operation.populate()

    def openIR_file(self):
        file_path = filedialog.askopenfilename(title="Open IR File", initialdir=self.controller.view.saveDir)
        if file_path == None or file_path == '':
            return None
        ir_exp = Experiment.Experiment("file", [file_path])
        measurement_ir = ir_exp.create_measurement(0, 0, -1,"lifetime", "", "")
        ir_exp.calculate_life_time(measurement_ir)
        self.ir_name_sv.set(ir_exp.file_name)
        self.controller.current_measurement.set_IR(ir_exp.file_name, measurement_ir.data, measurement_ir.time_axis)
        self.controller.update_analyze()

    def is_use_IR(self):
        self.change_IR(None)

    def select_bckgnd_w_cursor(self):
        pass

    def is_draw_overlay(self):
        self.controller.update_navigation()

    def fit_IR(self):
        d = fitIRFDialog(self.masterFrame, title="Initial fit parameters")
        if d.result is not None:
            iniParams = d.result
            self.controller.fit_IR(iniParams)



    def generateIR_file(self):
        #TODO Custom Dialog !
        answer = simpledialog.askfloat("Main IR width in picoSeconds", initialvalue=50.0, minvalue = 1.0)
        if answer != None:
            mainWidth = answer
            answer2 = simpledialog.askfloat("Secondary IR width in picoSeconds", initialvalue=100.0, minvalue=1.0)
            if answer2 != None:
                secondaryWidth = answer2
                answer3 = simpledialog.askfloat("Secondary IR relative intensity (between 0 and 1", initialvalue=0.05, minvalue=1.0, maxvalue=1.0)
                if answer3 != None:
                    secondaryAmplitude = answer3
                    answer4 = simpledialog.askfloat("Secondary IR relative intensity (between 0 and 1",
                                                    initialvalue=0.05, minvalue=1.0, maxvalue=1.0)
                    if answer4 != None:
                        timeOffset = answer4
                        self.controller.generate_artificial_IR(mainWidth, secondaryWidth, secondaryAmplitude, timeOffset)
                        self.ir_name_sv.set("artificial " + str(mainWidth) + " " +  str(secondaryWidth))


    def autoShiftIR(self):
        pass

    def change_IR(self, e):
        start = float(self.ir_start_sv.get())
        if start < 0:
            start = 0
        elif start > 100:
            start = 100
        self.controller.current_measurement.IR_start = start

        end = float(self.ir_end_sv.get())
        if end < 0:
            end = 0
        elif end > 100:
            end = 100
        self.controller.current_measurement.IR_end = end

        self.controller.current_measurement.IR_shift = float(self.shiftIR_amount_sv.get())
        self.controller.current_measurement.set_use_IR(self.isDraw_IR.get())

        self.controller.current_measurement.IR_bckg = int(self.bckg_IR_sv.get())

        if self.controller.current_measurement.process_IR() == "OK":
            self.controller.update_analyze()

    def launch_micro_time_histo(self):
        #FIXME
        self.controller.calculate_measurement()
        self.controller.graph_measurement()
        # self.mainGUI.controller.drawMicroTimeHisto(self.mainGUI.currentChannel, self.mainGUI.currentTimeWindow[0], self.mainGUI.currentTimeWindow[1])
        # print("launchMicroTimeHisto")

    def toggle_filter_mode(self):
        if self.is_keep_selection_for_filter:
            self.toggle_button.config(text='Filter selection')
            self.is_keep_selection_for_filter = False
        else:
            self.toggle_button.config(text='Keep selection')
            self.is_keep_selection_for_filter = True

    # def toggle_graph_x(self):
    #     if self.is_graph_x_ns:
    #         self.toggle_button_graph.config(text='num channel')
    #         self.is_graph_x_ns = False
    #     else:
    #         self.toggle_button.config(text='ns')
    #         self.is_graph_x_ns = True

    def microtime_filter(self):
        num_channel = int(self.num_channel_sv.get())
        self.controller.microtime_filter(num_channel, self.is_keep_selection_for_filter)

