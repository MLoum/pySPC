import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog
from .guiForFitOperation import guiForFitOperation
from core import Experiment

from tkinter import filedialog, messagebox, simpledialog

class guiForFitOperation_Lifetime(guiForFitOperation):

    def __init__(self, masterFrame, controller, modelNames, nbParamFit):
        super().__init__( masterFrame, controller, modelNames, nbParamFit, fitModeName="lifetime")

    def changeModel(self, event):
        # Methode virtuelle, voir les classes dérivées.
        nbFitParam = self.nbParamFit
        if self.comboBoxStringVar.get() == "One Decay":
            self.listLabelStringVariableFit[0].set("t0")
            self.listLabelStringVariableFit[1].set("amp")
            self.listLabelStringVariableFit[2].set("tau")
            self.listLabelStringVariableFit[3].set("cst")

            for i in range(4):
                self.listEntryParamFit[i].state(['!disabled'])

            for i in range(4, nbFitParam) :
                self.listLabelStringVariableFit[i].set("")
                self.listEntryParamFit[i].state(['disabled'])

            self.setFitFormula(r"cst + amp \times e^{-(t+t_0)/\tau}")

        elif self.comboBoxStringVar.get() == "Two Decays":
            self.listLabelStringVariableFit[0].set("t0")
            self.listLabelStringVariableFit[1].set("amp")
            self.listLabelStringVariableFit[2].set("tau")
            self.listLabelStringVariableFit[3].set("cst")
            self.listLabelStringVariableFit[4].set("amp2")
            self.listLabelStringVariableFit[5].set("tau2")

            for i in range(6):
                self.listEntryParamFit[i].state(['!disabled'])

            for i in range(6, nbFitParam) :
                self.listLabelStringVariableFit[i].set("")
                self.listEntryParamFit[i].state(['disabled'])

            self.setFitFormula(r"cst + amp \times e^{-(t-t_0)/\tau} + amp_2 \times e^{-(t+t_0)/\tau_2}")
        else :
            for i in range(nbFitParam) :
                self.listLabelStringVariableFit[i].set("")
                self.listEntryParamFit[i].state(['disabled'])

class lifeTimeAnalyze_gui():
    def __init__(self, masterFrame, controller, appearenceParam, measurement):
        self.masterFrame = masterFrame
        self.controller = controller
        self.appearenceParam = appearenceParam
        self.measurement = measurement
        self.is_keep_selection_for_filter = True
        # self.is_graph_x_ns = True

    def populate(self):
        self.frameMicro_graph = tk.LabelFrame(self.masterFrame, text="Graph",
                                         borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frameMicro_filter = tk.LabelFrame(self.masterFrame, text="Filter",
                                         borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frameMicro_IR = tk.LabelFrame(self.masterFrame, text="IR",
                                         borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frameMicro_fit = tk.LabelFrame(self.masterFrame, text="Fit",
                                         borderwidth=self.appearenceParam.frameLabelBorderWidth)

        self.frameMicro_graph.pack(side="left", fill="both", expand=True)
        self.frameMicro_filter.pack(side="left", fill="both", expand=True)
        self.frameMicro_IR.pack(side="left", fill="both", expand=True)
        self.frameMicro_fit.pack(side="left", fill="both", expand=True)


        #Graph

        b = ttk.Button(self.frameMicro_graph, text="Graph", width=6, command=self.launch_micro_time_histo)
        b.grid(row=0, column=0)

        self.isSemiLog = tk.IntVar()
        self.is_draw_IR_check_box = ttk.Checkbutton(self.frameMicro_graph, text="SemiLog ?", variable=self.isSemiLog)
        self.is_draw_IR_check_box.grid(row=0, column=1)

        # self.toggle_button_graph = ttk.Button(self.frameMicro_graph, text="ns", width=15, command=self.toggle_graph_x)
        # self.toggle_button_graph.grid(row=0, column=0)

        label = ttk.Label(self.frameMicro_graph, text='channel :')
        label.grid(row=1, column=0)
        self.num_channel_sv = tk.StringVar()
        e = ttk.Entry(self.frameMicro_graph, textvariable=self.num_channel_sv, justify=tk.CENTER, width=7)
        e.grid(row=1, column=1)
        self.num_channel_sv.set('0')

        #Filter

        self.toggle_button = ttk.Button(self.frameMicro_filter, text="Keep selection", width=15, command=self.toggle_filter_mode)
        self.toggle_button.grid(row=0, column=0)

        b = ttk.Button(self.frameMicro_filter, text="Filter", width=6, command=self.microtime_filter)
        b.grid(row=1, column=0)


        #IR
        self.isDraw_IR = tk.IntVar()
        self.is_draw_IR_check_box = ttk.Checkbutton(self.frameMicro_IR, text="Use IR ?", variable=self.isDraw_IR, command=self.is_use_IR)
        self.is_draw_IR_check_box.grid(row=0, column=0)

        b = ttk.Button(self.frameMicro_IR, text="Open file",  command=self.openIR_file)
        b.grid(row=0, column=1)

        b = ttk.Button(self.frameMicro_IR, text="Generate",  command=self.generateIR_file)
        b.grid(row=0, column=2)

        label = tk.Label(self.frameMicro_IR, text="IR Name :")
        label.grid(row=1, column=0)
        self.ir_name_sv = tk.StringVar()
        w = ttk.Entry(self.frameMicro_IR, textvariable=self.ir_name_sv, justify=tk.CENTER, width=50)
        w.grid(row=1, column=1, columnspan=2)
        self.ir_name_sv.set("None loaded yet")


        label = tk.Label(self.frameMicro_IR, text="Beginning % :")
        label.grid(row=2, column=0)
        self.ir_start_sv = tk.StringVar()
        w = ttk.Entry(self.frameMicro_IR, textvariable=self.ir_start_sv, justify=tk.CENTER, width=15)
        w.bind('<Return>', self.change_IR)
        self.ir_start_sv.set(0)
        w.grid(row=2, column=1)

        label = tk.Label(self.frameMicro_IR, text="End % :")
        label.grid(row=3, column=0)
        self.ir_end_sv = tk.StringVar()
        w = ttk.Entry(self.frameMicro_IR, textvariable=self.ir_end_sv, justify=tk.CENTER, width=15)
        w.bind('<Return>', self.change_IR)
        self.ir_end_sv.set(100)
        w.grid(row=3, column=1)

        label = tk.Label(self.frameMicro_IR, text="Shift (µchannel) :")
        label.grid(row=4, column=0)
        self.shiftIR_amount_sv = tk.StringVar()
        w = ttk.Entry(self.frameMicro_IR, textvariable=self.shiftIR_amount_sv, justify=tk.CENTER, width=15)
        w.bind('<Return>', self.change_IR)
        w.grid(row=4, column=1)
        self.shiftIR_amount_sv.set(0)

        b = ttk.Button(self.frameMicro_IR, text="Auto",  command=self.autoShiftIR)
        b.grid(row=3, column=2)

        #FIT
        self.gui_for_fit_operation = guiForFitOperation_Lifetime(self.frameMicro_fit, self.controller, ('One Decay', 'Two Decays', 'Rotation'),  nbParamFit=8)
        self.gui_for_fit_operation.populate()

    def openIR_file(self):
        file_path = filedialog.askopenfilename(title="Open IR File", initialdir=self.controller.view.saveDir)
        if file_path == None or file_path == '':
            return None
        ir_exp = Experiment.Experiment(file_path)
        measurement_ir = ir_exp.create_measurement(0, 0, -1,"lifetime", "", "")
        ir_exp.calculate_life_time(measurement_ir)
        self.ir_name_sv.set(ir_exp.file_name)
        self.controller.current_measurement.IR_raw = measurement_ir.data
        self.controller.current_measurement.IR_time_axis = measurement_ir.time_axis
        self.controller.update_analyze()

    def is_use_IR(self):
        self.change_IR(None)


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

