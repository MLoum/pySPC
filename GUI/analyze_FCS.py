import tkinter as tk
from tkinter import ttk
from .guiForFitOperation import guiForFitOperation


class guiForFitOperation_FCS(guiForFitOperation):

    def __init__(self, masterFrame, controller, modelNames, nbParamFit):
        super().__init__( masterFrame, controller, modelNames, nbParamFit, fitModeName="FCS")

    def changeModel(self, event):
        # Methode virtuelle, voir les classes dérivées.
        nbFitParam = self.nbParamFit
        if self.cb_model_sv.get() == "1 Diff":
            self.listLabelStringVariableFit[0].set("G0")
            self.listLabelStringVariableFit[1].set("tdiff")
            self.listLabelStringVariableFit[2].set("cst")

            for i in range(3):
                self.listEntryParamFit[i].state(['!disabled'])

            for i in range(3, nbFitParam) :
                self.listLabelStringVariableFit[i].set("")
                self.listEntryParamFit[i].state(['disabled'])

            self.setFitFormula(r"G_0 \frac{1}{1+t/tdiff} + cst")

        else :
            for i in range(nbFitParam) :
                self.listLabelStringVariableFit[i].set("")
                self.listEntryParamFit[i].state(['disabled'])



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

        b = ttk.Button(self.frame_Correlate, text="AutoCorrelation", width=12, command=self.launchAutoCorrelationFCS)
        b.grid(row=1, column=0)

        b = ttk.Button(self.frame_Correlate, text="CrossCorrelation", width=12, command=self.launchCrossCorrelationFCS)
        b.grid(row=1, column=1)

        #FIT
        self.gui_for_fit_operation = guiForFitOperation_FCS(self.frame_fit, self.controller, ('1 Diff', 'Rotation'),  nbParamFit=8)
        self.gui_for_fit_operation.populate()


    def launchAutoCorrelationFCS(self):
        self.controller.calculate_measurement()
        self.controller.graph_measurement()

    def launchCrossCorrelationFCS(self):
        # TODO askSimpleDialog for Cross
        pass
        #self.controller.correlateFCS()


