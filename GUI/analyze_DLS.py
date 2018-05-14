import tkinter as tk
from tkinter import ttk
from .guiForFitOperation import guiForFitOperation


class guiForFitOperation_DLS(guiForFitOperation):

    def __init__(self, masterFrame, controller, modelNames, nbParamFit):
        super().__init__( masterFrame, controller, modelNames, nbParamFit, fitModeName="DLS")

    def changeModel(self, event):
        # Methode virtuelle, voir les classes dérivées.
        nbFitParam = self.nbParamFit
        if self.comboBoxStringVar.get() == "Cumulant":
            self.listLabelStringVariableFit[0].set("B")
            self.listLabelStringVariableFit[1].set("beta")
            self.listLabelStringVariableFit[2].set("Gamma")
            self.listLabelStringVariableFit[3].set("mu2")
            self.listLabelStringVariableFit[4].set("mu3")
            self.listLabelStringVariableFit[5].set("mu4")

            for i in range(6):
                self.listEntryParamFit[i].state(['!disabled'])

            for i in range(6, nbFitParam) :
                self.listLabelStringVariableFit[i].set("")
                self.listEntryParamFit[i].state(['disabled'])


            self.setFitFormula(r"B + \beta e^{-2 \Gamma t} (1 + \frac{\mu_2}{2!} t^2 + \frac{\mu_3}{3!} t^3 + \frac{\mu_3}{4!} t^4)^2")

        elif self.comboBoxStringVar.get() == "Inv TL":
            for i in range(nbFitParam) :
                self.listLabelStringVariableFit[i].set("")
                self.listEntryParamFit[i].state(['disabled'])
        else :
            for i in range(nbFitParam) :
                self.listLabelStringVariableFit[i].set("")
                self.listEntryParamFit[i].state(['disabled'])





class DLS_Analyze_gui():
    def __init__(self, masterFrame, controller, appearenceParam):
        self.masterFrame = masterFrame
        self.controller = controller
        self.appearenceParam = appearenceParam


    def populate(self):
        self.frameDLS_Correlate = tk.LabelFrame(self.masterFrame, text="Correlate",
                                                borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frameDLS_fit = tk.LabelFrame(self.masterFrame, text="Fit",
                                          borderwidth=self.appearenceParam.frameLabelBorderWidth)

        self.frameDLS_Correlate.pack(side="left", fill="both", expand=True)
        self.frameDLS_fit.pack(side="left", fill="both", expand=True)

        #Correlate
        label = ttk.Label(self.frameDLS_Correlate, text='Start Time (µs)')
        label.grid(row=0, column=0)

        self.start_time_micro_sv = tk.StringVar()
        e = ttk.Entry(self.frameDLS_Correlate, textvariable=self.start_time_micro_sv, justify=tk.CENTER, width=7)
        e.grid(row=0, column=1)
        self.start_time_micro_sv.set('1')


        label = ttk.Label(self.frameDLS_Correlate, text='Max Correl Time (ms)')
        label.grid(row=1, column=0)

        self.maxCorrelTime_sv = tk.StringVar()
        e = ttk.Entry(self.frameDLS_Correlate, textvariable=self.maxCorrelTime_sv, justify=tk.CENTER, width=7)
        e.grid(row=1, column=1)
        self.maxCorrelTime_sv.set('1000')

        label = ttk.Label(self.frameDLS_Correlate, text='Precision')
        label.grid(row=2, column=0)

        self.corel_precision_sv = tk.StringVar()
        e = ttk.Entry(self.frameDLS_Correlate, textvariable=self.corel_precision_sv, justify=tk.CENTER, width=7)
        e.grid(row=2, column=1)
        self.corel_precision_sv.set('10')

        b = ttk.Button(self.frameDLS_Correlate, text="Correlate", width=12, command=self.launchCorrelationDLS)
        b.grid(row=3, column=0)


        #FIT
        self.guiForFitOperation_DLS = guiForFitOperation_DLS(self.frameDLS_fit, self.controller, ('Cumulant', 'Inv TL'),  nbParamFit=8)
        self.guiForFitOperation_DLS.populate()


    def launchCorrelationDLS(self):
        self.controller.view.currentOperation = "DLS"
        self.controller.update_analyze()


