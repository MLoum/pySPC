import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog
from .guiForFitOperation import guiForFitOperation


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
    def __init__(self, masterFrame, controller, appearenceParam):
        self.masterFrame = masterFrame
        self.controller = controller
        self.appearenceParam = appearenceParam


    def populate(self):
        self.frameMicro_graph = tk.LabelFrame(self.masterFrame, text="Graph",
                                         borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frameMicro_PTOFS = tk.LabelFrame(self.masterFrame, text="PTOFS",
                                         borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frameMicro_IR = tk.LabelFrame(self.masterFrame, text="IR",
                                         borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frameMicro_fit = tk.LabelFrame(self.masterFrame, text="Fit",
                                         borderwidth=self.appearenceParam.frameLabelBorderWidth)

        self.frameMicro_graph.pack(side="left", fill="both", expand=True)
        self.frameMicro_PTOFS.pack(side="left", fill="both", expand=True)
        self.frameMicro_IR.pack(side="left", fill="both", expand=True)
        self.frameMicro_fit.pack(side="left", fill="both", expand=True)


        #Graph

        b = ttk.Button(self.frameMicro_graph, text="Graph", width=6, command=self.launchMicroTimeHisto)
        b.pack(side=tk.LEFT, padx=2, pady=2)

        self.isSemiLog = tk.IntVar()
        self.isDrawIRCheckBox = ttk.Checkbutton(self.frameMicro_graph, text="SemiLog ?", variable=self.isSemiLog)
        self.isDrawIRCheckBox.pack(side=tk.TOP, fill=tk.X)

        #PTOFS


        #IR
        b = ttk.Button(self.frameMicro_IR, text="Open", width=6, command=self.openIR_file)
        b.pack(side=tk.TOP, padx=2, pady=2)

        b = ttk.Button(self.frameMicro_IR, text="Generate", width=6, command=self.openIR_file)
        b.pack(side=tk.TOP, padx=2, pady=2)

        self.isDraw_IR = tk.IntVar()
        self.isDrawIRCheckBox = ttk.Checkbutton(self.frameMicro_IR, text="Draw ?", variable=self.isDraw_IR)
        self.isDrawIRCheckBox.pack(side=tk.TOP, fill=tk.X)

        self.shiftIR_amount = tk.IntVar()
        w = ttk.Scale(self.frameMicro_IR, from_=0, to=200, orient=tk.HORIZONTAL, variable=self.shiftIR_amount, command=self.changeIR_shift)
        w.pack(side=tk.TOP, padx=2, pady=2)


        b = ttk.Button(self.frameMicro_IR, text="Shift", width=6, command=self.autoShiftIR)
        b.pack(side=tk.TOP, padx=2, pady=2)

        #FIT
        self.guiForFitOperation_Lifetime = guiForFitOperation_Lifetime(self.frameMicro_fit, self.controller, ('One Decay', 'Two Decays', 'Rotation'),  nbParamFit=8)
        self.guiForFitOperation_Lifetime.populate()

    def openIR_file(self):
        pass

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
                        self.controller.generateArtificialIR(mainWidth, secondaryWidth, secondaryAmplitude, timeOffset)


    def autoShiftIR(self):
        pass

    def changeIR_shift(self, e):
        pass

    def launchMicroTimeHisto(self):
        #FIXME
        self.currentOperation = "micro"
        self.controller.updateAnalyze()
        # self.mainGUI.controller.drawMicroTimeHisto(self.mainGUI.currentChannel, self.mainGUI.currentTimeWindow[0], self.mainGUI.currentTimeWindow[1])
        # print("launchMicroTimeHisto")

