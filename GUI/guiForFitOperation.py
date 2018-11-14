import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class guiForFitOperation():

    #NB class variable shared by all instance.
    # idx_lim_for_fit_min = 0
    # idx_lim_for_fit_max = -1
    # idx_lim_for_fit_min_sv = tk.StringVar()
    # idx_lim_for_fit_max_sv = tk.StringVar()

    def __init__(self, masterFrame, controller, modelNames, nbParamFit, fitModeName=""):
        self.masterFrame = masterFrame
        self.modelNames = modelNames
        self.controller = controller

        self.listLabelParamFit = []
        self.listLabelStringVariableFit = []
        self.listEntryParamFit = []
        self.listEntryStringVariableFit = []


        self.fitModeName = fitModeName

        self.nbParamFit = nbParamFit

    def populate(self):
        label = ttk.Label(self.masterFrame, text='Model')
        label.grid(row=0, column=0)

        self.comboBoxStringVar = tk.StringVar()
        cb = ttk.Combobox(self.masterFrame, width=15, justify=tk.CENTER, textvariable=self.comboBoxStringVar,
                          values='', state='readonly')
        cb.bind('<<ComboboxSelected>>', self.changeModel)
        cb['values'] = self.modelNames
        self.comboBoxStringVar.set(self.modelNames[0])
        cb.set(self.modelNames[0])
        cb.grid(row=0, column=1)

        b = ttk.Button(self.masterFrame, text="IniGuess", command=self.iniGuessFit)
        b.grid(row=1, column=0)

        b = ttk.Button(self.masterFrame, text="Eval", command=self.evalFit)
        b.grid(row=1, column=1)

        b = ttk.Button(self.masterFrame, text="Fit", command=self.fit)
        b.grid(row=1, column=2)

        label = ttk.Label(self.masterFrame, text='x1, x2')
        label.grid(row=2, column=0)

        self.idx_lim_for_fit_min_sv = tk.StringVar()
        e = ttk.Entry(self.masterFrame, textvariable=self.idx_lim_for_fit_min_sv, justify=tk.CENTER, width=12)
        e.grid(row=2, column=1)

        self.idx_lim_for_fit_max_sv = tk.StringVar()
        e = ttk.Entry(self.masterFrame, textvariable=self.idx_lim_for_fit_max_sv, justify=tk.CENTER, width=12)
        e.grid(row=2, column=2)


        self.formulaFrame = tk.Frame(master=self.masterFrame)
        self.formulaFrame.grid(row=3, column=0, columnspan=3)

        self.figTex = plt.Figure(figsize=(10, 1), dpi=30, frameon=False)
        self.axTex = self.figTex.add_axes([0, 0, 1, 1])

        self.axTex.axis('off')

        # self.axTex.get_xaxis().set_visible(False)
        # self.axTex.get_yaxis().set_visible(False)

        self.canvasTk = FigureCanvasTkAgg(self.figTex, master=self.formulaFrame)
        self.canvasTk.get_tk_widget().pack(side='top', fill='both', expand=1)



        for i in range(self.nbParamFit):
            # TODO validate that entries are numeric value (cf method in GUI_root)

            self.listLabelStringVariableFit.append(tk.StringVar())
            txt = "p" + str(i)
            self.listLabelParamFit.append(
                ttk.Label(self.masterFrame, text=txt, textvariable=self.listLabelStringVariableFit[i]))
            col = 3 + 2 * (int(i) % 2)
            row = int(i / 2)
            self.listLabelParamFit[i].grid(row=row, column=col)

            self.listEntryStringVariableFit.append(tk.StringVar())
            self.listEntryParamFit.append(
                ttk.Entry(self.masterFrame, textvariable=self.listEntryStringVariableFit[i], justify=tk.CENTER,
                          width=7, state=tk.DISABLED))
            col = 4 + 2 * (int(i) % 2)
            row = int(i / 2)
            self.listEntryParamFit[i].grid(row=row, column=col)

        self.changeModel(None)

        #TODO equation (format latex) du modèle cf snipset.


    def setFitFormula(self, formula, fontsize=40):
        formula = "$" + formula + "$"

        self.axTex.clear()
        self.axTex.text(0, 0.2, formula, fontsize=fontsize)
        self.canvasTk.draw()

    def changeModel(self, event):
        # Methode virtuelle, voir les classes dérivées.
        raise  NotImplementedError()

    def evalFit(self):
        params = []
        # TODO more pythonic
        for sv in self.listEntryStringVariableFit:
            strValue = sv.get()
            if strValue == "":
                params.append(0)
            else:
                params.append(float(sv.get()))

        if self.idx_lim_for_fit_min_sv.get() == "":
            xlimMinFit = 0
        else:
            xlimMinFit = float(self.idx_lim_for_fit_min_sv.get())

        if  self.idx_lim_for_fit_max_sv.get() == "":
            xlimMaxFit = -1
        else :
            xlimMaxFit = float(self.idx_lim_for_fit_max_sv.get())

        self.controller.fit(mode="eval", model_name=self.comboBoxStringVar.get(),
                            params=params, idx_start=xlimMinFit, idx_end=xlimMaxFit)


    def iniGuessFit(self):
        if self.idx_lim_for_fit_min_sv.get() == "":
            xlimMinFit = 0
        else:
            xlimMinFit = float(self.idx_lim_for_fit_min_sv.get())

        if self.idx_lim_for_fit_max_sv.get() == "":
            xlimMaxFit = -1
        else:
            xlimMaxFit = float(self.idx_lim_for_fit_max_sv.get())

        self.controller.guess_eval_fit(mode="guess", model_name=self.comboBoxStringVar.get(),
                            params=None, idx_start=xlimMinFit, idx_end=xlimMaxFit)


    def fit(self):
        params = []
        # TODO more pythonic
        for sv in self.listEntryStringVariableFit:
            strValue = sv.get()
            if strValue == "":
                params.append(0)
            else:
                params.append(float(sv.get()))

        if self.idx_lim_for_fit_min_sv.get() == "":
            xlimMinFit = 0
        else :
            xlimMinFit = float(self.idx_lim_for_fit_min_sv.get())

        if  self.idx_lim_for_fit_max_sv.get() == "":
            xlimMaxFit = -1
        else :
            xlimMaxFit = float(self.idx_lim_for_fit_max_sv.get())

        self.controller.guess_eval_fit(mode="fit", model_name=self.comboBoxStringVar.get(),
                            params=params, idx_start=xlimMinFit, idx_end=xlimMaxFit)

    def setParamsFromFit(self, params):
        i = 0
        for paramName, param in params.items():
            self.listEntryStringVariableFit[i].set(str(param.value))
            i += 1
