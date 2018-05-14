import tkinter as tk
from tkinter import ttk


from .analyze_Lifetime import lifeTimeAnalyze_gui
from .analyze_DLS import DLS_Analyze_gui
from .analyze_FCS import FCS_Analyze_gui

from .resultsArea import Results_area

class Analyze_area():
    def __init__(self, masterFrame, view, controller, appearenceParam):
        self.masterFrame = masterFrame
        self.view = view
        self.controller = controller
        self.appearenceParam = appearenceParam

    def populate(self):
        self.frameCommon = tk.LabelFrame(self.masterFrame, text="Common", borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frameCommon.pack(side="top", fill="both", expand=True)

        #combo box anlayse Source
        label = ttk.Label(self.frameCommon, text='Source')
        label.pack(side=tk.LEFT, padx=2, pady=2)
        self.analyzeComboBoxSource_sv = tk.StringVar()
        cb = ttk.Combobox(self.frameCommon, width=25, justify=tk.CENTER, textvariable=self.analyzeComboBoxSource_sv, values='')
        cb.bind('<<ComboboxSelected>>', self.changeAnalyzeSource)
        cb['values'] = ('Whole', 'Time Zoom', 'Selection')
        self.analyzeComboBoxSource_sv.set('Time Zoom')
        cb.pack(side=tk.LEFT, padx=2, pady=2)

        #ProgressBar
        """
        If your program cannot accurately depict the relative progress that this widget is supposed to display, use mode='indeterminate'. In this mode, a rectangle bounces back and forth between the ends of the widget once you use the .start() method.
        If your program has some measure of relative progress, use mode='determinate'. In this mode, your program can move the indicator to a specified position along the widget's track.
        """
        self.analyzePgb = ttk.Progressbar(self.frameCommon, orient="horizontal", length=15)
        self.analyzePgb.pack(side=tk.LEFT, fill=tk.X)
        #self.analyzePgb.ste

        #Live Analysis ?
        self.isLive = tk.IntVar()
        self.isLiveCheckBox =  ttk.Checkbutton(self.frameCommon, text="Live ?", variable=self.isLive)
        self.isLiveCheckBox.pack(side=tk.LEFT, fill=tk.X)

        self.frameOperation = tk.LabelFrame(self.masterFrame, text="Operation", borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frameOperation.pack(side="top", fill="both", expand=True)

        #Notebook Operation
        self.tabOperation = ttk.Notebook(self.frameOperation)
        self.tabOperationFrameMacroTime = ttk.Frame(self.tabOperation)
        self.tabOperationFrameMicroTime = ttk.Frame(self.tabOperation)
        self.tabOperationFrameFilter = ttk.Frame(self.tabOperation)
        self.tabOperationCorrelation = ttk.Frame(self.tabOperation)
        self.tabOperationDLS = ttk.Frame(self.tabOperation)

        self.tabOperation.add(self.tabOperationFrameMacroTime, text='Chronogram')
        self.tabOperation.add(self.tabOperationFrameMicroTime, text='MicroTime')
        self.tabOperation.add(self.tabOperationCorrelation, text='FCS')
        self.tabOperation.add(self.tabOperationFrameFilter, text='Filter')
        self.tabOperation.add(self.tabOperationDLS, text='DLS')

        self.tabOperation.pack(side="top", fill="both", expand=True)

        self.tabOperationFrameMicroTime.bind("<Visibility>", self.on_ChangeAnalyzeTab)
        self.tabOperationFrameMacroTime.bind("<Visibility>", self.on_ChangeAnalyzeTab)
        self.tabOperationFrameFilter.bind("<Visibility>", self.on_ChangeAnalyzeTab)
        self.tabOperationCorrelation.bind("<Visibility>", self.on_ChangeAnalyzeTab)
        self.tabOperationDLS.bind("<Visibility>", self.on_ChangeAnalyzeTab)



        self.chronogramBinSize_micros = ttk.Entry(self.tabOperationFrameMacroTime, width=6)
        #self.chronogramBinSize_micros.bind('<Enter>', self.changeChronoBinSize)
        self.chronogramBinSize_micros.pack(side=tk.LEFT, padx=2, pady=2)



        self.lifeTimeAnalyze_gui = lifeTimeAnalyze_gui(self.tabOperationFrameMicroTime, self.controller, self.appearenceParam)
        self.lifeTimeAnalyze_gui.populate()

        self.FCS_TimeAnalyze_gui = FCS_Analyze_gui(self.tabOperationCorrelation, self.controller,
                                                       self.appearenceParam)
        self.FCS_TimeAnalyze_gui.populate()

        self.DLS_TimeAnalyze_gui = DLS_Analyze_gui(self.tabOperationDLS, self.controller,
                                                       self.appearenceParam)
        self.DLS_TimeAnalyze_gui.populate()


        self.frameResult = tk.LabelFrame(self.masterFrame, text="Results", borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frameResult.pack(side="top", fill="both", expand=True)

        self.resultArea_gui = Results_area(self.frameResult, self.view, self.controller, self.appearenceParam)
        self.resultArea_gui.populate()


    def copyData(self, target):
        """
        We can't change a widget master with Tkintyer, so one way to ove" a widget from one point
        to another in the GUI is to have two instance of the GUI with different master and copy the -> data <- form
        one to the other
        """
        #TODO
        pass
        #self.resultArea_gui.graph_results.copyData(target.graph_navigation)



    def on_ChangeAnalyzeTab(self, event):
        if self.view.is_a_FileLoaded == False:
            return
        currentNumTab  =  self.tabOperation.index(self.tabOperation.select())
        if currentNumTab == 0:
            self.view.currentOperation = "macro"
        elif currentNumTab == 1:
            self.view.currentOperation = "micro"
        elif currentNumTab == 2:
            self.view.currentOperation = "filter"
        elif currentNumTab == 3:
            self.view.currentOperation = "FCS"
        elif currentNumTab == 4:
            self.view.currentOperation = "DLS"
        # self.view.controller.update_analyze()

    def changeAnalyzeSource(self):
        #TODO
        pass