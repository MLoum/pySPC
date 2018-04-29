import tkinter as tk

import tkinter.scrolledtext as tkst

from GUI.graph.Graph_Results import Graph_Results


class Results_area():
    def __init__(self, masterFrame, view, controller, appearenceParam):
        self.masterFrame = masterFrame
        self.view = view
        self.controller = controller
        self.appearenceParam = appearenceParam


    def populate(self):
        # Results from fit -- Text
        self.frameResultText = tk.LabelFrame(self.masterFrame, text="Text",
                                             borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.frameResultText.pack(side=tk.LEFT, fill="both", expand=True)

        self.resultFitTextArea = tkst.ScrolledText(self.frameResultText, wrap=tk.WORD, width=45, height=20)
        self.resultFitTextArea.pack(side=tk.LEFT, padx=2, pady=2)
        self.resultFitTextArea.insert(tk.INSERT, "Gimme Results !")

        # #Shared by all the "guiForFitOperation"
        # self.idx_lim_for_fit_min_sv = tk.StringVar()
        # self.idx_lim_for_fit_max_sv = tk.StringVar()

        # Graphs
        self.frameAnalyzeGraphs = tk.LabelFrame(self.masterFrame, text="Graph",
                                               borderwidth=self.appearenceParam.frameLabelBorderWidth)

        self.frameAnalyzeGraphs.pack(side=tk.LEFT, fill="both", expand=True)


        self.graph_results = Graph_Results(self.frameAnalyzeGraphs, self.view, self.controller,
                                             figsize=(15, 2), dpi=100)


    def setTextResult(self, text):
        self.resultFitTextArea.insert(tk.INSERT, text)




