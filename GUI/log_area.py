import tkinter as tk

import tkinter.scrolledtext as tkst

from GUI.graph.Graph_Results import Graph_Results


class Log_area():
    def __init__(self, masterFrame, view, controller, appearenceParam):
        self.masterFrame = masterFrame
        self.view = view
        self.controller = controller
        self.appearenceParam = appearenceParam


    def populate(self):
        self.logArea = tkst.ScrolledText(self.masterFrame, wrap=tk.WORD, width=100, height=10)

        self.logArea.pack(side=tk.LEFT, padx=2, pady=2)

        self.addLogMessage("Hi !")

    def addLogMessage(self, msg):
        self.logArea.insert(tk.INSERT, msg)
