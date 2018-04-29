import tkinter as tk
from tkinter import ttk

from GUI.graph.Graph_navigation import  Graph_navigation
from GUI.graph.interactiveGraphs import Graph_miniPCH
from GUI.graph.Graph_timeZoom import  Graph_timeZoom

class Status_area():
    def __init__(self, masterFrame, view, controller, appearenceParam):
        self.masterFrame = masterFrame
        self.view = view
        self.controller = controller
        self.appearenceParam = appearenceParam

    def populate(self):
        # self.frameTimeStatus = tk.LabelFrame(self.masterFrame, text="Status", borderwidth=self.appearenceParam.frameLabelBorderWidth)
        # self.frameTimeStatus.pack(side="top", fill="both", expand=True)

        # self.frameFileBasicInfo = tk.Label(self.masterFrame)
        # self.frameFileBasicInfo.pack(side=tk.LEFT)
        #
        #
        # label = ttk.Label(self.frameFileBasicInfo, text='Channel :')
        # label.grid(column=0, row=0)
        # self.channelNumber = tk.StringVar()
        # self.channelNumber.set('1')
        # self.spinBoxMaxOrder = tk.Spinbox(self.frameFileBasicInfo, width=8, textvariable=self.channelNumber,
        #                                   justify=tk.CENTER, from_=1, to=2)
        # self.spinBoxMaxOrder.grid(column=1, row=0)
        #
        # self.labeFileName = tk.StringVar()
        # l = ttk.Label(self.frameFileBasicInfo, width=60, textvariable=self.labeFileName)
        # l.grid(column=2, row=0, columnspan=2)
        #
        # label = ttk.Label(self.frameFileBasicInfo, text='nbOfPhoton :')
        # label.grid(column=0, row=1)
        #
        # self.nbOfPhoton_sv = tk.StringVar()
        # self.labelNbOfPhoton = ttk.Label(self.frameFileBasicInfo, textvariable=self.nbOfPhoton_sv)
        # self.labelNbOfPhoton.grid(column=1, row=1)
        #
        #
        # label = ttk.Label(self.frameFileBasicInfo, text='CPS :')
        # label.grid(column=2, row=1)
        #
        # self.CPS_sv = tk.StringVar()
        # self.labelCPS = ttk.Label(self.frameFileBasicInfo, textvariable=self.CPS_sv)
        # self.labelCPS.grid(column=3, row=1)


        label = ttk.Label(self.masterFrame, text='Channel :')
        label.pack(side=tk.LEFT, padx=2, pady=2)
        self.channelNumber = tk.StringVar()
        self.channelNumber.set('1')
        self.spinBoxMaxOrder = tk.Spinbox(self.masterFrame, width=8, textvariable=self.channelNumber,
                                          justify=tk.CENTER, from_=1, to=2)
        self.spinBoxMaxOrder.pack(side=tk.LEFT, padx=2, pady=2)

        self.labeFileName = tk.StringVar()
        l = ttk.Label(self.masterFrame, width=60, textvariable=self.labeFileName)
        l.pack(side=tk.LEFT, padx=2, pady=2)

        label = ttk.Label(self.masterFrame, text='nbOfPhoton :')
        label.pack(side=tk.LEFT, padx=2, pady=2)

        self.nbOfPhoton_sv = tk.StringVar()
        self.labelNbOfPhoton = ttk.Label(self.masterFrame, textvariable=self.nbOfPhoton_sv)
        self.labelNbOfPhoton.pack(side=tk.LEFT, padx=2, pady=2)


        label = ttk.Label(self.masterFrame, text='CPS :')
        label.pack(side=tk.LEFT, padx=2, pady=2)

        self.CPS_sv = tk.StringVar()
        self.labelCPS = ttk.Label(self.masterFrame, textvariable=self.CPS_sv)
        self.labelCPS.pack(side=tk.LEFT, padx=2, pady=2)


        self.toggleDockButton_text_sv = tk.StringVar()
        b = ttk.Button(self.masterFrame, textvariable=self.toggleDockButton_text_sv, width=10, command=self.toggleDock)
        b.pack(side=tk.RIGHT, padx=2, pady=2)
        self.toggleDockButton_text_sv.set("undock")

        b = ttk.Button(self.masterFrame, text="?", width=2, command=self.askFileInfo)
        b.pack(side=tk.RIGHT, padx=2, pady=2)


    def setFileName(self, name):
        self.labeFileName.set(name)

    def setNbOfPhotonAndCPS(self, nbOfPhoton=0, CPS=0):
        self.nbOfPhoton_sv.set(str(nbOfPhoton))
        self.CPS_sv.set(str(int(CPS)))


    def askFileInfo(self):
        pass


    def toggleDock(self):
        if self.toggleDockButton_text_sv.get() == "undock":
            self.toggleDockButton_text_sv.set("dock")
        else:
            self.toggleDockButton_text_sv.set("undock")

        self.view.archi.toggleDock()

