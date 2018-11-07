import tkinter as tk
from tkinter import ttk

from GUI.graph.Graph_navigation import  Graph_navigation
from GUI.graph.interactiveGraphs import Graph_miniPCH
from GUI.graph.Graph_timeZoom import  Graph_timeZoom

class Status_area():
    def __init__(self, master_frame, view, controller, appearence_param):
        self.master_frame = master_frame
        self.view = view
        self.controller = controller
        self.appearence_param = appearence_param

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

        self.combobox_file_sv = tk.StringVar()
        self.combobox_file = ttk.Combobox(self.master_frame, width=80, justify=tk.CENTER, textvariable=self.combobox_file_sv, values='')
        self.combobox_file.pack(side="top", fill="both", expand=True)

        self.combobox_file.bind('<<ComboboxSelected>>', self.on_file_selected_via_combobox)



        label = ttk.Label(self.master_frame, text='Channel :')
        label.pack(side=tk.LEFT, padx=2, pady=2)
        self.channelNumber = tk.StringVar()
        self.channelNumber.set('1')
        self.spinBoxMaxOrder = tk.Spinbox(self.master_frame, width=8, textvariable=self.channelNumber,
                                          justify=tk.CENTER, from_=1, to=2)
        self.spinBoxMaxOrder.pack(side=tk.LEFT, padx=2, pady=2)

        self.labeFileName = tk.StringVar()
        l = ttk.Label(self.master_frame, width=60, textvariable=self.labeFileName)
        l.pack(side=tk.LEFT, padx=2, pady=2)

        label = ttk.Label(self.master_frame, text='nbOfPhoton :')
        label.pack(side=tk.LEFT, padx=2, pady=2)

        self.nbOfPhoton_sv = tk.StringVar()
        self.labelNbOfPhoton = ttk.Label(self.master_frame, textvariable=self.nbOfPhoton_sv)
        self.labelNbOfPhoton.pack(side=tk.LEFT, padx=2, pady=2)


        label = ttk.Label(self.master_frame, text='CPS :')
        label.pack(side=tk.LEFT, padx=2, pady=2)

        self.CPS_sv = tk.StringVar()
        self.labelCPS = ttk.Label(self.master_frame, textvariable=self.CPS_sv)
        self.labelCPS.pack(side=tk.LEFT, padx=2, pady=2)

        b = ttk.Button(self.master_frame, text="?", width=2, command=self.ask_file_info)
        b.pack(side=tk.RIGHT, padx=2, pady=2)

    def add_file_combobox(self, file_name):
        pass

    def remove_file_combobox(self):
        pass

    def on_file_selected_via_combobox(self):
        pass

    def set_file_name(self, name):
        self.labeFileName.set(name)

    def set_nb_of_photon_and_CPS(self, nbOfPhoton=0, CPS=0):
        self.nbOfPhoton_sv.set(str(nbOfPhoton))
        self.CPS_sv.set(str(int(CPS)))


    def ask_file_info(self):
        pass


