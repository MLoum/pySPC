import tkinter as tk
from tkinter import ttk

#https://stackoverflow.com/questions/673174/file-dialogs-of-tkinter-in-python-3

import tkinter.scrolledtext as tkst

#matplotlib.use("TkAgg")
#import midiControl

from .navigation_area import navigation_area
from .status_area import Status_area
from .analyze_area import Analyze_area
from .log_area import Log_area



class Architecture():
    def __init__(self, master, main_GUI):
        self.master = master
        self.mainGUI = main_GUI
        self.controller = main_GUI.controller
        self.appearenceParam = main_GUI.appearenceParam

        # self.isDocked = False
        self.create_architecture()

    def create_architecture(self):
        # self.createDockedArchitecture()
        # self.createWindowArchitecture()

        # 1 Status and file selection
        self.frame_status = tk.LabelFrame(self.master, text="Status", borderwidth=self.appearenceParam.frameLabelBorderWidth)

        self.status_area = Status_area(self.frame_status, self.mainGUI, self.controller, self.appearenceParam)
        self.status_area.populate()
        self.frame_status.pack(side="top", fill="both", expand=True)

        # 2 Navigation
        self.top_level_navigation = tk.Toplevel(self.master)
        self.top_level_navigation.title("Navigation")
        self.frame_navigation = tk.LabelFrame(self.top_level_navigation, text="Navigation",
                                              borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.navigation_area = navigation_area(self.frame_navigation, self.mainGUI, self.controller, self.appearenceParam)
        self.navigation_area.populate()
        self.frame_navigation.pack(side="top", fill="both", expand=True)
        # self.topLevelNavigation.withdraw()

        # 3 Analyze
        self.top_level_analyze = tk.Toplevel(self.master)
        self.top_level_analyze.title("Analysis")
        self.frame_analyze = tk.LabelFrame(self.top_level_analyze, text="Analyze", borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.analyze_area = Analyze_area(self.frame_analyze, self.mainGUI, self.controller, self.appearenceParam)
        self.analyze_area.populate()
        self.frame_analyze.pack(side="top", fill="both", expand=True)
        # self.topLevelAnalyze.withdraw()

        # 4 Log
        self.top_level_log = tk.Toplevel(self.master)
        self.frame_log = tk.LabelFrame(self.top_level_log, text="Log", borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.log_area = Log_area(self.frame_log, self.mainGUI, self.controller, self.appearenceParam)
        self.log_area.populate()
        self.frame_log.pack(side="top", fill="both", expand=True)



"""
    def createDockedArchitecture(self):

        self.frame_status = tk.LabelFrame(self.master, text="Status", borderwidth=self.appearenceParam.frameLabelBorderWidth)

        self.status_area = Status_area(self.frame_status, self.mainGUI, self.controller, self.appearenceParam)
        self.status_area.populate()
        self.frame_status.pack(side="top", fill="both", expand=True)

        self.frameNavigation_docked = tk.LabelFrame(self.master, text="Navigation",
                                                borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.navigation_area_docked = navigation_area(self.frameNavigation_docked, self.mainGUI, self.controller, self.appearenceParam)
        self.navigation_area_docked.populate()
        #self.frameNavigation.pack(side="top", fill="both", expand=True)

        self.frameAnalyze_docked = tk.LabelFrame(self.master, text="Analyze", borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.analyze_area_docked = Analyze_area(self.frameAnalyze_docked, self.mainGUI, self.controller, self.appearenceParam)
        self.analyze_area_docked.populate()
        #self.frameAnalyze.pack(side="top", fill="both", expand=True)

        self.frame_log = tk.LabelFrame(self.master, text="Log", borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.log_area = Log_area(self.frame_log, self.mainGUI, self.controller, self.appearenceParam)
        self.log_area.populate()
        self.frame_log.pack(side="top", fill="both", expand=True)

    def createWindowArchitecture(self):
        #http://infohost.nmt.edu/tcc/help/pubs/tkinter/web/toplevel.html
        self.top_level_navigation = tk.Toplevel(self.master)
        self.top_level_navigation.title("Navigation")
        self.frameNavigation_window = tk.LabelFrame(self.top_level_navigation, text="Navigation",
                                                    borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.navigation_area_window = navigation_area(self.frameNavigation_window, self.mainGUI, self.controller, self.appearenceParam)
        self.navigation_area_window.populate()
        self.frameNavigation_window.pack(side="top", fill="both", expand=True)
        # self.topLevelNavigation.withdraw()

        self.top_level_analyze = tk.Toplevel(self.master)
        self.top_level_analyze.title("Analysis")
        self.frame_analyze = tk.LabelFrame(self.top_level_analyze, text="Analyze", borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.analyze_area = Analyze_area(self.frame_analyze, self.mainGUI, self.controller, self.appearenceParam)
        self.analyze_area.populate()
        self.frame_analyze.pack(side="top", fill="both", expand=True)
        # self.topLevelAnalyze.withdraw()

    def dock(self):

        self.frame_log.pack_forget()
        self.frameNavigation_docked.pack(side="top", fill="both", expand=True)
        self.frameAnalyze_docked.pack(side="top", fill="both", expand=True)
        self.frame_log.pack(side="top", fill="both", expand=True)

        self.navigation_area_window.copyData(self.navigation_area_docked)
        self.analyze_area.copyData(self.analyze_area_docked)

        self.top_level_navigation.withdraw()
        self.top_level_analyze.withdraw()

        self.navigation_area = self.navigation_area_docked
        self.analyze_area = self.analyze_area_docked

        self.controller.update_navigation()
        self.controller.update_analyze()


    def undock(self):

        self.navigation_area_docked.copyData(self.navigation_area_window)
        self.analyze_area_docked.copyData(self.analyze_area)

        self.frameNavigation_docked.pack_forget()
        self.frameAnalyze_docked.pack_forget()

        #https://stackoverflow.com/questions/22834150/difference-between-iconify-and-withdraw-in-python-tkinter
        self.top_level_navigation.deiconify()
        self.top_level_analyze.deiconify()
        
        self.navigation_area = self.navigation_area_window
        self.analyze_area = self.analyze_area

        self.controller.update_navigation()
        self.controller.update_analyze()


    def toggleDock(self):

        
        if self.isDocked == False:
            self.undock()
            self.isDocked = True
        else:
            self.dock()
            self.isDocked = False







"""
