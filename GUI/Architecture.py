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
    def __init__(self, master, mainGUI):
        self.master = master
        self.mainGUI = mainGUI
        self.controller = mainGUI.controller
        self.appearenceParam = mainGUI.appearenceParam

        self.isDocked = False




        self.createArchitecture()

    def createArchitecture(self):
        self.createDockedArchitecture()
        self.createWindowArchitecture()





    def createDockedArchitecture(self):
        self.frameStatus = tk.LabelFrame(self.master, text="Status", borderwidth=self.appearenceParam.frameLabelBorderWidth)

        self.status_area = Status_area(self.frameStatus, self.mainGUI, self.controller, self.appearenceParam)
        self.status_area.populate()
        self.frameStatus.pack(side="top", fill="both", expand=True)

        self.frameNavigation_docked = tk.LabelFrame(self.master, text="Navigation",
                                                borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.navigation_area_docked = navigation_area(self.frameNavigation_docked, self.mainGUI, self.controller, self.appearenceParam)
        self.navigation_area_docked.populate()
        #self.frameNavigation.pack(side="top", fill="both", expand=True)

        self.frameAnalyze_docked = tk.LabelFrame(self.master, text="Analyze", borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.analyze_area_docked = Analyze_area(self.frameAnalyze_docked, self.mainGUI, self.controller, self.appearenceParam)
        self.analyze_area_docked.populate()
        #self.frameAnalyze.pack(side="top", fill="both", expand=True)

        self.frameLog = tk.LabelFrame(self.master, text="Log", borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.log_area = Log_area(self.frameLog, self.mainGUI, self.controller, self.appearenceParam)
        self.log_area.populate()
        self.frameLog.pack(side="top", fill="both", expand=True)

    def createWindowArchitecture(self):
        #http://infohost.nmt.edu/tcc/help/pubs/tkinter/web/toplevel.html
        self.topLevelNavigation = tk.Toplevel(self.master)
        self.topLevelNavigation.title("Navigation")
        self.frameNavigation_window = tk.LabelFrame(self.topLevelNavigation, text="Navigation",
                                                borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.navigation_area_window = navigation_area(self.frameNavigation_window, self.mainGUI, self.controller, self.appearenceParam)
        self.navigation_area_window.populate()
        self.frameNavigation_window.pack(side="top", fill="both", expand=True)
        self.topLevelNavigation.withdraw()

        self.topLevelAnalyze = tk.Toplevel(self.master)
        self.topLevelAnalyze.title("Analysis")
        self.frameAnalyze_window = tk.LabelFrame(self.topLevelAnalyze, text="Analyze", borderwidth=self.appearenceParam.frameLabelBorderWidth)
        self.analyze_area_window= Analyze_area(self.frameAnalyze_window, self.mainGUI, self.controller, self.appearenceParam)
        self.analyze_area_window.populate()
        self.frameAnalyze_window.pack(side="top", fill="both", expand=True)
        self.topLevelAnalyze.withdraw()


    def dock(self):
        self.frameLog.pack_forget()
        self.frameNavigation_docked.pack(side="top", fill="both", expand=True)
        self.frameAnalyze_docked.pack(side="top", fill="both", expand=True)
        self.frameLog.pack(side="top", fill="both", expand=True)

        self.navigation_area_window.copyData(self.navigation_area_docked)
        self.analyze_area_window.copyData(self.analyze_area_docked)

        self.topLevelNavigation.withdraw()
        self.topLevelAnalyze.withdraw()

        self.navigation_area = self.navigation_area_docked
        self.analyze_area = self.analyze_area_docked

        self.controller.update_navigation()
        self.controller.update_analyze()


    def undock(self):
        self.navigation_area_docked.copyData(self.navigation_area_window)
        self.analyze_area_docked.copyData(self.analyze_area_window)

        self.frameNavigation_docked.pack_forget()
        self.frameAnalyze_docked.pack_forget()

        #https://stackoverflow.com/questions/22834150/difference-between-iconify-and-withdraw-in-python-tkinter
        self.topLevelNavigation.deiconify()
        self.topLevelAnalyze.deiconify()
        
        self.navigation_area = self.navigation_area_window
        self.analyze_area = self.analyze_area_window

        self.controller.update_navigation()
        self.controller.update_analyze()


    def toggleDock(self):
        if self.isDocked == False:
            self.undock()
            self.isDocked = True
        else:
            self.dock()
            self.isDocked = False








