import tkinter as tk
from tkinter import ttk

#https://stackoverflow.com/questions/673174/file-dialogs-of-tkinter-in-python-3
from tkinter import filedialog, messagebox, simpledialog

import tkinter.scrolledtext as tkst


import os.path
from pylab import *
from IPython import embed
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#matplotlib.use("TkAgg")
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import time
import numpy as np
#import midiControl
from matplotlib.widgets import SpanSelector


from . import Menu

from . import Architecture

from .PlotParam import PlotParam

import logging


from .appearanceParameters import appearenceParam

class View():
    def __init__(self, master=None, controller=None):
        self.master = master
        self.controller = controller

        self.appearenceParam = appearenceParam()
        # self.plot_param = PlotParam()

        self.logger = None

        self.currentTimeWindow = [0,0] #µs
        self.current_time_zoom_window = [0, 0]  # µs
        self.current_graph_result_limit = [0, 0]  # µs
        self.currentChannel = 0
        self.timezoom_bin_size_s = 0.01

        self.is_a_FileLoaded = False

        #TODO gestion des modfiers.
        self.ctrl_is_held = False
        self.alt_is_held = False
        self.shift_is_held = False



        # def callback_crtl(event):
        #     self.crtl_isPressed = True
        #
        # def callback_alt(event):
        #     self.alt_isPressed = True
        #
        # def callback_shift(event):
        #     self.shift_isPressed = True
        #
        #
        #
        # self.master.bind("<Control_L>", callback_crtl)
        # self.master.bind("<Alt_L>", callback_alt)
        # self.master.bind("<Shift_L>", callback_shift)
        #
        # self.master.bind("<KeyRelease>", callback_shift)


        #https://stackoverflow.com/questions/27215326/tkinter-keypress-keyrelease-events
        self.initialize()
        self.create_syntaxic_shortcut()
        self.create_shorcut()

        self.saveDir = None

        #print master
        # self.onStart()
    def create_shorcut(self):
        # https://stackoverflow.com/questions/7299955/tkinter-binding-a-function-with-arguments-to-a-widget
        self.master.bind_all("<Control-o>", lambda event : self.menu.askOpenSPC_file())
        self.master.bind_all("<Control-l>", lambda event : self.menu.load_state())
        self.master.bind_all("<Control-s>", lambda event : self.menu.save_state())
        self.master.bind_all("<Control-q>", lambda event : self.menu.quit())

    def create_syntaxic_shortcut(self):
        self.graph_result = self.archi.analyze_area.resultArea_gui.graph_results
        self.gui_for_fit_operation = self.archi.analyze_area.gui_for_fit_operation

    def onStart(self):
        pass

    def onQuit(self):
        pass


    def lockGUI(self):
        pass

    def unlockGUI(self):
        pass

    def update(self):
        pass

    def initialize(self):
        self.menu = Menu.Menu(self.master, self)
        #https: // stackoverflow.com / questions / 25751730 / resizing - tkinter - windows - and -contents
        #For the windoww to be resizable
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)

        self.padx, self.pady = 2,2

        #In order to check if some uesr value entered via Entry are really numeric
        self.vcmd = (self.master.register(self.validateEntryNumeric),
                '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')

        self.archi = Architecture.Architecture(self.master, self)

    def pressed_key_shortcut(self, event):
        # print(event.char)
        if event.char == "a":
            self.archi.status_area.master_frame.focus_set()
        if event.char == "z":
            self.archi.navigation_area.master_frame.focus_set()
        if event.char == "e":
            self.archi.analyze_area.master_frame.focus_set()
        if event.char == "r":
            self.archi.log_area.master_frame.focus_set()
        if event.keysym == "Control_L":
            self.ctrl_is_held = True
            # print("Control_L")
        if event.keysym == "Shift_L":
            self.shift_is_held = True
        if event.keysym == "Alt_L":
            self.alt_is_held = True


    def validateEntryNumeric(self, action, index, value_if_allowed,
                 prior_value, text, validation_type, trigger_type, widget_name):
        #https://stackoverflow.com/questions/8959815/restricting-the-value-in-tkinter-entry-widget
        if text in '0123456789.-+':
            try:
                float(value_if_allowed)
                return True
            except ValueError:
                return False
        else:
            return False

    def changeChronoBinSize(self):
        print("Change bin size")

    def launchMicroTimeHisto(self):
        self.controller.drawMicroTimeHisto(self.currentChannel, self.currentTimeWindow[0], self.currentTimeWindow[1])
        print("launchMicroTimeHisto")

    def launchCorrelation(self):
        #c1 c2 maxCorrel_1s, B
        self.controller.drawCorrelation(0,0, 1, 10)


    def on_MicroTimevisibility(self, event):
        if self.is_a_FileLoaded:
            self.currentTab = "micro"
            self.controller.change_tab()

    def on_MacroTimevisibility(self, event):
        if self.is_a_FileLoaded:
            self.currentTab = "macro"
            self.controller.change_tab()

    def zoomOnMainChrono(self, t1_ms, t2_ms):
        self.controller.drawMainChronogram(self.currentChannel, t1_ms, t2_ms, self.timezoom_bin_size_s)



    def fill_view_with_measurement_params(self, measurement):
        pass

    def fill_view_with_exp_params(self, measurement):
        pass

    def fill_view_with_expS_params(self, measurement):
        pass

    def saveState(self, shelf):
        shelf['appearenceParam'] = self.appearenceParam
        # shelf['menu'] = self.menu
        shelf['appearenceParam'] = self.appearenceParam = appearenceParam()

        shelf['currentTimeWindow'] = self.currentTimeWindow
        shelf['current_time_zoom_window'] = self.current_time_zoom_window
        shelf['current_graph_result_limit'] = self.current_graph_result_limit
        shelf['currentChannel'] = self.currentChannel
        shelf['timezoom_bin_size_s'] = self.timezoom_bin_size_s

        # shelf['exp_iid_dict'] = self.archi.status_area.exp_iid_dict
        # shelf['mes_iid_dict'] = self.archi.status_area.mes_iid_dict

    def loadState(self, shelf):
        self.appearenceParam = shelf['appearenceParam']
        # self.menu = shelf['menu']

        self.currentTimeWindow = shelf['currentTimeWindow']
        self.current_time_zoom_window = shelf['current_time_zoom_window']
        self.current_graph_result_limit = shelf['current_graph_result_limit']
        self.currentChannel = shelf['currentChannel']
        self.timezoom_bin_size_s = shelf['timezoom_bin_size_s']

        # self.archi.status_area.exp_iid_dict = shelf['exp_iid_dict']
        # self.archi.status_area.mes_iid_dict = shelf['mes_iid_dict']






