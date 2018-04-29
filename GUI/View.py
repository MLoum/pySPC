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


from .appearanceParameters import appearenceParam

class View():
    def __init__(self, master = None, controller=None):
        self.master = master
        self.controller = controller

        self.appearenceParam = appearenceParam()

        self.currentTab = "macro"
        self.currentTimeWindow = [0,0] #µs
        self.currentChannel = 0
        self.currentBinSize_s = 0.01

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


        self.currentOperation = "macro"
        self.initialize()
        self.saveDir = None

        #print master
        # self.onStart()


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




    def onclickMainGraph(self, event):
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, event.xdata, event.ydata))

    def onclickNavigation(self, event):
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              (event.button, event.x, event.y, event.xdata, event.ydata))


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
            self.controller.changeTab()

    def on_MacroTimevisibility(self, event):
        if self.is_a_FileLoaded:
            self.currentTab = "macro"
            self.controller.changeTab()

    def zoomOnMainChrono(self, t1_ms, t2_ms):
        self.controller.drawMainChronogram(self.currentChannel, t1_ms, t2_ms, self.currentBinSize_s)


    def saveState(self, shelf):
        shelf['appearenceParam'] = self.appearenceParam
        shelf['menu'] = self.menu
        shelf['archi'] = self.archi

    def loadState(self, shelf):
        self.appearenceParam =  shelf['appearenceParam']
        self.menu = shelf['menu']
        self.archi = shelf['archi']



