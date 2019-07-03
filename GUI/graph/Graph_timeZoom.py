import tkinter as tk
from tkinter import ttk


#from pylab import *
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np

from tkinter import filedialog

from matplotlib.widgets import SpanSelector
from matplotlib.widgets import Cursor
from matplotlib.widgets import MultiCursor
import matplotlib.patches as patches

from .interactiveGraphs import InteractiveGraph

class Graph_timeZoom(InteractiveGraph):
    """
        Il s'agit d'une zone où on peut zoomer sur la time trace pour faire des analyses précises,
        voire à la particule unique si le S/N le permet.

        On aussi sur la droite un Photon Counting Histogramm qui est bien pratique pour fixer des seuils
        de filtrage.

        Il y a aussi une zone permettant d'identifier certaine partie de la courbe de temps de vie ou de corrélation.
        :return:
    """

    def __init__(self, master_frame, view, controller, figsize, dpi):
        super().__init__(master_frame, view, controller, figsize, dpi)
        #self.ax.axis('off')
        self.figure.tight_layout()

        self.threshold = None
        self.threshold_flank = None
        self.chrono = None


        self.createWidgets()
        self.createCallBacks()

    def plot(self, chrono):
        self.chrono = chrono

        if self.ax == None:
            self.mainAx = self.figure.add_subplot(111)
            self.subplot3D = None
        self.ax.clear()

        #reduce nb of point to 1000 (approximative size in pixel
        if chrono.nb_of_bin > 1000:
            skipsize = int(chrono.nb_of_bin/1000)
            idx = np.arange(0, len(chrono.data), skipsize)
            chronoPlot = chrono.data[idx]
            chronoPlotX = chrono.time_axis[idx]
        else:
            chronoPlot = chrono.data
            chronoPlotX = chrono.time_axis

        self.ax.set_xlim(chrono.time_axis[0], chrono.time_axis[-1])

        self.ax.plot(chronoPlotX, chronoPlot)
        self.ax.fill_between(chronoPlotX, 0, chronoPlot, alpha=0.3)

        if self.threshold is not None:
            self.ax.hlines(self.threshold, chrono.time_axis[0], chrono.time_axis.max(), linewidth=4)

        if self.threshold_flank is not None:
            self.ax.hlines(self.threshold_flank, chrono.time_axis[0], chrono.time_axis.max(), linewidth=4)

        if self.view.current_time_zoom_window != [0, 0]:
            self.ax.add_patch(
                patches.Rectangle(
                    (self.view.current_time_zoom_window[0], 0),  # (x,y)
                    self.view.current_time_zoom_window[1]-self.view.current_time_zoom_window[0],  # width
                    chrono.data.max(),  # height
                    alpha=0.2
                )
            )

        self.figure.canvas.draw()

    def scrollEvent(self, event):
        if self.shift_is_held==False:
            #On translate
            if event.button == 'up':
                delta = self.view.currentTimeWindow[1] - self.view.currentTimeWindow[0]
                self.view.currentTimeWindow[0] += delta / 2
                self.view.currentTimeWindow[1] += delta / 2

                # TODO vérfier borne depassement fichier.
            elif event.button == 'down':
                delta = self.view.currentTimeWindow[1] - self.view.currentTimeWindow[0]
                self.view.currentTimeWindow[0] -= delta / 2
                self.view.currentTimeWindow[1] -= delta / 2
                if self.view.currentTimeWindow[0] < 0:
                    self.view.currentTimeWindow[0] = 0

            #FIXME something is wrong here and make the soft carchshs
            # self.controller.zoom()
        else:
            #On zoom
            pass


    def button_press_event(self, event):
        #print('you pressed', event.button, event.xdata, event.ydata)
        if event.button == 2:
            # pass
            # # click gauche
            # if event.key == "shift":

            # if event.dblclick:
            file_path = filedialog.asksaveasfile(title="Export file name ?",
                                                 initialdir=self.controller.view.saveDir)
            if file_path == None or file_path == '':
                return None

            # Export the graph
            if self.chrono is not None:
                data = np.column_stack((self.chrono.time_axis,
                                        self.chrono.data))
                np.savetxt(file_path.name, data, header="time chrono")

            #Test if filter mode or if the cursord is needed.
            #put horizontal cursor at mouse position
            # self.cursor_h.set_active(True)
            # self.cursor_h.eventson = True
            #activate cursor drag
            #tell other graph, via the controller ?, that the cursor has changed its position
            pass

    def button_release_event(self, event):
        if event.button == 1:
            pass
            # click gauche

            #self.cursor_h.set_active(False)
            #self.cursor_h.eventson = False
            # self.cursor_h.visible = True
            #self.cursor_h.drawon = True

    def onSpanMove(self, xmin, xmax):
        if xmin > xmax:
            # swap
            xmax, xmin = xmin, xmax
        #print(xmin, xmax)
        self.view.current_time_zoom_window = [xmin, xmax]
        # self.controller.zoom()

    def onSpanSelect(self, xmin, xmax):
        if xmin > xmax:
            # swap
            xmax, xmin = xmin, xmax
        #print(xmin, xmax)
        self.view.current_time_zoom_window = [xmin, xmax]
        self.controller.update_navigation()

    def createWidgets(self):
        super().createWidgets()
        # self.cursor_h = Cursor(self.ax, useblit=True, color='red', horizOn=True, vertOn=False, linewidth=2)
        #
        # self.cursor_h.set_active(False)
        # self.cursor_h.drawon = True
        #drawon
        #eventson
        # self.setOnOffCursors(True)


    def createCallBacks(self):
        super().createCallBacks()
        # self.cursor_h.connect_event('onmove', callback=self.cursorMove)


    def cursorMove(self, event):
        print('Cursor move')
        print(event)

