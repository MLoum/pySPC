import tkinter as tk
from tkinter import ttk


#from pylab import *
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches

import numpy as np


from matplotlib.widgets import SpanSelector
from matplotlib.widgets import Cursor
from matplotlib.widgets import MultiCursor

from .interactiveGraphs import InteractiveGraph

class Graph_navigation(InteractiveGraph):
    """
    Il s'agit d'un graphe montrant la totalité du fichier.
    On peut cliquer et selctionner des zones pour naviguer dans le fichier
    """

    def __init__(self, master_frame, view, controller, figsize, dpi):
        super().__init__(master_frame, view, controller, figsize, dpi)
        self.ax.axis('off')
        self.figure.tight_layout()
        self.createCallBacks()
        self.createWidgets()

    def plot(self, mainChrono, tSelec_1=0, tSelec_2=-1):

        self.ax.clear()


        #FIXME test if 1000 bins ?
        # reduce nb of point to 1000 (approximative size in pixel
        min_nb_of_bin = 1000
        if mainChrono.nb_of_bin  > 1000:
            skipsize = int(mainChrono.nb_of_bin / 1000)
            idx = np.arange(0, len(mainChrono.data), skipsize)
            plot = mainChrono.data[idx]
            plotX = mainChrono.time_axis[idx]
        else:
            plot = mainChrono.data
            plotX = mainChrono.time_axis



        self.ax.plot(plotX, plot)
        self.ax.set_xlim(mainChrono.time_axis[0], mainChrono.time_axis[-1])

        x_start = tSelec_1
        if tSelec_2 == -1:
            x_end = mainChrono.time_axis.max()
        else:
            x_end = tSelec_2

        self.ax.add_patch(
            patches.Rectangle(
                (x_start, 0),  # (x,y)
                x_end-x_start,  # width
                mainChrono.data.max(),  # height
                alpha=0.2
            )
        )

        self.figure.canvas.draw()


    def onSpanMove(self, xmin, xmax):
        if xmin > xmax:
            # swap
            xmax, xmin = xmin, xmax
        #print(xmin, xmax)
        self.view.currentTimeWindow = [xmin, xmax]
        self.controller.update_navigation()
        #self.controller.zoom()

    def onSpanSelect(self, xmin, xmax):
        if xmin > xmax:
            # swap
            xmax, xmin = xmin, xmax
        #print(xmin, xmax)
        self.view.currentTimeWindow = [xmin, xmax]
        self.controller.update_navigation()

    def scrollEvent(self, event):
        if self.ctrl_is_held==True:
            #On zoom
            if event.button == 'up':
                mean = (self.view.currentTimeWindow[0] + self.view.currentTimeWindow[1]) / 2
                delta = self.view.currentTimeWindow[1] - self.view.currentTimeWindow[0]
                self.view.currentTimeWindow[0] = self.view.currentTimeWindow[0] + delta / 4
                self.view.currentTimeWindow[1] = self.view.currentTimeWindow[1] - delta / 4

                # self.zoomOnMainChrono(self.currentTimeWindow[0], self.currentTimeWindow[1])
            elif event.button == 'down':
                mean = (self.view.currentTimeWindow[0] + self.view.currentTimeWindow[1]) / 2
                delta = self.view.currentTimeWindow[1] - self.view.currentTimeWindow[0]
                self.view.currentTimeWindow[0] = mean - delta * 2
                self.view.currentTimeWindow[1] = mean + delta * 2
                self.view.currentTimeWindow[0] = max(0, self.view.currentTimeWindow[0])
                # self.currentTimeWindow[1] = min(self.currentTimeWindow[1], )

            self.controller.update_navigation()
        else:
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

            self.controller.update_navigation()

