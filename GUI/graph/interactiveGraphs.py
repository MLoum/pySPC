import tkinter as tk
from tkinter import ttk


#from pylab import *
import matplotlib.pyplot as plt
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
from matplotlib.widgets import Cursor
from matplotlib.widgets import MultiCursor






class InteractiveGraph():

    def __init__(self, masterFrame, view, controller, figsize, dpi, createSpan=True, createCursors=False):
        self.masterFrame = masterFrame
        self.view = view
        self.controller = controller
        self.appearenceParam = view.appearenceParam

        self.ctrl_is_held = False
        self.shift_is_held = False

        self.data = None

        self.frame= tk.Frame(self.masterFrame)
        self.frame.pack(side="top", fill="both", expand=True)

        self.figure = plt.Figure(figsize=figsize, dpi=dpi)
        self.ax = self.figure.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)

    def show(self):
        self.canvas.show()


    def copyData(self, target):
        pass

    def createCallBacks(self):
        #https://stackoverflow.com/questions/18141069/matplotlib-how-to-pick-up-shift-click-on-figure
        self.figure.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.figure.canvas.mpl_connect('key_release_event', self.on_key_release)

        self.figure.canvas.mpl_connect('scroll_event', self.scrollEvent)
        self.figure.canvas.mpl_connect('button_press_event', self.button_press_event)
        self.figure.canvas.mpl_connect('button_release_event', self.button_release_event)
        self.figure.canvas.mpl_connect('motion_notify_event', self.motion_notify_event)



    def createWidgets(self):
        # set useblit True on gtkagg for enhanced performance
        # TODO right button click ?
        # TODO Cursors
        self.spanSelec = SpanSelector(self.ax, self.onSpanSelect, 'horizontal', useblit=True,
                                    onmove_callback=self.onSpanMove,
                                    rectprops=dict(alpha=0.5, facecolor='red'), span_stays=True)

    def on_key_press(self, event):
        print(event.key)
        if event.key == 'ctrl':
            self.ctrl_is_held = True

    def on_key_release(self, event):
        if event.key == 'ctrl':
            self.ctrl_is_held = False



    #Virtual methods

    def onSpanSelect(self, xmin, xmax):
        pass
    def onSpanMove(self, xmin, xmax):
        pass
    def scrollEvent(self, event):
        pass
    def button_press_event(self, event):
        pass
    def motion_notify_event(self, event):
        pass
    def button_release_event(self, event):
        pass



# class Graph_navigation(InteractiveGraph):
#     """
#     Il s'agit d'un graphe montrant la totalité du fichier.
#     On peut cliquer et selctionner des zones pour naviguer dans le fichier
#     """
#
#     def __init__(self, masterFrame, view, controller, figsize, dpi):
#         super().__init__(masterFrame, view, controller, figsize, dpi)
#         self.ax.axis('off')
#         self.figure.tight_layout()
#         self.createCallBacks()
#         self.createWidgets()
#
#     def plot(self, mainChrono, tSelec_1=0, tSelec_2=-1):
#
#         self.ax.clear()
#
#         # reduce nb of point to 1000 (approximative size in pixel
#         skipsize = int(mainChrono.nbOfBin / 1000)
#         idx = np.arange(0, len(mainChrono.data), skipsize)
#
#         plot = mainChrono.data[idx]
#         plotX = mainChrono.xAxis[idx]
#
#         self.ax.plot(plotX, plot)
#         self.ax.set_xlim(mainChrono.xAxis[0], mainChrono.xAxis[-1])
#
#         x_start = tSelec_1
#         if tSelec_2 == -1:
#             x_end = mainChrono.xAxis.max()
#         else:
#             x_end = tSelec_2
#
#         self.ax.add_patch(
#             patches.Rectangle(
#                 (x_start, 0),  # (x,y)
#                 x_end-x_start,  # width
#                 mainChrono.data.max(),  # height
#                 alpha=0.2
#             )
#         )
#
#         self.figure.canvas.draw()
#
#
#     def onSpanMove(self, xmin, xmax):
#         if xmin > xmax:
#             # swap
#             xmax, xmin = xmin, xmax
#         print(xmin, xmax)
#         self.view.currentTimeWindow = [xmin, xmax]
#         self.controller.zoom()
#
#     def onSpanSelect(self, xmin, xmax):
#         if xmin > xmax:
#             # swap
#             xmax, xmin = xmin, xmax
#         print(xmin, xmax)
#         self.view.currentTimeWindow = [xmin, xmax]
#         self.controller.zoom()
#
#     def scrollEvent(self, event):
#         if self.ctrl_is_held==True:
#             #On zoom
#             if event.button == 'up':
#                 mean = (self.view.currentTimeWindow[0] + self.view.currentTimeWindow[1]) / 2
#                 delta = self.view.currentTimeWindow[1] - self.view.currentTimeWindow[0]
#                 self.view.currentTimeWindow[0] = self.view.currentTimeWindow[0] + delta / 4
#                 self.view.currentTimeWindow[1] = self.view.currentTimeWindow[1] - delta / 4
#
#                 # self.zoomOnMainChrono(self.currentTimeWindow[0], self.currentTimeWindow[1])
#             elif event.button == 'down':
#                 mean = (self.view.currentTimeWindow[0] + self.view.currentTimeWindow[1]) / 2
#                 delta = self.view.currentTimeWindow[1] - self.view.currentTimeWindow[0]
#                 self.view.currentTimeWindow[0] = mean - delta * 2
#                 self.view.currentTimeWindow[1] = mean + delta * 2
#                 self.view.currentTimeWindow[0] = max(0, self.view.currentTimeWindow[0])
#                 # self.currentTimeWindow[1] = min(self.currentTimeWindow[1], )
#
#             self.controller.zoom()
#         else:
#             #On translate
#             if event.button == 'up':
#                 delta = self.view.currentTimeWindow[1] - self.view.currentTimeWindow[0]
#                 self.view.currentTimeWindow[0] += delta / 2
#                 self.view.currentTimeWindow[1] += delta / 2
#
#                 # TODO vérfier borne depassement fichier.
#
#             elif event.button == 'down':
#
#                 delta = self.view.currentTimeWindow[1] - self.view.currentTimeWindow[0]
#                 self.view.currentTimeWindow[0] -= delta / 2
#                 self.view.currentTimeWindow[1] -= delta / 2
#                 if self.view.currentTimeWindow[0] < 0:
#                     self.view.currentTimeWindow[0] = 0
#
#             self.controller.zoom()
#


class SnaptoCursor(object):
    """
    Like Cursor but the crosshair snaps to the nearest x,y point
    For simplicity, I'm assuming x is sorted
    """

    def __init__(self, ax, x, y):
        self.ax = ax
        self.lx = ax.axhline(color='k')  # the horiz line
        self.ly = ax.axvline(color='k')  # the vert line
        self.x = x
        self.y = y
        # text location in axes coords
        self.txt = ax.text(0.7, 0.9, '', transform=ax.transAxes)

    def mouse_move(self, event):

        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata

        indx = np.searchsorted(self.x, [x])[0]
        x = self.x[indx]
        y = self.y[indx]
        # update the line positions
        self.lx.set_ydata(y)
        self.ly.set_xdata(x)

        self.txt.set_text('x=%1.2f, y=%1.2f' % (x, y))
        print('x=%1.2f, y=%1.2f' % (x, y))
        plt.draw()





class Graph_miniPCH(InteractiveGraph):
    """
    doc todo
    """

    def __init__(self, masterFrame, view, controller, figsize, dpi):
        super().__init__(masterFrame, view, controller, figsize, dpi)
        #self.ax.axis('off')
        self.figure.tight_layout()

        self.createCallBacks()
        self.createWidgets()

    def plot(self, PCH):
        print(np.size(PCH.xAxis))
        print(np.size(PCH.data))

        # reduce nb of point to 1000 (approximative size in pixel
        skipsize = int(PCH.nbOfBin / 1000)
        idx = np.arange(0, len(PCH.data), skipsize)

        plot = PCH.data[idx]
        plotX = PCH.xAxis[idx]

        self.ax.plot(plot, plotX)

        self.figure.canvas.draw()


    def onSpanMove(self, xmin, xmax):
        pass

    def onSpanSelect(self, xmin, xmax):
        pass

    def scrollEvent(self, event):
        pass


