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

    def __init__(self, master_frame, view, controller, figsize, dpi, createSpan=True, createCursors=False):
        self.masterFrame = master_frame
        self.view = view
        self.controller = controller
        self.appearanceParam = view.appearenceParam

        self.ctrl_is_held = False
        self.shift_is_held = False

        self.data_x = None
        self.data_y = None
        self.data_fit = None
        self.data_residual = None

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

    def __init__(self, master_frame, view, controller, figsize, dpi):
        super().__init__(master_frame, view, controller, figsize, dpi)
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


