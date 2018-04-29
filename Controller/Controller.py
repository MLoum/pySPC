#!/usr/bin/env python
#-*- coding: utf-8 -*-

# Note on MVC pattern :
# https://realpython.com/blog/python/the-model-view-controller-mvc-paradigm-summarized-with-legos/
# http://sametmax.com/quest-de-que-mvc-et-a-quoi-ca-sert/


# http://cirw.in//blog/time-to-move-on.html

#De toute façon, aucun MVC n’est parfait, et un peu de vue dégouline parfois sur le contrôleur, un peu de contrôleur coule dans le modèle, ou inversement. Il ne sert à rien d’être un nazi du MVC, c’est une bonne pratique, pas un dogme religieux.

import tkinter as Tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from core  import ExpParam
from core  import Results
from core  import Data
from core  import Experiment

from GUI  import View
from GUI  import guiForFitOperation

import shelve

class Controller():
    def __init__(self):
        self.root = Tk.Tk()
        self.model = Experiment.Experiment()
        self.view = View.View(self.root, self)
        #Has to be done when all have been initialized
        self.view.archi.dock()
        #self.createMenuCommands()
        #self.createShortCut()
        self.root.protocol("WM_DELETE_WINDOW", self.onQuit)  # Exit when x pressed, notice that its the name of the function 'self.handler' and not a method call self.handler()

        # self.view.sidepanel.plotBut.bind("<Button>",self.my_plot)
        # self.view.sidepanel.clearButton.bind("<Button>",self.clear)

    def run(self):
        self.root.title("pySPC")
        self.root.deiconify()
        self.root.mainloop()

############

    def open_SPC_File(self, filePath):
        #filePath = self.view.menu.askOpenFile('Choose the SPC file to analyse (.spc, .pt3, .ttt, ...')
        #TODO test extension
        self.model.newExp("file", [filePath])

        #FIXME le channel 0 is hardcoded
        self.view.currentTimeWindow = [0, self.model.convertTicksInSeconds(self.model.data.channels[0].endTick)*1E6]
        self.view.is_a_FileLoaded = True
        self.view.currentChannel = 0
        self.setChronoBinSize_s(0.01)

        self.updateStatus()
        self.updateNavigation()

        #self.view.archi.navigation_area.timeZoom.graph_timeZoom.plot(self.model.results.mainChronogram)
        #self.view.plotMainPCH(self.model.results.mainPCH)
        #self.view.archi.navigation_area.graph_navigation.plot(self.model.results.mainChronogram)

    def generatePoissonNoiseFile(self, time_s, count_per_secound):
        self.model.newExp("generate", ["Poisson", time_s, count_per_secound])

        #TODO don't duplicate from open SPC.
        #FIXME le channel 0 is hardcoded
        self.view.currentTimeWindow = [0, self.model.convertTicksInSeconds(self.model.data.channels[0].endTick)*1E6]
        self.view.is_a_FileLoaded = True

        self.updateStatus()
        self.updateNavigation()


############

    def updateAll(self):
        self.updateStatus()
        self.updateNavigation()
        self.updateAnalyze()

    def updateStatus(self):
        if self.view.is_a_FileLoaded == False:
            return

        self.view.archi.status_area.setFileName(self.model.fileName)
        c = self.model.data.channels[self.view.currentChannel]
        self.view.archi.status_area.setNbOfPhotonAndCPS(c.nbOfTick, c.CPS)

    def updateAnalyze(self):
        if self.view.is_a_FileLoaded == False:
            return

        t1_microsec = self.view.currentTimeWindow[0]
        t2_microsec = self.view.currentTimeWindow[1]
        t1_tick, t2_tick = self.model.convertSecondsInTicks(t1_microsec / 1E6), self.model.convertSecondsInTicks(
            t2_microsec / 1E6)
        channel = self.view.currentChannel
        guiGraphResult = self.view.archi.analyze_area.resultArea_gui.graph_results

        if self.view.currentOperation == "macro":
            # FIXME main channel and bin size
            binSize_s = 100

            binInTick = self.model.convertSecondsInTicks(binSize_s)
            self.model.data.results.chronograms[channel] = self.model.data.chronogram(channel, t1_tick, t2_tick,
                                                                                      binInTick)
            #self.view.plotMainChronogram(self.model.results.chronograms[channel])
            #TODO plot dans la fenetre result.

        elif self.view.currentOperation == "micro":
            self.model.data.microTimeLifeTime(channel, t1_tick, t2_tick)
            guiGraphResult.plot("lifetime", self.model.results.lifeTimeMeasurements[channel])

        elif self.view.currentOperation == "FCS":
            gui = self.view.archi.analyze_area.FCS_TimeAnalyze_gui
            #FIXME recuperer les data
            channel1 = channel2 = channel

            max_correlTime_ms = float(gui.maxCorrelTime_sv.get())
            self.model.data.FCS(channel1, channel2, t1_tick, t2_tick, max_correlTime_ms)
            guiGraphResult.plot("FCS", self.model.results.FCS_Measurements[channel])

        elif self.view.currentOperation == "DLS":
            gui = self.view.archi.analyze_area.DLS_TimeAnalyze_gui
            #FIXME recuperer les data
            channel1 = channel2 = channel
            max_correlTime_ms = float(gui.maxCorrelTime_sv.get())
            self.model.data.DLS(channel1, channel2, t1_tick, t2_tick, max_correlTime_ms)
            guiGraphResult.plot("DLS", self.model.results.FCS_Measurements[channel])




    #def updateNavigation(self, channel, t1_microsec, t2_microsec, binSize_s=0.01):
    def updateNavigation(self):
        if self.view.is_a_FileLoaded == False:
            return
        channel = self.view.currentChannel
        t1_microsec, t2_microsec = self.view.currentTimeWindow[0], self.view.currentTimeWindow[1]
        binSize_s = 0.01

        #FIXME main channel and bin size
        t1_tick, t2_tick = self.model.convertSecondsInTicks(t1_microsec/1E6), self.model.convertSecondsInTicks(t2_microsec/1E6)
        binInTick = self.model.convertSecondsInTicks(binSize_s)
        self.model.data.results.chronograms[channel] = self.model.data.chronogram(channel, t1_tick, t2_tick, binInTick)
        self.view.archi.navigation_area.timeZoom.graph_timeZoom.plot(self.model.results.chronograms[channel])

        self.view.archi.navigation_area.graph_navigation.plot(self.model.results.navigationChronogram,
                                                              self.view.currentTimeWindow[0],
                                                              self.view.currentTimeWindow[1])

        self.view.archi.navigation_area.timeZoom.chronoStart_sv.set(str(int(self.view.currentTimeWindow[0] / 1000)))
        self.view.archi.navigation_area.timeZoom.chronoEnd_sv.set(str(int(self.view.currentTimeWindow[1] / 1000)))

    def setChronoBinSize_s(self, binSize_s):
        self.view.archi.navigation_area.timeZoom.binSizeMicros_sv.set(str(binSize_s*1E6))
        self.view.currentBinSize_s = binSize_s

    def changeTab(self):
        if self.view.currentTab == 'macro':
            self.view.currentOperation = "macro"
            self.updateAnalyze()

            #self.drawMainChronogram(self.view.currentChannel, self.view.currentTimeWindow[0], self.view.currentTimeWindow[1], self.view.currentBinSize_s)

        elif self.view.currentTab == 'micro':
            self.view.currentOperation = "micro"
            self.updateAnalyze()

            #self.drawMicroTimeHisto(self.view.currentChannel, self.view.currentTimeWindow[0], self.view.currentTimeWindow[1])

    # def actualizeChronogramm(self):
    #     self.drawMainChronogram(self.view.currentChannel, self.view.currentTimeWindow[0],
    #                             self.view.currentTimeWindow[1], self.view.currentBinSize_s)
    #     self.view.archi.chronoStart_sv.set(str(int(self.view.currentTimeWindow[0] / 1000)))
    #     self.view.archi.chronoEnd_sv.set(str(int(self.view.currentTimeWindow[1] / 1000)))

    # def zoom(self):
    #     if self.view.currentTab == 'macro':
    #         self.drawMainChronogram(self.view.currentChannel, self.view.currentTimeWindow[0],
    #                                 self.view.currentTimeWindow[1], self.view.currentBinSize_s)
    #         self.view.archi.navigation_area.timeZoom.chronoStart_sv.set(str(int(self.view.currentTimeWindow[0] / 1000)))
    #         self.view.archi.navigation_area.timeZoom.chronoEnd_sv.set(str(int(self.view.currentTimeWindow[1] / 1000)))
    #
    #     # elif self.view.currentTab == 'micro':
    #     #     self.drawMicroTimeHisto(self.view.currentChannel, self.view.currentTimeWindow[0],
    #     #                             self.view.currentTimeWindow[1])
    #
    #     self.view.archi.navigation_area.graph_navigation.plot(self.model.results.mainChronogram, self.view.currentTimeWindow[0], self.view.currentTimeWindow[1])


    def createShortCut(self):
        pass





    def setLimX_fit(self, idxStart, idxEnd):
        self.view.archi.analyze_area.FCS_TimeAnalyze_gui.guiForFitOperation_FCS.idx_lim_for_fit_min_sv.set(str(idxStart))
        self.view.archi.analyze_area.FCS_TimeAnalyze_gui.guiForFitOperation_FCS.idx_lim_for_fit_max_sv.set(str(idxEnd))

        self.view.archi.analyze_area.lifeTimeAnalyze_gui.guiForFitOperation_Lifetime.idx_lim_for_fit_min_sv.set(str(idxStart))
        self.view.archi.analyze_area.lifeTimeAnalyze_gui.guiForFitOperation_Lifetime.idx_lim_for_fit_max_sv.set(str(idxEnd))

        self.view.archi.analyze_area.DLS_TimeAnalyze_gui.guiForFitOperation_DLS.idx_lim_for_fit_min_sv.set(str(idxStart))
        self.view.archi.analyze_area.DLS_TimeAnalyze_gui.guiForFitOperation_DLS.idx_lim_for_fit_max_sv.set(str(idxEnd))


    def fit(self, analyzeType,  mode, modelName, params, idxStart=0, idxEnd=-1):
        """
        :param analyzeType: is a string for lifetime FCS DLS ...
        :param mode can be fit, eval or guess
        :param modelName string for the model name
        :param params: list of paramaters for the fit
        xlims_idx : index of the limits of the x axis where to perform the fit.
        :return:
        """
        data, gui, fitPlotMode  = None, None, None

        channel = self.view.currentChannel
        #TODO cursor with fit limits.

        if analyzeType == "lifetime":
            data = self.model.results.lifeTimeMeasurements[channel]
            gui = self.view.archi.analyze_area.lifeTimeAnalyze_gui.guiForFitOperation_Lifetime
            fitPlotMode = "lifetime"
        elif analyzeType == "DLS":
            data = self.model.results.DLS_Measurements[channel]
            gui = self.view.archi.analyze_area.DLS_TimeAnalyze_gui.guiForFitOperation_DLS
            fitPlotMode = "DLS"
        elif analyzeType == "FCS":
            data = self.model.results.FCS_Measurements[channel]
            gui = self.view.archi.analyze_area.FCS_TimeAnalyze_gui.guiForFitOperation_FCS
            fitPlotMode = "FCS"

        if data != None :
            if mode == "eval":
                data.setModel(modelName)
                data.setParams(params)
                data.evalParams(idxStart, idxEnd)

            elif mode == "guess":
                data.setModel(modelName)
                data.guess(idxStart, idxEnd)
                gui.setParamsFromFit(data.params)

            elif mode == "fit":
                data.setModel(modelName)
                data.setParams(params)
                fitResults = data.fit(idxStart, idxEnd)
                #gui.setParamsFromFit(data.params)
                self.view.archi.analyze_area.resultArea_gui.setTextResult(fitResults.fit_report())

            self.view.archi.analyze_area.resultArea_gui.graph_results.plotFit(fitPlotMode, data)




    def shiftIR(self, mainWidth, secondaryWidth, secondaryAmplitude, timeOffset):
        channel = self.view.currentChannel
        lf = self.model.results.lifeTimeMeasurements[channel]
        lf.generateArtificialIR(mainWidth, secondaryWidth, secondaryAmplitude, timeOffset)

    ##############


    def saveState(self, savefilePath):

        self.shelf = shelve.open(savefilePath, 'n')  # n for new

        self.model.saveState(self.shelf)
        self.view.saveState(self.shelf)
        self.shelf.close()

    def loadState(self, loadfilePath):
        self.shelf = shelve.open(loadfilePath)
        self.model.loadState(self.shelf)
        self.view.loadState(self.shelf)
        self.shelf.close()

        self.model.update()
        self.view.update()


    def onQuit(self):
        # paramFile = open('param.ini', 'w')
        # paramFile.write(self.saveDir)
        self.root.destroy()
        self.root.quit()