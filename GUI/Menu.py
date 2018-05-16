import tkinter as tk
from tkinter import ttk

#https://stackoverflow.com/questions/673174/file-dialogs-of-tkinter-in-python-3
from tkinter import filedialog, messagebox, simpledialog
import os

from .Dialog import generatePoissonianDialog


class Menu():
    def __init__(self, master, mainGUI):
        self.master = master
        self.mainGUI = mainGUI
        self.controller = mainGUI.controller
        self.createMenu()

    def createMenu(self):
        self.menuSystem = tk.Menu(self.master)

        # FILE#############
        self.menuFile = tk.Menu(self.menuSystem, tearoff=0)


        # self.menuFile_open = tk.Menu(self.menuFile)
        #
        # self.menuFile.add_cascade(label="Open", menu=self.menuFile_open)
        # self.menuFile_open.add_command(label='open ONE spectrum', command=self.askOpenOneSpectrum)
        # self.bind_all("<Alt-o>", self.askOpenOneSpectrumShortCut)
        # self.menuFile_open.add_command(label='open a collection', underline=1, accelerator="Ctrl+O",
        # command=self.askOpenCollectionOfSpectra)
        # self.bind_all("<Control-o>", self.askPreProcessCollectionOfSpectraShortCut)

        # self.menuFile_open.add_command(label='preprocess a collection', command=self.askPreProcessCollectionOfSpectra)

        # self.menuFile.add_command(label='Save State', underline=1, accelerator="Ctrl+S", command=self.askSaveState)
        # self.bind_all("<Control-s>", self.askSaveStateShortCut)
        # self.menuFile.add_command(label='Load State', underline=1, accelerator="Ctrl+L", command=self.askLoadState)
        # self.bind_all("<Control-l>", self.askLoadStateShortCut)
        # self.menuFile.add_command(label='Export Igor Current Spectrum', command=self.askExportIgorSpec)
        # self.menuFile.add_command(label='Export NPZ Current Collection', command=self.askExportIgorCollec)
        # self.menuFile.add_command(label='Export Igor all spectra in list', command=self.askExportIgorAllSpectraInList)
        # self.menuFile.add_command(label='Export Igor Mie Calculated Spectrum', command=self.askExportMieIgorSpec)

        #TODO some shortcuts seems to autolaunch at the begining of the soft ???

        self.menuFile.add_command(label='Open', underline=1, accelerator="Ctrl+o", command=self.askOpenSPC_file)
        #self.master.bind_all("<Control-o>", self.askOpenSPC_file)

        self.menuFile_generate = tk.Menu(self.menuFile)

        self.menuFile.add_cascade(label="Generate", menu=self.menuFile_generate)
        self.menuFile_generate.add_command(label='Poissonian Noise', command=self.askGeneratePoissonianNoise)

        self.menuFile.add_command(label='Save State',  underline=1, accelerator="Ctrl+s", command=self.saveState)
        #self.master.bind_all("<Control-s>", self.saveState)
        self.menuFile.add_command(label='Load State', underline=1, accelerator="Ctrl+l", command=self.loadState)
        #self.master.bind_all("<Control-l>", self.loadState)

        self.menuFile.add_command(label='Preferences', command=self.openPreferences)
        self.menuFile.add_command(label='Quit', command=self.quit)

        self.menuSystem.add_cascade(label="File", menu=self.menuFile)


        self.master.config(menu=self.menuSystem)


    def quit(self):
        self.mainGUI.on_quit()


    def askGeneratePoissonianNoise(self):
        d = generatePoissonianDialog(self.master, title="Generate Poissonian Noise")
        if d.result != None:
            time_s, count_per_secound = d.result

            self.controller.generate_poisson_noise_file(time_s, count_per_secound)


    def saveState(self, event):
        filePath = filedialog.askopenfilename(title="Save State", initialdir=self.mainGUI.saveDir)
        if filePath == None or filePath == '':
            return None
        self.controller.save_state(filePath)

    def loadState(self, event):
        filePath = filedialog.askopenfilename(title="Load State", initialdir=self.mainGUI.saveDir)
        if filePath == None or filePath == '':
            return None
        self.controller.load_state(filePath)

    def openPreferences(self, event):
        pass

    def askOpenSPC_file(self):
        filePath = filedialog.askopenfilename(title="Open SPC File")
        #TODO logic in controller ?
        if filePath == None or filePath == '':
            return None
        else:
            extension = os.path.splitext(filePath)[1]
            if extension  not in (".spc", ".pt3", ".ttt"):
                messagebox.showwarning("Open file", "The file has not the correct .spc, .pt3, .ttt, extension. Aborting")
                return None
            else:
                self.mainGUI.saveDir = os.path.split(filePath)[0]
                self.controller.open_SPC_File(filePath)
                return filePath


    # def askOpenFile(self, title_):
    #     #filePath = filedialog.askopenfilename(title=title_, initialdir=self.mainGUI.saveDir) #
    #     #TODO initial directory
    #     filePath = filedialog.askopenfilename(title=title_)  #
    #     #TODO logic in controller ?
    #     if filePath == None or filePath == '':
    #         return None
    #     else:
    #         extension = os.path.splitext(filePath)[1]
    #         if extension  not in (".spc", ".pt3", ".ttt"):
    #             messagebox.showwarning("Open file", "The file has not the correct .spc, .pt3, .ttt, extension. Aborting")
    #             return None
    #         else:
    #             self.saveDir = os.path.split(filePath)[0]
    #             self.controller.open_SPC_File(filePath)
    #             return filePath