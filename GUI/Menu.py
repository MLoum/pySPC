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
        self.create_menu()

    def create_menu(self):
        self.menuSystem = tk.Menu(self.master)

        # FILE#############
        self.menu_file = tk.Menu(self.menuSystem, tearoff=0)


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

        self.menu_file.add_command(label='New / Clear', underline=1, accelerator="Ctrl+n", command=self.clear)
        #self.master.bind_all("<Control-o>", self.askOpenSPC_file)

        self.menu_file.add_command(label='Add Exp', underline=1, accelerator="Ctrl+o", command=self.askOpenSPC_file)
        #self.master.bind_all("<Control-o>", self.askOpenSPC_file)

        self.menuFile_generate = tk.Menu(self.menu_file)

        self.menu_file.add_cascade(label="Generate", menu=self.menuFile_generate)
        self.menuFile_generate.add_command(label='Poissonian Noise', command=self.ask_generate_poissonian_noise)

        self.menu_file.add_command(label='Save State', underline=1, accelerator="Ctrl+s", command=self.save_state)
        #self.master.bind_all("<Control-s>", self.saveState)
        self.menu_file.add_command(label='Load State', underline=1, accelerator="Ctrl+l", command=self.load_state)
        #self.master.bind_all("<Control-l>", self.loadState)

        self.menu_file.add_command(label='Preferences', command=self.openPreferences)
        self.menu_file.add_command(label='Quit', command=self.quit)

        self.menuSystem.add_cascade(label="File", menu=self.menu_file)

        self.menu_debug = tk.Menu(self.menuSystem, tearoff=0)
        self.menu_debug.add_command(label='Get raw macrotimes', underline=1, command=self.get_raw_macrotimes)

        self.menuSystem.add_cascade(label="Debug", menu=self.menu_debug)

        self.master.config(menu=self.menuSystem)

    def get_raw_macrotimes(self):
        self.controller.get_raw_data()


    def quit(self):
        result = messagebox.askquestion("Quit ?", "Are You Sure ?", icon='warning')
        if result == 'yes':
            self.mainGUI.on_quit()


    def clear(self):
        result = messagebox.askquestion("Clear experiments", "Are You Sure ?", icon='warning')
        if result == 'yes':
            self.controller.clear_exp()




    def ask_generate_poissonian_noise(self):
        d = generatePoissonianDialog(self.master, title="Generate Poissonian Noise")
        if d.result is not None:
            time_s, count_per_secound = d.result

            self.controller.generate_poisson_noise_file(time_s, count_per_secound)


    def save_state(self):
        filePath = filedialog.asksaveasfile(title="Save State", initialdir=self.mainGUI.saveDir)
        if filePath == None or filePath.name == '':
            return None
        self.controller.save_state(filePath.name)

    def load_state(self):
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
            if extension not in (".spc", ".pt3", ".ttt", ".ptn"):
                messagebox.showwarning("Open file", "The file has not the correct .spc, .pt3, .ttt, extension. Aborting")
                return None
            else:
                self.mainGUI.saveDir = os.path.split(filePath)[0]
                self.controller.open_SPC_File(filePath)
                return filePath


