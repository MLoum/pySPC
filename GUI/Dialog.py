import tkinter as tk
from tkinter import ttk

#https://stackoverflow.com/questions/673174/file-dialogs-of-tkinter-in-python-3
from tkinter import filedialog, messagebox, simpledialog


class generatePoissonianDialog(simpledialog.Dialog):

    def body(self, master):

        ttk.Label(master, text="Time (s):").grid(row=0)
        ttk.Label(master, text="Count Per Second:").grid(row=1)

        self.e1 = ttk.Entry(master)
        self.e2 = ttk.Entry(master)

        #default value
        self.e1.insert(tk.END, '10.0')
        self.e2.insert(tk.END, '1000.0')

        self.e1.grid(row=0, column=1)
        self.e2.grid(row=1, column=1)

        self.result = None
        return self.e1 # initial focus

    def apply(self):
        first = float(self.e1.get())
        second = float(self.e2.get())
        self.result =  first, second