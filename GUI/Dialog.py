import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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

class exploreChiSqaureDialog(simpledialog.Dialog):

    def body(self, master):
        self.master_frame = master.master_frame
        self.list_label_string_variable_fit = master.list_label_string_variable_fit
        self.list_label_string_variable_fit.append("None")

        ttk.Label(self.master_frame, text='var').grid(row=0, column=0)
        ttk.Label(self.master_frame, text='limit min').grid(row=0, column=1)
        ttk.Label(self.master_frame, text='limit max').grid(row=0, column=2)

        self.cb_explore_chi_var1_sv = tk.StringVar()
        cb = ttk.Combobox(self.master_frame, width=15, justify=tk.CENTER, textvariable=self.cb_explore_chi_var1_sv,
                          values='', state='readonly')
        cb['values'] = self.list_label_string_variable_fit
        self.cb_explore_chi_var1_sv.set(self.list_label_string_variable_fit[0])
        cb.set(self.list_label_string_variable_fit[0])
        cb.grid(row=1, column=0)

        self.cb_explore_chi_var2_sv = tk.StringVar()
        cb = ttk.Combobox(self.master_frame, width=15, justify=tk.CENTER, textvariable=self.cb_explore_chi_var2_sv,
                          values='', state='readonly')
        cb['values'] = self.list_label_string_variable_fit
        self.cb_explore_chi_var2_sv.set(self.list_label_string_variable_fit[1])
        cb.set(self.list_label_string_variable_fit[1])
        cb.grid(row=2, column=0)

        self.e_lim_min_var1 = ttk.Entry(master)
        self.e_lim_max_var1 = ttk.Entry(master)
        self.e_lim_min_var2 = ttk.Entry(master)
        self.e_lim_max_var2 = ttk.Entry(master)

        self.e_lim_min_var1.grid(row=1, column=1)
        self.e_lim_max_var1.grid(row=1, column=2)
        self.e_lim_min_var2.grid(row=2, column=1)
        self.e_lim_max_var2.grid(row=1, column=2)


        #default value
        # self.e1.insert(tk.END, '100.0')
        # self.e2.insert(tk.END, '100.0')
        # self.e3.insert(tk.END, '100.0')



        self.result = None
        return self.e_lim_min_var1 # initial focus

    def apply(self):
        first = float(self.e_lim_min_var1.get())
        second = float(self.e_lim_max_var1.get())
        third = float(self.e_lim_min_var2.get())
        fourth = float(self.e_lim_max_var2.get())
        fifth = self.cb_explore_chi_var1_sv.get()
        sixth = self.cb_explore_chi_var2_sv.get()
        self.result = first, second, third, fourth, fifth, sixth

class fitIRFDialog(simpledialog.Dialog):

    def body(self, master):

        self.formulaFrame = tk.Frame(master=master)
        self.formulaFrame.grid(row=0, column=0, columnspan=2)

        self.figTex = plt.Figure(figsize=(10, 1), dpi=30, frameon=False)
        self.axTex = self.figTex.add_axes([0, 0, 1, 1])

        self.axTex.axis('off')

        # self.axTex.get_xaxis().set_visible(False)
        # self.axTex.get_yaxis().set_visible(False)

        self.canvasTk = FigureCanvasTkAgg(self.figTex, master=self.formulaFrame)
        self.canvasTk.get_tk_widget().pack(side='top', fill='both', expand=1)

        formula = r"f(x;\mu,\sigma,\lambda) = \frac{\lambda}{2} e^{\frac{\lambda}{2} (2 \mu + \lambda \sigma^2 - 2 x)} \operatorname{erfc} (\frac{\mu + \lambda \sigma^2 - x}{ \sqrt{2} \sigma})"
        formula = "$" + formula + "$"

        self.axTex.clear()
        self.axTex.text(0, 0.2, formula, fontsize=40)
        self.canvasTk.draw()

        ttk.Label(master, text="Sigma (ps):").grid(row=1)
        ttk.Label(master, text="Mu (ps):").grid(row=2)
        ttk.Label(master, text="Lambda (ps):").grid(row=3)

        self.e1 = ttk.Entry(master)
        self.e2 = ttk.Entry(master)
        self.e3 = ttk.Entry(master)

        #default value
        self.e1.insert(tk.END, '100.0')
        self.e2.insert(tk.END, '100.0')
        self.e3.insert(tk.END, '100.0')

        self.e1.grid(row=1, column=1)
        self.e2.grid(row=2, column=1)
        self.e3.grid(row=3, column=1)

        self.result = None
        return self.e1 # initial focus

    def apply(self):
        first = float(self.e1.get())
        second = float(self.e2.get())
        third = float(self.e3.get())
        self.result =  first, second, third