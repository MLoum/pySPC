import Tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
class Application():
    def __init__(self, master):
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)


        frame2 = Tkinter.Frame(master, height=510, width=770, bg='red')
        frame2.grid(row=0, column=0, sticky='nsew')
        frame2.columnconfigure(0, weight=1)
        frame2.rowconfigure(0, weight=1)

        frame2a = Tkinter.Frame(frame2, height=80, width=770, bg='blue')
        frame2a.grid(row=0, column=0, sticky='nsew')
        frame2a.columnconfigure(0, weight=1)
        frame2a.rowconfigure(0, weight=1)

        frame2b = Tkinter.Frame(frame2, height=410, width=770, bg='green')
        frame2b.grid(row=1, column= 0, sticky='nsew')
        frame2b.columnconfigure(0, weight=1)
        frame2b.rowconfigure(1, weight=1)

        # add plot
        fig = Figure(figsize=(9.5,5.2), facecolor='white')
        fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=frame2b)

        canvas.show()
        canvas.get_tk_widget().pack(side='top', fill='both', expand=1)

if __name__ == '__main__' :

    root = Tkinter.Tk()
    root.geometry("770x510")
    app = Application(root)
    root.mainloop()