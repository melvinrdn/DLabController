import tkinter as tk


class MainWindow:
    def __init__(self, master):
        self.master = master

        self.entry = tk.Entry(master, width=20)
        self.entry.grid(row=0, column=0)
        self.update_button = tk.Button(master, text='Update', command=self.update_label)
        self.update_button.grid(row=0, column=1)
        self.label = tk.Label(master, text='Entered Number: ')
        self.label.grid(row=1, column=0)
        self.open_window_button = tk.Button(master, text='Open Window', command=self.open_detached_window)
        self.open_window_button.grid(row=2, column=0)

    def update_label(self):
        entered_number = self.entry.get()
        try:
            number = float(entered_number)
            self.label.config(text=f'Entered Number: {number}')
            if hasattr(self, 'detached_window'):
                self.detached_window.update_detached_label(number)
        except ValueError:
            self.label.config(text='Invalid input')

    def open_detached_window(self):
        self.detached_window = DetachedWindow(self.master, self.update_main_label)
        self.update_label()

    def update_main_label(self, number):
        self.entry.delete(0, tk.END)
        self.entry.insert(0, str(number))
        self.label.config(text=f'Entered Number: {number}')


class DetachedWindow:
    def __init__(self, master, callback):
        self.master = master
        self.callback = callback
        self.detached_window = tk.Toplevel(master)
        self.detached_window.title('Detached Window')

        self.detached_entry = tk.Entry(self.detached_window, width=20)
        self.detached_entry.grid(row=0, column=0)
        self.update_main_button = tk.Button(self.detached_window, text='Update Main', command=self.update_main_label)
        self.update_main_button.grid(row=1, column=0)
        self.detached_label = tk.Label(self.detached_window, text='Entered Number: ')
        self.detached_label.grid(row=2, column=0)
        self.close_window_button = tk.Button(self.detached_window, text='Close Window',
                                             command=self.close_detached_window)
        self.close_window_button.grid(row=3, column=0)

    def update_detached_label(self, number):
        self.detached_entry.delete(0, tk.END)
        self.detached_entry.insert(0, str(number))
        self.detached_label.config(text=f'Entered Number: {number}')

    def update_main_label(self):
        entered_number = self.detached_entry.get()
        try:
            number = float(entered_number)
            self.callback(number)
        except ValueError:
            pass

    def close_detached_window(self):
        self.detached_window.destroy()


root = tk.Tk()
main_window = MainWindow(root)
root.mainloop()
