import tkinter as tk
from tkinter import ttk


class PopupQuestion:
    """
    Simple popup box to ask a question with two possible answers.
    Two possible answers can be specified.
    Once the popup is closed, callback is called with the answer as parameter.
    """

    def __init__(self, callback, title, question, answer_true, answer_false):
        """
        Parameters
        ----------
        callback : function
            A function to be called with the user's answer as a boolean argument.
        title : str
            The title of the popup window.
        question : str
            The question to display in the popup window.
        answer_true : str
            The label for the 'True' button.
        answer_false : str
            The label for the 'False' button.
        """
        self.popup = tk.Toplevel()
        self.popup.title(title)
        self.popup.protocol("WM_DELETE_WINDOW", self.true_pressed)
        self.callback = callback
        self.answer = None
        label = ttk.Label(self.popup, text=question, width=60, wraplength=340)
        label.pack(side="top", fill="x", pady=10)
        BT = ttk.Button(self.popup, text=answer_true, command=self.true_pressed)
        BT.pack(side="left", padx=10, pady=10)
        BC = ttk.Button(self.popup, text='Cancel', command=self.cancelled)
        BC.pack(side="right", padx=10, pady=10)
        BF = ttk.Button(self.popup, text=answer_false, command=self.false_pressed)
        BF.pack(side="bottom", padx=10, pady=10)
        self.popup.mainloop()

    def true_pressed(self):
        """
        Called when the 'True' button is pressed.
        Destroys the popup window and calls the callback with a value of True.
        """
        self.popup.destroy()
        self.callback(True)

    def false_pressed(self):
        """
        Called when the 'False' button is pressed.
        Destroys the popup window and calls the callback with a value of False.
        """
        self.popup.destroy()
        self.callback(False)

    def cancelled(self):
        """
        Called when the 'Cancel' button is pressed.
        Destroys the popup window.
        """
        self.popup.destroy()
