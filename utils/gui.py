"""
This is an attempt to create a GUI in order to change parameters more easily. However, this method is slightly obsolete,
since the 'settings.yaml' file handles this issue much better. The user is free to use either method. Note this file is
not working because of subprocess error. Feel free to fix it or not.
"""

import os
import tkinter as tk
from tkinter import ttk
import tkinter.font as font
import torch.nn as nn
import subprocess

from utils.logger import logger


class ScriptGUI:
    def __init__(self, master):
        self.master = master
        master.title("Script Parameters")

        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()

        # Calculate the x and y coordinates of the top-left corner of the window
        x = int((screen_width - 800) / 2)
        y = int((screen_height - 600) / 2)

        # Set the size and position of the window
        self.master.geometry("800x600+{}+{}".format(x, y))

        # Prevent the widgets from resizing the frame
        self.master.grid_propagate(False)

        # Handle styling for the different element
        font_size = 12
        font_type = "CMU Bright"
        font_bold = "CMU Serif"
        font_italic = "CMU Classical Serif"

        style_exit = ttk.Style()  # create a ttk.Style object for exit button
        style_exit.configure('Red.TButton',
                        foreground='red', font=(font_type, font_size))  # configure the style to set the background color of the button to red

        style_button = ttk.Style()
        style_button.configure('Custom.TButton', font=(font_type, font_size))

        style_radio = ttk.Style()  # create a custom style for the Radiobutton widget
        style_radio.configure("Custom.TRadiobutton", font=(font_type, font_size))

        style_dropdown = ttk.Style()  # create a custom style for the dropdown menu widget
        style_dropdown.configure("Custom.TMenubutton", font=(font_italic, font_size))

        style_label = ttk.Style()  # create a custom style for the label widget
        style_label.configure("Custom.TLabel", font=(font_bold, font_size))

        style_tab = ttk.Style()
        style_tab.configure("Custom.TNotebook.Tab", font=(font_bold, font_size + 3))
        # font_tab = font.Font("Arial", size=30, weight="bold")  # font tab setting
        # font_button = font.Font("Arial", size=20)  # font button setting

        # Create variables for the parameters
        self.run_name = tk.StringVar(self.master, value="experimental")  # run type experimental by default

        self.x_files = ["./saved/" + f for f in os.listdir("./saved/") if f.endswith(".pt")]  # load tensor files
        self.y_files = ["./saved/" + f for f in os.listdir("./saved/") if f.endswith(".txt")]  # load txt files with angles
        self.x_var = tk.StringVar(value=self.x_files[0])
        self.y_var = tk.StringVar(value=self.y_files[0])

        self.script_name = tk.StringVar(self.master, value="./models/run_regression.py")  # by default will run regression task

        self.learning_rate = tk.StringVar(self.master, value="0.001")
        self.loss_fn = nn.MSELoss()  # by default use MSE loss
        self.loss_name = tk.StringVar(self.master, value="MSE")
        self.n_epochs = tk.StringVar(self.master, value="100")
        self.batch_size = tk.StringVar(self.master, value="16")

        # Create a notebook to organize the categories
        self.notebook = ttk.Notebook(self.master, style="Custom.TNotebook")
        self.notebook.pack(padx=10, pady=10)

        # Create the "Setup" category
        self.setup_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.setup_frame, text="Setup")

        # Run name buttons
        self.run_name_label = ttk.Label(self.setup_frame, text="Run name:", style="Custom.TLabel")
        self.run_name_label.grid(row=0, column=0, padx=5, pady=5)
        self.experimental_button = ttk.Radiobutton(self.setup_frame, text="Experimental", variable=self.run_name, value="experimental", style="Custom.TRadiobutton")
        self.experimental_button.grid(row=0, column=1, padx=5, pady=5)
        self.synthetic_button = ttk.Radiobutton(self.setup_frame, text="Synthetic", variable=self.run_name, value="synthetic", style="Custom.TRadiobutton")
        self.synthetic_button.grid(row=0, column=2, padx=5, pady=5)

        # Load files buttons
        self.x_label = ttk.Label(self.setup_frame, text="X file:", style="Custom.TLabel")
        self.x_label.grid(row=1, column=0, padx=5, pady=5)
        self.x_dropdown = ttk.OptionMenu(self.setup_frame, self.x_var, *self.x_files, style="Custom.TMenubutton")
        self.x_dropdown.grid(row=1, column=1, padx=5, pady=5)

        self.y_label = ttk.Label(self.setup_frame, text="Y file:", style="Custom.TLabel")
        self.y_label.grid(row=2, column=0, padx=5, pady=5)
        self.y_dropdown = ttk.OptionMenu(self.setup_frame, self.y_var, *self.y_files, style="Custom.TMenubutton")
        self.y_dropdown.grid(row=2, column=1, padx=5, pady=5)

        # Create the "Network settings" category
        self.network_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.network_frame, text="Network settings")

        # Setup network task type
        self.script_name_label = ttk.Label(self.network_frame, text="Script name:", style="Custom.TLabel", justify="left")
        self.script_name_label.grid(row=0, column=0, padx=5, pady=5)
        self.cnn_button = ttk.Radiobutton(self.network_frame, text="CNN", variable=self.script_name, value="./models/run_cnn.py", style="Custom.TRadiobutton")
        self.cnn_button.grid(row=0, column=1, padx=5, pady=5)
        self.feedforward_button = ttk.Radiobutton(self.network_frame, text="Feedforward", variable=self.script_name, value="./models/run_regression.py", style="Custom.TRadiobutton")
        self.feedforward_button.grid(row=0, column=2, padx=5, pady=5)

        self.learning_rate_label = ttk.Label(self.network_frame, text="Learning rate:", style="Custom.TLabel", justify="left")
        self.learning_rate_label.grid(row=1, column=0, padx=5, pady=5)
        self.learning_rate_entry = ttk.Entry(self.network_frame, textvariable=self.learning_rate)
        self.learning_rate_entry.grid(row=1, column=1, padx=5, pady=5)

        self.loss_fn_label = ttk.Label(self.network_frame, text="Loss function:", style="Custom.TLabel", anchor="w")
        self.loss_fn_label.grid(row=2, column=0, padx=5, pady=5)

        loss_function = ["MSE", "SmoothL1Loss", "MAE"]
        # Create a StringVar object to hold the selected option
        selected_option = tk.StringVar(self.master, value=loss_function[0])

        self.loss_fn_dropdown = ttk.OptionMenu(self.network_frame, self.loss_name, selected_option.get(), *loss_function, command=self.set_loss_name, style="Custom.TMenubutton")
        self.loss_fn_dropdown.grid(row=2, column=1, padx=5, pady=5)

        self.n_epochs_label = ttk.Label(self.network_frame, text="Number of epochs:", style="Custom.TLabel")
        self.n_epochs_label.grid(row=3, column=0, padx=5, pady=5)
        self.n_epochs_entry = ttk.Entry(self.network_frame, textvariable=self.n_epochs)
        self.n_epochs_entry.grid(row=3, column=1, padx=5, pady=5)

        self.batch_size_label = ttk.Label(self.network_frame, text="Batch size:", style="Custom.TLabel")
        self.batch_size_label.grid(row=4, column=0, padx=5, pady=5)
        self.batch_size_entry = ttk.Entry(self.network_frame, textvariable=self.batch_size)
        self.batch_size_entry.grid(row=4, column=1, padx=5, pady=5)

        # Create a button to submit the parameters
        self.submit_button = ttk.Button(self.master, text="Submit", command=self.run_script, style="Custom.TButton")
        self.submit_button.pack(padx=10, pady=10)

        # Create the exit button
        self.exit_button = ttk.Button(self.master, text="Exit", style='Red.TButton', command=self.master.destroy)
        self.exit_button.pack(padx=10, pady=10)

    def set_loss_name(self, value):
        if value == "MSE":
            self.loss_name.set("MSE")
            self.loss_fn = nn.MSELoss()
        elif value == "SmoothL1Loss":
            self.loss_name.set("SmoothL1")
            self.loss_fn = nn.SmoothL1Loss()
        elif value == "MAELoss":
            self.loss_name.set("MAE")
            self.loss_fn = nn.L1Loss()

    def run_script(self):
        arg1 = self.run_name.get()
        arg2 = self.x_var.get()
        arg3 = self.y_var.get()
        arg4 = self.script_name.get()
        arg5 = self.learning_rate.get()
        arg6 = self.loss_fn
        arg7 = self.n_epochs.get()
        arg8 = self.batch_size.get()

        try:
            subprocess.run(["./venv/Scripts/python.exe", arg4, "--run_name", arg1, "--X_path", arg2, "--y_path", arg3, "--learning_rate", arg5, "--loss_fn", str(arg6), "--n_epochs", arg7, "--batch_size", arg8])
        except:
            # logger.error("Error", "Failed to run script")
            logger.error("Error: %s", "Failed to run script", exc_info=1)


root = tk.Tk()
my_gui = ScriptGUI(root)
root.mainloop()
