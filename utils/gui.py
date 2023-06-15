import os
import tkinter as tk
from tkinter import ttk
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

        # Create variables for the parameters
        self.run_name = tk.StringVar(value="experimental")  # run type experimental by default

        self.x_files = [f for f in os.listdir("./saved/") if f.endswith(".pt")]  # load tensor files
        self.y_files = [f for f in os.listdir("./saved/") if f.endswith(".txt")]  # load txt files with angles
        self.x_var = tk.StringVar(value=self.x_files[0])
        self.y_var = tk.StringVar(value=self.y_files[0])

        self.script_name = tk.StringVar(value="./models/run_regression.py")  # by default will run regression task

        self.learning_rate = tk.StringVar(value="0.001")
        self.loss_fn = nn.MSELoss()  # by default use MSE loss
        self.loss_name = tk.StringVar(value="MSE")
        self.n_epochs = tk.StringVar(value="100")
        self.batch_size = tk.StringVar(value="16")

        # Create a notebook to organize the categories
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(padx=10, pady=10)

        # Create the "Setup" category
        self.setup_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.setup_frame, text="Setup")

        # Run name buttons
        self.run_name_label = ttk.Label(self.setup_frame, text="Run name:")
        self.run_name_label.grid(row=0, column=0, padx=5, pady=5)
        self.experimental_button = ttk.Radiobutton(self.setup_frame, text="Experimental", variable=self.run_name, value="experimental")
        self.experimental_button.grid(row=0, column=1, padx=5, pady=5)
        self.synthetic_button = ttk.Radiobutton(self.setup_frame, text="Synthetic", variable=self.run_name, value="synthetic")
        self.synthetic_button.grid(row=0, column=2, padx=5, pady=5)

        # Load files buttons
        self.x_label = ttk.Label(self.setup_frame, text="X file:")
        self.x_label.grid(row=1, column=0, padx=5, pady=5)
        self.x_dropdown = ttk.OptionMenu(self.setup_frame, self.x_var, *self.x_files)
        self.x_dropdown.grid(row=1, column=1, padx=5, pady=5)

        self.y_label = ttk.Label(self.setup_frame, text="Y file:")
        self.y_label.grid(row=2, column=0, padx=5, pady=5)
        self.y_dropdown = ttk.OptionMenu(self.setup_frame, self.y_var, *self.y_files)
        self.y_dropdown.grid(row=2, column=1, padx=5, pady=5)

        # Create the "Network settings" category
        self.network_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.network_frame, text="Network settings")

        # Setup network task type
        self.script_name_label = ttk.Label(self.network_frame, text="Script name:")
        self.script_name_label.grid(row=0, column=0, padx=5, pady=5)
        self.cnn_button = ttk.Radiobutton(self.network_frame, text="CNN", variable=self.script_name, value="./models/run_cnn.py")
        self.cnn_button.grid(row=0, column=1, padx=5, pady=5)
        self.feedforward_button = ttk.Radiobutton(self.network_frame, text="Feedforward", variable=self.script_name, value="./models/run_regression.py")
        self.feedforward_button.grid(row=0, column=2, padx=5, pady=5)

        self.learning_rate_label = ttk.Label(self.network_frame, text="Learning rate:")
        self.learning_rate_label.grid(row=1, column=0, padx=5, pady=5)
        self.learning_rate_entry = ttk.Entry(self.network_frame, textvariable=self.learning_rate)
        self.learning_rate_entry.grid(row=1, column=1, padx=5, pady=5)

        self.loss_fn_label = ttk.Label(self.network_frame, text="Loss function:")
        self.loss_fn_label.grid(row=2, column=0, padx=5, pady=5)
        self.loss_fn_dropdown = ttk.OptionMenu(self.network_frame, self.loss_name, "MSE", "SmoothL1Loss", "MAE", command=self.set_loss_name)
        self.loss_fn_dropdown.grid(row=2, column=1, padx=5, pady=5)

        self.n_epochs_label = ttk.Label(self.network_frame, text="Number of epochs:")
        self.n_epochs_label.grid(row=3, column=0, padx=5, pady=5)
        self.n_epochs_entry = ttk.Entry(self.network_frame, textvariable=self.n_epochs)
        self.n_epochs_entry.grid(row=3, column=1, padx=5, pady=5)

        self.batch_size_label = ttk.Label(self.network_frame, text="Batch size:")
        self.batch_size_label.grid(row=4, column=0, padx=5, pady=5)
        self.batch_size_entry = ttk.Entry(self.network_frame, textvariable=self.batch_size)
        self.batch_size_entry.grid(row=4, column=1, padx=5, pady=5)

        # Create a button to submit the parameters
        self.submit_button = ttk.Button(self.master, text="Submit", command=self.run_script)
        self.submit_button.pack(padx=10, pady=10)

        # Create the exit button

        style = ttk.Style()  # create a ttk.Style object
        style.configure('Red.TButton', foreground='red')  # configure the style to set the background color of the button to red

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
        arg6 = str(self.loss_fn)
        arg7 = self.n_epochs.get()
        arg8 = self.batch_size.get()

        try:
            subprocess.run(["./venv/scripts/python.exe", arg4, "--run_name", arg1, "--X", arg2, "--y", arg3, "--learning_rate", arg5, "--loss_fn", arg6, "--n_epochs", arg7, "--batch_size", arg8])
        except:
            # logger.error("Error", "Failed to run script")
            logger.error("Error: %s", "Failed to run script", exc_info=1)


root = tk.Tk()
my_gui = ScriptGUI(root)
root.mainloop()
