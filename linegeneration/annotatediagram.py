import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import numpy as np

from utils.angleoperations import calculate_angle


# initialize variables
x1, y1, x2, y2 = None, None, None, None
positions_list, angles_list = [], []


def reset():
    # function to handle reset button click
    global x1, y1, x2, y2, positions_list, angles_list, points, line
    # clear lists
    positions_list = []
    angles_list = []
    # remove line and points from plot
    if line is not None:
        line.remove()
    if points is not None:
        points.remove()
    # redraw empty plot
    line, = ax.plot([], [], color='red')
    points = None
    # reset points
    x1, y1, x2, y2 = None, None, None, None
    # clear angle label
    angle_var.set("")
    fig.canvas.draw()


def on_click(event):
    """
    Function to handle on_click event
    :param event:
    :return:
    """
    global x1, y1, x2, y2, positions_list, angles_list, points
    if x1 is None and y1 is None:
        # first click
        x1, y1 = event.xdata, event.ydata
    elif x2 is None and y2 is None:
        # second click
        x2, y2 = event.xdata, event.ydata
        # draw line and calculate angle
        line.set_data([x1, x2], [y1, y2])
        angle = calculate_angle(x1, y1, x2, y2)
        angle_var.set("Angle: {:.2f}".format(angle))
        # add to lists
        positions_list.append((x1, y1, x2, y2))
        angles_list.append(angle)
        # remove points and update scatter plot
        if points is not None:
            points.remove()
        points = ax.scatter([x1, x2], [y1, y2], color='red')
        # reset points
        x1, y1, x2, y2 = None, None, None, None
    fig.canvas.draw()


# create tkinter window
root = tk.Tk()
root.title("Image Annotation")

# create matplotlib figure
fig = Figure(figsize=(5, 5), dpi=100)
ax = fig.add_subplot(111)
line, = ax.plot([], [], 'r-', lw=2)
points = ax.scatter([], [], c='b', marker='o')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# create canvas to display figure
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# create button to reset points
reset_button = tk.Button(root, text="Reset Points", command=reset)
reset_button.pack(side=tk.BOTTOM)

# create label to display angle
angle_var = tk.StringVar()
angle_var.set("Angle: ")
angle_label = tk.Label(root, textvariable=angle_var)
angle_label.pack(side=tk.BOTTOM)

# bind mouse click event to canvas
canvas.mpl_connect('button_press_event', on_click)

root.mainloop()
