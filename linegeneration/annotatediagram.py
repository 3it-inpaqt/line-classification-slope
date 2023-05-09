import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import os

from utils.angleoperations import calculate_angle


# Initialize variables
x1, y1, x2, y2 = None, None, None, None
positions_list, angles_list = [], []
lines = []

# Load diagram to use
directory = "C:\\Users\\wilan\\OneDrive\\Documents\\M1 GP\\Stage Sherbrooke\\Data\\interpolated_img\\interpolated_img\\2.0mV\\single\\eva_dupont_ferrier"
os.chdir(directory)

# Set file to write coordinates of line in


def reset():
    """
    Function to handle reset button click
    :return:
    """
    global x1, y1, x2, y2, positions_list, angles_list, points, line, fig, ax
    fig.clf()  # Clear entire figure
    ax = fig.add_subplot(111)  # Add new blank Axes object
    ax.set_xlim([0, 10])  # Set x limits
    ax.set_ylim([0, 10])  # Set y limits
    ax.set_aspect('equal')  # Set aspect ratio to be equal
    ax.set_title('Click to add points and create lines')  # Set title
    x1, y1, x2, y2 = None, None, None, None
    positions_list = []
    angles_list = []
    points = None
    line = None
    fig.canvas.draw()


def on_click(event):
    """
    Function to handle on_click event
    :param event:
    :return:
    """
    global x1, y1, x2, y2, positions_list, angles_list, points, line, points, fig, ax
    if event.inaxes == ax:
        if x1 is None and y1 is None:
            # first click
            x1, y1 = event.xdata, event.ydata
            # draw point on plot
            points, = ax.plot(x1, y1, 'o', color='black')
        elif x2 is None and y2 is None:
            # second click
            x2, y2 = event.xdata, event.ydata
            # draw line on plot
            if line is not None:
                if abs(x2 - x1) > abs(y2 - y1):
                    color = 'red'
                else:
                    color = 'blue'
                line, = ax.plot([x1, x2], [y1, y2], color=color)
            else:
                if abs(x2 - x1) > abs(y2 - y1):
                    color = 'red'
                else:
                    color = 'blue'
                line, = ax.plot([x1, x2], [y1, y2], color=color)

            # calculate angle and add to lists
            angle = calculate_angle(x1, y1, x2, y2)
            positions_list.append([(x1, y1), (x2, y2)])
            angles_list.append(angle)

            # update angle label
            angle_var.set(f"Angle: {angle:.2f} degrees")

            # remove points and update scatter plot
            if points is not None:
                points.remove()
            points = ax.scatter([x1, x2], [y1, y2], color=color)

            # reset points
            x1, y1, x2, y2 = None, None, None, None
        fig.canvas.draw()


# create tkinter window
root = tk.Tk()
root.title("Image Annotation")

# create matplotlib figure
fig = Figure(figsize=(5, 5), dpi=100)
ax = fig.add_subplot(111)


ax.set_xlim([0, 10])  # Set x limits
ax.set_ylim([0, 10])  # Set y limits
ax.set_aspect('equal')  # Set aspect ratio to be equal
ax.set_title('Click to add points and create lines')  # Set title

line, = ax.plot([], [], 'r-', lw=2)
points = ax.scatter([], [], c='b', marker='o')

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
