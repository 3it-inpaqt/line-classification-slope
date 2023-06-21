"""
This script is a recreation, from scratch of the software LabelBox. This was made before I could access the labeled
diagrams. You do not have to take it into account, but it's still useful.
Make sure to input the right directory name
"""


import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk

from utils.angle_operations import calculate_angle
from utils.misc import list_files
from utils.settings import settings


# Initialize variables
x1, y1, x2, y2 = None, None, None, None
positions_list, angles_list = [], []
lines = []

# Load diagram to use
run_name = settings.run_name
directory = f""  # CHANGE DIRECTORY PATH HERE
files_list = list_files(directory)
current_image_index = 0
file = files_list[current_image_index]


def reset():
    """
    Function to handle reset button click
    :return:
    """
    global x1, y1, x2, y2, positions_list, angles_list, points, line, ax
    ax.cla()  # Clear current Axes object
    ax.axis('off')  # Hide axis
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


def change_background():
    """
    Function to change background image
    """
    global ax, line, points, fig

    # get selected file from dropdown
    selected_file = bg_var.get()

    # clear current Axes object
    ax.clear()
    ax = fig.add_subplot(111)

    # load selected file as image
    img = plt.imread(selected_file)
    plt.axis('off')

    # create new Axes object with the selected image as background
    ax.imshow(img)
    ax.set_xlim([0, img.shape[1]])
    ax.set_ylim([0, img.shape[0]])
    ax.axis('off')
    # redraw canvas
    fig.canvas.draw()


# Create tkinter window
root = tk.Tk()
root.title("Image Annotation")

# Create matplotlib figure
fig = Figure(figsize=(5, 5), dpi=100)
ax = fig.add_subplot(111)

ax.set_aspect('equal')  # Set aspect ratio to be equal
ax.set_title('Click to add points and create lines')  # Set title
ax.axis('off')

line, = ax.plot([], [], 'r-', lw=2)
points = ax.scatter([], [], c='b', marker='o')

# Create canvas to display figure
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Create button to reset points
reset_button = tk.Button(root, text="Reset Points", command=reset)
reset_button.pack(side=tk.RIGHT)

# Create label to display angle
angle_var = tk.StringVar()
angle_var.set("Angle: ")
angle_label = tk.Label(root, textvariable=angle_var)
angle_label.pack(side=tk.BOTTOM)

# Create dropdown menu to select background image
bg_var = tk.StringVar(value=files_list[0])
bg_dropdown = tk.OptionMenu(root, bg_var, *files_list)
bg_dropdown.pack(side=tk.TOP)

# create button to change background image
bg_button = tk.Button(root, text="Change Background", command=change_background)
bg_button.pack(side=tk.TOP)

# Bind mouse click event to canvas
canvas.mpl_connect('button_press_event', on_click)

root.mainloop()