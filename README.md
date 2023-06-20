# Line classification slope

Regression task to identify angles on charge stability diagrams of double quantum dots. However, you can also apply this
model to single dot, make sure you change the setting accordingly.

# Created files

* `data\` : Contains diagrams data (**should be downloaded by the user**) and generated cache files
* `out\ ` : Generated directory that contains run results log and plots if `run_name` setting field is defined
* `settings.yaml` : Project configuration file (**should be created by the user**)

# Settings

Create a file `settings.yaml` to override settings documented in `utils\settings.py`.

### For example:

```yaml
# Related to the diagrams
run_name: 'tmp'
research_group: 'eva_dupont_ferrier'
pixel_size: 0.002
dot_number: 2
logger_console_level: info

# Related to the simulation
x_path: './saved/double_dot_patches_Dx_normalized.pt'
y_path: './saved/double_dot_normalized_angles.txt'

model_type: 'FF'
loss_fn: 'SmoothL1Loss'
learning_rate: 0.001
n_epochs: 1200   # number of epochs to run
batch_size: 10  # size of each batch
kernel_size_conv:  4  # for convolution

# Related to synthetic data
synthetic: False
n_synthetic: 500  # number of synthetic data to create
anti_alias: False
sigma: 0.1
```

From there, you can decide what model to use (feed-forward, CNN, etc.), and set the different hyperparameters. 

# Files structure

* `classes\` : Custom classes and data structure definition
  * `data_structure.py`: bunch of dataclasses and enumerations to structure information and simplify code
  * `diagram.py` : abstract class that describe the interface of a diagram.
  * `diagram_offline.py` : handle the diagram data and its annotations 
  * `diagram_ndot_offline.py` : handle the diagram data and its annotations for N dots
  * `qdsq.py` : Quantum Dots Stability Diagrams (QDSD) dataset, transition line classification task.

* `linegeneration\` : Custom classes and methods to handle lines annotation
  * `annotatediagram.py` : Methods to fetch diagrams annotations and plot them
  * `generate_lines.py` :  Generate synthetic patches (binary images with lines)

* `models\` : Neural network models
  * `model.py` : simple class to generate a NN model
  * `cnn.py` : simple class to generate a CNN model
  * `run_cnn.py` : perform convolution task and save the model generated
  * `run_regression.py` : perform regression task and save the model generated

* `plot\` : Generate figures of diagrams
  * `data.py` : plot the interpolated image
  * `line_visualisation.py` : generate figures with several plots to see different lines orientation (synthetic diagrams)

* `utils\` : Miscellaneous utility code
  * `angle_operations.py` : different methods to calculate angles, fetch them from a file, etc.
  * `misc.py` : miscellaneous scripts (random numbers, fetching from file, etc.)
  * `output.py` : prepare the output directory when loading diagrams
  * `save_model.py`: save NN model after training
  * `settings.py` : storing all settings for this program with default values
  * `statistics.py` : miscellaneous code to compute statistic

# Workflow

If you wish to simply run the program, then use `run.py` after setting proper parameters in the `settings.yaml` file. 
This allows you to be more flexible. However, you can also proceed like explained below.

## Loading diagrams

You first need to load the diagrams from the files. `load_dataset.py` will extract the data from the interpolated csv folder
you supposedly added to a folder called data, along with the json file containing the labels' information. Furthermore, 
the data are then converted to pytorch tensors and stored in a folder (`.\saved`) that you need to create. 

Below is an example of patches with lines drawn on top to better visualise them (in this case the derivative of the diagrams is used):
![image](https://github.com/3it-inpaqt/line-classification-slope/assets/86256324/db24c29f-580a-48b4-8f99-dd66d22bf49a)


## Generate a model

You then need to generate a model using `run_regression.py` or `run_cnn.py`. The data structure changes whether you perform the regression
or the convolution task. When loading the diagrams you have to make sure to set up the tensor accordingly. Make sure you have set up the 
right directory to store the output file. This script generates a dataset itself to train the network on. The model is stored in `.\saved\model`.

## Feedforward

After training your network, you can use `test_network.py` to test the network on the data. Make sure whenever you change 
the network structure in `run_regression.py` to also change it in `model.py` for compatibility. Otherwise, you will get 
layer size related errors. The file also generates a plot to see if the predicted angle is correct, but I suggest you
compute the standard deviation to know if the network works as intended. 

The plot created can look like this (regression example):

### Training period
![image](https://github.com/3it-inpaqt/line-classification-slope/assets/86256324/05540df2-7cb9-481c-b414-95b09a96c1eb)

### Prediction on some patches
![image](https://github.com/3it-inpaqt/line-classification-slope/assets/86256324/75c01784-4ab3-4d92-8d41-4ad01df5cc76)


## Switching between branches

It can be interesting to switch between the `double-dot` branch and the `main` branch. The point is to train the network
on synthetic data and then test it on experimental one. You could also fit tune the network this way. Switching between
branches is easy since they can generate independent files (tensors and NN models). I decided to not merge them for this
very reason.
