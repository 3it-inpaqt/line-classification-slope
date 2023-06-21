# Line classification slope

Regression task to identify angles on charge stability diagrams of double quantum dots. However, you can also apply this
model to single dot, make sure you change the setting accordingly.

_**<span style="color:lime;">DISCLAIMER:</span> most of the experimental data used to run the trainings are under NDA and therefore cannot be shared. The plots you might have not correspond to the one I obtained during my internship.**_

# Install

### General information

Required python `>= 3.8` and `pip`

```
pip install -r requirements.txt
```

You then need to download the data set and unzip it in a `data` folder at the root of this project. The mandatory files
are:

* `interpolated_csv.zip`
* `labels.json`

### Using Pycharm

If you are using Pycharm follow these simple steps:

1. On **GitHub**, click the button <span style="color:lime;"><> Code</span> and copy the link
2. On **Pycharm**, go to `Git > Clone`
3. Paste the url of the **GitHub** repository you just copied and click clone.
4. You can now access the files. The GUI should automatically ask you if you want to download the requirements

# Created files

* `data\` : Contains diagrams data (**should be downloaded by the user**) and generated cache files
* `out\ ` : Generated directory that contains run results log and plots if `run_name` setting field is defined
* `settings.yaml` : Project configuration file (**should be created by the user**)

# Settings

_<span style="color:gold;">Note:</span> The settings do not change the structure of the neural network. This has to be
done manually in either `models/model.py` (feed-forward) or `models/cnn.py` (convolution)_

Create a file `settings.yaml` to override settings documented in `utils\settings.py`.

### For example:

The following settings set the dataset as synthetic with argument `synthetic: True`. You should obtain a very small
standard deviation (between `1e-7` and `1e-8`).

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
learning_rate: 0.0001
n_epochs: 2500   # number of epochs to run
batch_size: 18  # size of each batch
kernel_size_conv:  4  # for convolution

# Related to synthetic data
synthetic: True
n_synthetic: 500  # number of synthetic data to create
anti_alias: False
sigma: 0.1
```

From there, you can decide what model to use (feed-forward, CNN, etc.), and set the different hyperparameters. 

# Files structure

### Classes and methods

* `classes\` : Custom classes and data structure definition
  * `data_structure.py`: bunch of dataclasses and enumerations to structure information and simplify code
  * `diagram.py` : abstract class that describe the interface of a diagram.
  * `diagram_offline.py` : handle the diagram data and its annotations 
  * `diagram_ndot_offline.py` : handle the diagram data and its annotations for N dots
  * `qdsq.py` : Quantum Dots Stability Diagrams (QDSD) dataset, transition line classification task.

* `linegeneration\` : Custom classes and methods to handle lines annotation
  * `annotatediagram.py` : Recreate LabelBox in a Tkinter window (<span style="color:gold;">not relevant</span>)
  * `generate_lines.py` :  Generate synthetic patches (binary images with lines)

* `models\` : Neural network models
  * `cnn.py` : simple class to generate a CNN model
  * `model.py` : simple class to generate a NN model
  * `run_cnn.py` : perform convolution task and save the model generated
  * `run_regression.py` : perform regression task and save the model generated

* `plot\` : Generate figures of diagrams
  * `data.py` : plot the interpolated image
  * `line_visualisation.py` : generate figures with several plots to see different lines orientation (synthetic diagrams)

* `utils\` : Miscellaneous utility code
  * `angle_operations.py` : different methods to calculate angles, fetch them from a file, etc.
  * `gui.py`: tkinter interface aimed to change parameters faster (<span style="color:tomato;">not working/won't fix</span>)
  * `logger.py`: handles log methods to display warning or error messages more clearly
  * `misc.py` : miscellaneous scripts (random numbers, fetching from file, etc.)
  * `output.py` : prepare the output directory when loading diagrams
  * `save_model.py`: save NN model after training
  * `settings.py` : storing all settings for this program with default values
  * `statistics.py` : miscellaneous code to compute statistic

### Additional files:

* `load_dataset.py`: Load the dataset and save the tensors and lists in separated files for later use if specified.
* `run.py`: Controls the training of the network. Centralized the script execution (<span style="color:skyblue;">main script</span>)
* `test_network.py`: Train pre-trained network on a dataset (<span style="color:gold;">not relevant</span>)
# Workflow

If you wish to simply run the program, then use `run.py` after setting proper parameters in the `settings.yaml` file and
running `load_dataset.py` if you have experimental diagrams to work with. This allows you to be more flexible. However, 
you can also proceed like explained below.

## Loading diagrams

_<span style="color:gold;">Note:</span> This phase is necessary only if you wish to work with experimental data. You can
skip this step if you only want to use synthetic diagrams._

You first need to load the diagrams from the files. `load_dataset.py` will extract the data from the interpolated csv folder
you supposedly added to a folder called data, along with the json file containing the labels' information. Furthermore, 
the data are then converted to pytorch tensors and stored in a folder (`.\saved`) that you need to create. 

Once the diagrams are loaded, you can access them in the folder `out`. With the annotated diagrams you will also find a
sample of patches with lines drawn on top of them and the associated angles to show you how the angles are calculated.

## Generate a model

_<span style="color:coral;">Warning:</span> The data structure changes whether you perform the regression or the 
convolution task. When loading the diagrams you have to make sure to set up the tensor accordingly. Make sure you have 
set up the right directory to store the output file. This script generates a dataset itself to train the network on. 
The model is stored in `.\saved\model`._

## Feedforward

After training your network, you can use `test_network.py` to test the network on the data. Make sure whenever you change 
the network structure in `run_regression.py` to also change it in `model.py` for compatibility. Otherwise, you will get 
layer size related errors. The file also generates a plot to see if the predicted angle is correct, but I suggest you
compute the standard deviation to know if the network works as intended.

### Training period

During the training period you should see a bar indicating the number of batch already used, and the value of the loss.
Once the training is over, a graph should appear with the evolution of the loss during the training, like the one below:

![best_model_experimental_Dx_regression_SmoothL1Loss_batch32_epoch2500_loss](https://github.com/3it-inpaqt/line-classification-slope/assets/86256324/1412d443-f847-42d9-b173-717fed307ca1)

Sigma indicates the standard deviation, a much more accurate representation of the accuracy of the network in this case.
Be aware that a value of `0.1` is actually not good since the values of the angles are normalised over `2*pi`. You have
to first multiply by `2*pi` to get the standard deviation in radian and then by `180 / pi` to get it in degree.

### Prediction on some patches

Below is an example of angle predictions on few patches from a synthetic dataset.

![best_model_synthetic_regression_SmoothL1Loss_batch18_epoch500_patches](https://github.com/3it-inpaqt/line-classification-slope/assets/86256324/59b7792c-53c9-489b-a742-6319de439049)

## Switching between branches

It can be interesting to switch between the `double-dot` branch and the `pure-synthetic` branch. The point is to train 
the network on synthetic data and then test it on experimental one. You could also fit tune the network this way. 
Switching between branches is easy since they can generate independent files (tensors and NN models). I decided to not 
merge them for this very reason, but also because they differ a lot. Instead, I rebased the `double-dot` branch as the
default one, and rename the `main` branch as `pure-synthetic`.

Note you do not have to switch to the `pure-synthetic` branch to train a neural network with synthetic data. You can
indeed specify this option in `settings.yaml` in the `double-dot` branch.
