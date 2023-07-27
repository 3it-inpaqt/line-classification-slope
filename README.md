# Line classification slope

Regression task to identify angles on charge stability diagrams of double quantum dots. However, you can also apply this
model to single dot, make sure you change the setting accordingly.

_**<span style="color:lime;">DISCLAIMER:</span> most of the experimental data used to run the trainings are under NDA and therefore cannot be shared. The plots you might have could not correspond to the one I obtained during my internship. If you wish to obtain the data, please e-mail me on my university mailbox or at chriszethird.contact@gmail.com**_

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

## Folders 

* `data\ ` : Contains diagrams data (**should be downloaded by the user**) and generated cache files
* `out\ ` : Generated directory that contains run results log and plots if `run_name` setting field is defined
* `saved\ ` : Loss evolution plot, CSV files, `.pt` model files, etc. should be stored in this folder.

## CSV files

CSV files contain the settings of each run you will perform, given they were the best runs, with the best loss and
standard deviation. These files are stored in `saved\csv_files`. For each loss function, one file is created. Make sure
you are working on one specific research group, otherwise you risk mixing all the data. The purpose of these files is to
study the correlation between the hyperparameters and the metrics obtained. You can find a plotting function at the end
of `utils\statistics.py` if you are interested. Simply note this function was not developed further, but can provide
a basis for future work.

# Settings

_<span style="color:gold;">Note:</span> The settings do not change the structure of the neural network. This has to be
done manually in either `models\model.py` (feed-forward) or `models\cnn.py` (convolution)_

Create a file `settings.yaml` to override settings documented in `utils\settings.py`. From there, you can decide what model to use (feed-forward, CNN, etc.), and set the different hyperparameters. 

### Example

Below is an example of the parameters you can set to train a neural network. You will be able to see what values each
parameter can take and what they mean.

```yaml
# Related to the diagrams
run_name: 'tmp'
research_group: 'louis_gaudreau'
pixel_size: 0.001
dot_number: 2
logger_console_level: info

patch_size_x: 18
patch_size_y: 18

# Related to angles calculation
full_circle: False

# Related to the dataset
x_path: './saved/double_dot_louis_gaudreau_populated_patches_normalized_18_18.pt'
y_path: './saved/double_dot_louis_gaudreau_populated_angles_18_18.txt'

dx: False
rotate_patch: False
include_synthetic: True

# Related to the network
run_number: 1
model_type: 'FF'
n_hidden_layers: 2
loss_fn: 'SmoothL1Loss'
beta: 0.06
num_harmonics: 10
use_threshold_loss: True
threshold_loss: 135
learning_rate: 0.0001
n_epochs: 200   # number of epochs to run
batch_size: 18  # size of each batch

kernel_size_conv:  8  # for convolution

# Related to synthetic data
synthetic: False
n_synthetic: 2000  # number of synthetic data to create
anti_alias: True
background: True
sigma: 0.1
minimum_length: 0.95  # minimum length of line to draw on synthetic patches

# Gaussian distribution for synthetic data parameters
mean_gaussian: 0.9
scale_gaussian: 0.05
```

#### Loss function
In this example the `SmoothL1Loss` is used for the training. You can change it by checking the name of available
functions in `models\loss.py` and setting `loss_fn` to one of the following values:

* SmoothL1Loss (*pytorch*)
* MSE (*pytorch*)
* MAE (*pytorch*)
* AngleDiff (*Custom*)
* WeightedSmoothL1 (*Custom + Pytorch*)
* HarmonicMeanLoss (*Custom + Pytorch*)
* HarmonicFunctionLoss (*Custom*)

The name of the function should be given as a *string*. From there two extra parameters should be defined: `beta` for
the *SmoothL1Loss*, *WeightedSmoothL1* and *HarmonicMeanLoss*
(see [definition](https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html)); and `num_harmonics` for 
*HarmonicFunctionLoss*.

#### Model type

Two types of model exists: regression (`'FF'`) and convolution (`'CNN'`). Entering a different name without specifying it
in the `settings.py` file in `utils\ ` will result in an error.

#### Network size

The implementation of the network is really simple, and hard-coded. You can switch to a single to two hidden layers
network with the `n_hidden_layers` parameters. Feel free to modify the layers input size in `model.py`, or add more
options, or simply change the definition of the model. 

#### Loss threshold

It is possible to change the method of loss calculation. The network can indeed take into account two loss values: one
between the prediction and expected value, and one between re-symmetrize prediction and expected value. Then it will
keep the lowest value. To enable this option, set `use_threshold_loss` to `True` and adjust the `threshold_loss` to your
liking. Note it makes more sense to take value above 110° when considering line symmetry.

#### Data augmentation

The main issue with the different dataset is the lack of data, and the unbalanced distribution of angles. To solve this,
one can either rotate randomly chosen patches to get more angles values, or add extra data synthetically generated. This
**or** is not exclusive meaning you can also do both. In my approach I decided to implement patch rotation on one hand,
and synthetic data generation with patches rotation together on the other hand. To enable the first one you'll need to
set `rotate_patch` to `True`, and the second `include_synthetic`. It is preferable you enable one option at a time so
either have `True/False` xor `False/True`. 

_<span style="color:gold;">Note:</span> When synthetic data are generated, the parameters set in the synthetic data 
section of `settings.yaml` will be used._

#### Misc parameters

Here is a list of some interesting but not essential parameters:

* `full_circle` : If `True`, the angles will be calculated between 0° and 360° instead of taking the symmetry into
account
* `dx` : Take the derivative of the patches, used in `load_diagram.py` (make sure to change it according to your
simulation)
* `run_name` : If you set this parameter to `tmp`, the target directory will be reset each time you load the diagrams.
* `run_number` : This parameter sets the amount of time a network will be trained. Note only the best values of standard
deviation will let the program save the model, meaning you can run the model 20 times and only save 3 models, as they
got the best performance.
 
# Files structure

### Classes and methods

_<span style="color:gold;">Note:</span> If a file is not mentioned, consider it is not relevant/useful. So you should
not worry about. Feel free to contact me if you think there is a mistake._


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
  * `loss.py` : definition of custom loss function. Feel free to add your own, simply add it to the dictionary defined
  below the classes' definition.
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
  * `populate.py` : manage data augmentation by creating synthetic data for missing angle values in the dataset
  * `rotation.py` : handle patch and line label rotation 
  * `save_model.py`: save NN model after training
  * `settings.py` : storing all settings for this program with default values
  * `statistics.py` : miscellaneous code to compute statistic

### Additional files:

* `load_dataset.py`: Load the dataset and save the tensors and lists in separated files for later use if specified.
* `run.py`: Controls the training of the network. Centralized the script execution (<span style="color:skyblue;">main 
script</span>)
* `test_network.py`: Train pre-trained network on a dataset

# Workflow

If you wish to simply run the program, then use `run.py` after setting proper parameters in the `settings.yaml` file and
running `load_dataset.py` if you have experimental diagrams to work with. This allows you to be more flexible. However, 
you can also proceed like explained below.

## Loading diagrams

_<span style="color:gold;">Note:</span> This phase is necessary only if you wish to work with experimental data. You can
skip this step if you only want to use synthetic diagrams._

You first need to load the diagrams from the files. `load_dataset.py` will extract the data from the interpolated csv folder
you supposedly added to a folder called data, along with the json file containing the labels' information. Furthermore, 
the data are then converted to pytorch tensors and stored in a folder (`saved\ `) that you need to create. 

Once the diagrams are loaded, you can access them in the folder `out`. With the annotated diagrams you will also find a
sample of patches with lines drawn on top of them and the associated angles to show you how the angles are calculated.

## Generate a model

_<span style="color:coral;">Warning:</span> The data structure changes whether you perform the regression or the 
convolution task. When loading the diagrams you have to make sure to set up the tensor accordingly. Make sure you have 
set up the right directory to store the output file._
 
The model will be stored in a folder located in `saved\ ` named after the `research_group`. In this folder, each model
is sorted after the type of network (regression, convolution) and loss function for more clarity. With each model comes 
a loss evolution plot to summarize the settings and the best loss/standard deviation obtained.

## Feedforward

After training your network, you can use `test_network.py` to test the network on the data. The file also generates a plot to see if the predicted angle is correct, but I suggest you
compute the standard deviation to know if the network works as intended.

### Training period

During the training period you should see a bar indicating the number of batch already used, and the value of the loss.
Once the training is over, a graph should appear with the evolution of the loss during the training, like the one below:

![MicrosoftTeams-image](https://github.com/3it-inpaqt/line-classification-slope/assets/86256324/7193e4c8-2e54-4002-8090-58b3aa719c3e)

You can use these data yourself and try obtaining similar results.

_<span style="color:gold;">Note:</span> Sigma indicates the standard deviation, a much more accurate representation of 
the accuracy of the network.
A value of `0.1` for example is actually not good since the values of the angles are normalised over `2*pi`. Therefore, 
you have to multiply by 360 to get it in degree, giving a standard deviation of 36°._

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
