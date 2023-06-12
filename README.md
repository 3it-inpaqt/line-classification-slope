# Line classification slope

Machine learning algorithm to identify angles of lines in 2D binary images. The goal is to later apply to more complex images like stability diagrams of DQD.

# Files structure

* `classes\` : Custom classes and data structure definiton
  * `data_structure.py`: bunch of dataclasses and enumerations to structure information and simplify code

* `linegeneration\` : Custom classes and methods to handle lines annotation
  * `annotatediagram.py` : Methods to fetch diagrams annotations and plot them
  * `generatelines.py` :  Generate synthetic patches (binary images with lines)

* `models\` : Neural network models
  * `model.py` : simple class to generate a NN model
  * `simple_network.py` : perform regression task and save the model generated

* `plot\` : Create a multiplot to geberate figures with different lines orientation and vizualise them
* `utils\` : Miscellaneous utility code
  * `angleoperations.py` : different methods to calculate angles, fetch them from a file, etc.
  * `direandfile.py` : directory and file operations related script
  * `savemodel.py`: save NN model after training
  * `statistics.py` : miscellaneous code to compute statistic

# Workflow

## Generate a model

You first need to generate a model using `simple_network.py`. Make sure you have set up the right directory to store the output file. This script generates a dataset itself to train the network on.

The MSE should look like this with respect to the epoch:
![image](https://github.com/3it-inpaqt/line-classification-slope/assets/86256324/d0fa7c54-b185-4a42-9325-feac852dfa92)


## Feedforward

After training your network, you can use `run.py` to test it on a new dataset. Make sure whenever you change the network structure in `simple_network.py` to also change it in `model.py` for compatibility. Otherwise you will get layer size related errors. The file also generates a plot to see if the predicted angle is correct, but I suggest you compute the standard deviation to know if the network works as intended. 

You should obtain the following diagram:
![image](https://github.com/3it-inpaqt/line-classification-slope/assets/86256324/7a59e1b3-8859-40b3-a614-2ef729f570f3)
