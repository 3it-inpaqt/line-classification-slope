from models.network import AngleNet
from linegeneration.generatelines import create_batch
from models.train import train

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


""" Read data """

N = 18
n = 10000

# Generate lines
batch, angle_list = create_batch(n, N)
# Convert angle value from [0,pi] to [0,1]
normalize_angle = [theta/np.pi for theta in angle_list]

""" Define model """
model = AngleNet(N)

# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 100  # number of epochs to run
batch_size = 10  # size of each batch

# defining the loss function
criterion = nn.CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

print(model)

""" Prepare data for training """
train_x, val_x, train_y, val_y = train_test_split(batch, normalize_angle, test_size=0.1)
print((train_x.shape, len(train_y)), (val_x.shape, len(val_y)))

# Converting training images into torch format
n_train = int(0.9*n)
train_x = train_x.reshape(n_train, 1, N, N)
train_x = torch.tensor(train_x, dtype=torch.float32)

# Converting the target into torch format
train_y = torch.tensor(np.array(train_y), dtype=torch.float32)

# Converting rest of data
val_x = torch.tensor(val_x, dtype=torch.float32)
val_y = torch.tensor(np.array(val_y), dtype=torch.float32)

# shape of training data
print(train_x.shape, train_y.shape)

""" Run model """

train_losses, val_losses = train(model, optimizer, criterion, train_x, train_y, val_x, val_y, n_epochs)

""" plotting the training and validation loss """

fig = plt.figure()
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.show()

""" Prediction for training set """

with torch.no_grad():
    output = model(train_x.cuda())

softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)

# accuracy on training set
accuracy_score(train_y, predictions)