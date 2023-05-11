import copy

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split

from linegeneration.generate_lines import create_image_set
from utils.save_model import save_model
from utils.angle_operations import normalize_angle

# Read data
n = 10000  # number of images to create
N = 18  # size of the images (NxN)

# Read data
X, y = create_image_set(n, N)
y_normalized = normalize_angle(y)

# train-test split for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y_normalized, train_size=0.7, shuffle=True)

# Convert to 2D PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# Define the model
model = nn.Sequential(
        nn.Linear(N*N, 24),
        nn.ReLU(),
        nn.Linear(24, 12),
        nn.ReLU(),
        nn.Linear(12, 6),
        nn.ReLU(),
        nn.Linear(6, 1)
    )
# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 100   # number of epochs to run
batch_size = 10  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)

# Hold the best model
best_mse = np.inf   # init to infinity
best_weights = None
history = []

for epoch in range(n_epochs):
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            # take a batch
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            X_batch = X_batch.flatten(1)  # flatten array for matrix multiplication
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # print progress
            bar.set_postfix(mse=float(loss))
    # evaluate accuracy at end of each epoch
    model.eval()
    X_test = X_test.flatten(1)
    y_pred = model(X_test)
    mse = loss_fn(y_pred, y_test)
    mse = float(mse)
    history.append(mse)
    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(model.state_dict())

# restore model and return best accuracy
model.load_state_dict(best_weights)

# Save the state dictionary
save_model(model, 'best_model')

# Plot accuracy
plt.figure(1)
print("MSE: %.4f" % best_mse)
print("RMSE: %.2f" % np.sqrt(best_mse))
plt.xlabel('Epoch')
plt.ylabel('Mean Square Error (MSE)')
plt.plot(history)
# Add a text box to the plot
textstr = f'Best MSE: {best_mse}'
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
plt.text(0.05, 0.95, textstr, fontsize=14, verticalalignment='top', bbox=props)
plt.show()
