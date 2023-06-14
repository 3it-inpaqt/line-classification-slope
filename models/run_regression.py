import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split

from linegeneration.generate_lines import create_image_set
# from plot.lines_visualisation import create_multiplots
# from utils.angle_operations import normalize_angle
from utils.save_model import save_model
from utils.statistics import calculate_std_dev
from utils.settings import settings
from utils.misc import load_list_from_file, renorm_all_tensors, enhance_contrast

# Read data
n = 500  # number of images to create
N = 18  # size of the images (NxN)

# Load patches
# patches = torch.load('./saved/double_dot_patches.pt'),
# n = patches[0].shape[0]
# N = 18

# X, y = torch.load('./saved/double_dot_patches.pt'), [float(x) for x in load_list_from_file('./saved/double_dot_normalized_angles.txt')]
# X, y = torch.load('./saved/double_dot_patches_Dx.pt'), [float(x) for x in load_list_from_file('./saved/double_dot_normalized_angles.txt')]
# X, y = torch.load('./saved/single_dot_patches_rot.pt'), [float(x) for x in load_list_from_file('./saved/single_dot_normalized_angles_rot.txt')]
# X, y = torch.load('./saved/double_dot_patches_resample_20.pt'), [float(x) for x in load_list_from_file('./saved/double_dot_normalized_angles_resample_20.txt')]
# n, N = X.shape
# X = (renorm_all_tensors(X.reshape((n, settings.patch_size_x, settings.patch_size_y)), True)).reshape((n, N))
# print(X.shape)


# Read Synthetic data
X, y = create_image_set(n, N)  # n images of size NxN
X = X.reshape(n, N*N)
# y_normalized = normalize_angle(y)

# fig, axes = create_multiplots(X, y, number_sample=16)
# plt.tight_layout()
# plt.show()

# train-test split for model evaluation
# X_train, X_test, y_train, y_test = train_test_split(X, y_normalized, train_size=0.7, shuffle=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# Convert to 2D PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# Define the model
model = nn.Sequential(
        nn.Linear(N, 24),
        nn.LeakyReLU(),
        nn.Linear(24, 12),
        nn.LeakyReLU(),
        nn.Linear(12, 6),
        nn.ReLU(),
        nn.Linear(6, 1)
    )

# loss function and optimizer
learning_rate = 1e-5
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

n_epochs = 1000   # number of epochs to run
batch_size = 16  # size of each batch
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

            # X_batch = X_batch.flatten(1)  # flatten array for matrix multiplication
            # forward pass
            y_pred = model(X_batch)
            # print('Y pred: ', y_pred)
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
y_pred = model(X_test)
# print("y test: ", type(y_test), y_test.shape)
# print("y pred: ", type(y_pred), y_pred.shape)
std = calculate_std_dev(y_pred, y_test)
# Save the state dictionary
save_model(model, 'best_model_synthetic_LeakyReLU_Dx_low_lr')

# Plot accuracy
fig, ax = plt.subplots()
# plt.suptitle('Training on the experimental patches (DQD)')
ax.set_title(f'Training on the derivative of patches (regression) \n Learning rate: {learning_rate} | Epochs: {n_epochs}')
print("MSE: %.4f" % best_mse)
print("RMSE: %.4f" % np.sqrt(best_mse))
plt.xlabel('Epoch')
plt.ylabel('Mean Square Error (MSE)')
plt.plot(history)

# Add a text box to the plot
textstr = '\n'.join((
    r'$MSE = %.4f$' % (best_mse, ),
    r'$RMSE = %.4f$' % (np.sqrt(best_mse), ),
    r'$\sigma = %.2f$' % (std, )
))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.9, 0.9, textstr, transform=ax.transAxes, fontsize=14, ha='right', va='top', bbox=props)

plt.show()
