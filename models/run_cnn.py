import copy
import matplotlib.pyplot as plt
from numpy import sqrt, inf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn import CNN
from linegeneration.generate_lines import create_image_set
from utils.logger import logger
from plot.lines_visualisation import create_multiplots
from utils.angle_operations import normalize_angle
from utils.misc import load_list_from_file
from utils.save_model import save_model
from utils.statistics import calculate_std_dev


# Set hyperparameters
batch_size = 32
learning_rate = 0.0001
num_epochs = 200

# Initialize model
kernel_size_conv = 4
kernel_size_maxpool = 2
network = CNN(kernel_size_conv)
criterion = nn.MSELoss()  # loss function
optimizer = optim.Adam(network.parameters(), lr=learning_rate)  # optimizer
logger.info('Model has been initialized')

# Load data
# X, y = torch.load('./saved/double_dot_patches_cnn_Dx.pt'), [float(x) for x in load_list_from_file('./saved/double_dot_normalized_angles.txt')]
X_exp = torch.load('./saved/double_dot_patches_cnn_Dx.pt')

n = X_exp.shape[0]
N = 18

# Read Synthetic data
sigma = 0.9
anti_alias = False
X, y = create_image_set(n, N, sigma, anti_alias)  # n images of size NxN
# y_normalized = normalize_angle(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# Convert to 2D PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # adds a dimension of size 1 at index 1, reshaped to have a size of [n, 1, N, N]
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)  # adds a dimension of size 1 at index 1, reshaped to have a size of [n, 1, N, N]
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
logger.info('Dataset has been set up successfully')

# Move network and data tensors to device
device = network.device
network.to(device)
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

# We use the pre-defined number of epochs to determine how many iterations to train the network on
batch_start = torch.arange(0, len(X_train), batch_size)
# Hold the best model
best_mse = inf   # init to infinity
best_weights = None
history = []

# Initialize the progress bar
# logger.info('Starting training')
pbar = tqdm(range(num_epochs), desc="Training Progress", unit="epoch")

for epoch in range(num_epochs):
    network.train()  # prepare model for training
    # Load in the data in batches using the train_loader object
    for start in batch_start:
        # Take a batch
        X_batch = X_train[start:start + batch_size]
        y_batch = y_train[start:start + batch_size]

        # Forward pass
        y_pred = network(X_batch)
        loss = criterion(y_pred, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate accuracy at the end of each epoch
    network.eval()
    y_pred = network(X_test)

    mse = criterion(y_pred, y_test)
    mse = float(mse)
    history.append(mse)

    # Update the progress bar description
    # pbar.update(1)
    pbar.set_postfix({"MSE": mse})

    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(network.state_dict())

# Close the progress bar
pbar.close()
# logger.info('Training is ended')

# Restore model and return best accuracy
network.load_state_dict(best_weights)
y_pred = network(X_test)
# print("y test: ", type(y_test), y_test.shape)
# print("y pred: ", type(y_pred), y_pred.shape)
std = calculate_std_dev(y_pred, y_test)

# Save the state dictionary
model_name = f'best_model_cnn_synthetic_gaussian{sigma}_kernel{kernel_size_conv}.pt'
if anti_alias:
    model_name = f'best_model_cnn_synthetic_gaussian{sigma}_kernel{kernel_size_conv}_aa.pt'
# save_model(network, model_name)

# Plot accuracy
fig, ax = plt.subplots()
title = '\n'.join((
    r'CNN Training on the synthetic patches (gaussian blur $\sigma = %.4f $)' % (sigma, ),
    f'Kernel size: {kernel_size_conv} | Batch size: {batch_size} | Epochs: {num_epochs} | lr: {learning_rate}'
))

ax.set_title(title, fontsize=12)
print("MSE: %.4f" % best_mse)
print("RMSE: %.4f" % sqrt(best_mse))
print("STD: % .4f" % std)
ax.set_xlabel('Epoch')
ax.set_ylabel('Mean Square Error (MSE)')
ax.plot(history)

# Add a text box to the plot
textstr = '\n'.join((
    r'$MSE = %.4f$' % (best_mse, ),
    r'$RMSE = %.4f$' % (sqrt(best_mse), ),
    r'$\sigma = %.2f$' % (std, )
))

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.9, 0.9, textstr, transform=ax.transAxes, fontsize=14, ha='right', va='top', bbox=props)
plt.savefig(f".\saved\plot\{model_name.removesuffix('.pt')}_MSE.png")
plt.show()


# Plot some lines and patches
if torch.cuda.is_available():
    y_pred_numpy = y_pred.cpu().cpu().detach().numpy()
else:
    y_pred_numpy = y_pred.cpu().detach().numpy()

fig1, axes1 = create_multiplots(X_test.cpu(), y_test.cpu(), y_pred_numpy, number_sample=16)
plt.tight_layout()
plt.savefig(f".\saved\plot\{model_name.removesuffix('.pt')}_patches.png")
plt.show()
