import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from linegeneration.generate_lines import create_image_set
from plot.lines_visualisation import create_multiplots
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
X, y = create_image_set(n, N, aa=True)  # n images of size NxN
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
        nn.Linear(N*N, 24),  # change N to N*N if you use synthetic data
        nn.LeakyReLU(),
        nn.Linear(24, 6),
        nn.LeakyReLU(),
        nn.Linear(6, 1),
    )

# loss function and optimizer
learning_rate = 1e-5
# loss_fn = nn.MSELoss()  # mean square error
loss_fn = nn.SmoothL1Loss()  # mean absolute error
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

n_epochs = 1000   # number of epochs to run
batch_size = 16  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)

# Hold the best model
best_mea = np.inf   # init to infinity
best_weights = None
history = []

pbar = tqdm(range(n_epochs), desc="Training Progress", unit="epoch")

for epoch in range(n_epochs):
    model.train()
    for start in batch_start:
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
        # update progress bar
        # tqdm.write("Epoch {}, Batch {}: Loss = {:.4f}".format(epoch, start, loss), end="\r")
    pbar.update(1)
    # evaluate accuracy at end of each epoch
    model.eval()
    X_test = X_test.flatten(1)
    y_pred = model(X_test)

    mea = loss_fn(y_pred, y_test)
    mea = float(mea)
    history.append(mea)
    pbar.set_postfix({"MSE": mea})

    if mea < best_mea:
        best_mea = mea
        best_weights = copy.deepcopy(model.state_dict())
        std = calculate_std_dev(y_pred, y_test)

pbar.close()
# restore model and return best accuracy
model.load_state_dict(best_weights)
# y_pred = model(X_test)
# print("y test: ", type(y_test), y_test.shape)
# print("y pred: ", type(y_pred), y_pred.shape)
# std = calculate_std_dev(y_pred, y_test)
# Save the state dictionary
save_model(model, 'best_model_synthetic_LeakyReLU_Dx_SmoothL1Loss')

# Plot accuracy
fig, ax = plt.subplots()
# plt.suptitle('Training on the experimental patches (DQD)')
ax.set_title(f'Training on the derivative of patches (regression) \n Learning rate: {learning_rate} | Epochs: {n_epochs}')
print("Loss: %.4f" % best_mea)
# print("RMAE: %.4f" % np.sqrt(best_mea))
plt.xlabel('Epoch')
# plt.ylabel('Mean Absolute Error (MAE)')
# plt.ylabel('Mean Square Error (MSE)')
plt.ylabel('Loss (SmoothL1Loss)')

plt.plot(history)

# Add a text box to the plot
textstr = '\n'.join((
    r'$Loss = %.4f$' % (best_mea, ),
    r'$RMSE = %.4f$' % (np.sqrt(best_mea), ),
    r'$\sigma = %.2f$' % (std, )
))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.9, 0.9, textstr, transform=ax.transAxes, fontsize=14, ha='right', va='top', bbox=props)

plt.show()

# Plot some lines and patches
# if torch.cuda.is_available():
#     y_pred_numpy = y_pred.cpu().detach().numpy()
# else:
#     y_pred_numpy = y_pred.cpu().detach().numpy()

fig1, axes1 = create_multiplots(X_test.detach().numpy(), y_test.detach().numpy(), y_pred.detach().numpy(), number_sample=16)
plt.tight_layout()
# plt.savefig(f".\saved\plot\{model_name.removesuffix('.pt')}_patches.png")
plt.show()

