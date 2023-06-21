import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from linegeneration.generate_lines import create_image_set
from models.model import loss_fn_dic, find_loss
from plot.lines_visualisation import create_multiplots
from utils.save_model import save_model
from utils.statistics import calculate_std_dev
from utils.settings import settings
from utils.misc import load_list_from_file, dec_to_sci


def main():

    if settings.synthetic:
        # Read Synthetic data
        N = settings.patch_size_x
        n = settings.n_synthetic
        X, y = create_image_set(n, N, gaussian_blur=settings.sigma, aa=settings.anti_alias)  # n images of size NxN
        X = X.reshape(n, N*N)
        # Set title for loss evolution with respect to epoch and model name
        model_name = f'best_model_synthetic_regression_{settings.loss_fn}_batch{settings.batch_size}_epoch{settings.n_epochs}'
        # custom_suffix = '_new_loss'
        # if len(custom_suffix) > 0:
        #     model_name += custom_suffix
        ax_title = f'Training on the synthetic patches (regression) \n Learning rate: {settings.learning_rate} | Epochs: {settings.n_epochs} | Batch: {settings.batch_size}'

    else:
        X_path = settings.x_path
        y_path = settings.y_path
        X, y = torch.load(X_path), [float(x) for x in load_list_from_file(y_path)]
        # Set title for loss evolution with respect to epoch and model name
        model_name = f'best_model_experimental_Dx_regression_{settings.loss_fn}_batch{settings.batch_size}_epoch{settings.n_epochs}'
        ax_title = f'Training on the experimental patches (regression + Dx) \n Learning rate: {settings.learning_rate} | Epochs: {settings.n_epochs} | Batch: {settings.batch_size}'

    # fig, axes = create_multiplots(X, y, number_sample=16)
    # plt.tight_layout()
    # plt.show()

    # train-test split for model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

    # Convert to 2D PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    # Move network and data tensors to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # X_train_gpu = X_train.to(device)
    # y_train_gpu = y_train.to(device)
    # X_test_gpu = X_test.to(device)
    # y_test_gpu = y_test.to(device)

    input_size = settings.patch_size_x * settings.patch_size_y
    model = nn.Sequential(
            nn.Linear(input_size, 24),
            nn.LeakyReLU(),
            nn.Linear(24, 6),
            nn.LeakyReLU(),
            nn.Linear(6, 1)
        )

    # Loss function and optimizer
    learning_rate = settings.learning_rate
    name_criterion = settings.loss_fn
    criterion = loss_fn_dic[name_criterion]
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    n_epochs = settings.n_epochs   # number of epochs to run
    batch_size = settings.batch_size  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_loss = np.inf   # init to infinity
    best_weights = None
    history = []

    pbar = tqdm(range(n_epochs), desc="Training Progress", unit="epoch")

    for epoch in range(n_epochs):
        model.train()
        for start in batch_start:
            # Take a batch
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            X_batch = X_batch.flatten(1)  # flatten array for matrix multiplication
            # Forward pass
            y_pred = model(X_batch)
            # loss_1 = criterion(y_pred, y_batch)
            # loss_2 = criterion(y_pred - 0.5, y_batch)

            # loss = find_loss(y_pred, y_batch, criterion, threshold=0.49)
            loss = criterion(y_pred, y_batch)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # Update weights
            optimizer.step()

        # Update progress bar
        pbar.update(1)
        # Evaluate accuracy at end of each epoch
        model.eval()
        X_test = X_test.flatten(1)
        y_pred = model(X_test)
        # TODO find another way to calculate the loss
        loss_value = criterion(y_pred, y_test)
        loss_value = float(loss_value)
        history.append(loss_value)
        pbar.set_postfix({name_criterion: loss_value})

        if loss_value < best_loss:
            best_loss = loss_value
            best_weights = copy.deepcopy(model.state_dict())
            y_pred_best = model(X_test)
            std = calculate_std_dev(y_pred, y_test)

    pbar.close()

    # Restore model and return best accuracy
    model.load_state_dict(best_weights)
    # Save the state dictionary
    save_model(model, model_name)

    # Plot accuracy
    fig, ax = plt.subplots()

    ax.set_title(ax_title)
    print("Loss: %.4f" % best_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.plot(history)

    # Add a text box to the plot
    textstr = '\n'.join((
        r'$Best Loss = {{{loss}}}$'.format(loss=dec_to_sci(best_loss), ),
        r'$\sigma = {{{deviation}}} $'.format(deviation=dec_to_sci(std), )
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.9, 0.9, textstr, transform=ax.transAxes, fontsize=14, ha='right', va='top', bbox=props)

    plt.savefig(f".\saved\plot\{model_name}_loss.png")
    plt.show()

    # Plot some lines and patches
    if torch.cuda.is_available():
        y_pred_numpy = y_pred_best.cpu().detach().numpy()
    else:
        y_pred_numpy = y_pred_best.cpu().detach().numpy()

    fig1, axes1 = create_multiplots(X_test, y_test, y_pred_numpy, number_sample=25)
    plt.tight_layout()
    plt.savefig(f".\saved\plot\{model_name}_patches.png")
    plt.show()


if __name__ == '__main__':
    main()

