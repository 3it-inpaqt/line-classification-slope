import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from linegeneration.generate_lines import create_image_set
from models.model import loss_fn_dic, AngleNet
from plot.lines_visualisation import create_multiplots
from utils.save_model import save_model
from utils.statistics import calculate_std_dev, accuracy
from utils.settings import settings
from utils.misc import load_list_from_file, dec_to_sci, resymmetrise_tensor

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})


def main():

    if settings.synthetic:
        # Read Synthetic data
        N = settings.patch_size_x
        n = settings.n_synthetic
        X, y = create_image_set(n, N, gaussian_blur=settings.sigma, background=settings.background, aa=settings.anti_alias)  # n images of size NxN
        X = X.reshape(n, N*N)
        # Set title for loss evolution with respect to epoch and model name
        model_name = f'best_model_synthetic_regression_{settings.loss_fn}_beta{int(settings.beta)}_batch{settings.batch_size}_epoch{settings.n_epochs}'
        # custom_suffix = '_new_loss'
        # if len(custom_suffix) > 0:
        #     model_name += custom_suffix
        ax_title = f'Learning rate: {settings.learning_rate} | Epochs: {settings.n_epochs} | Batch: {settings.batch_size}'

    else:
        X_path = settings.x_path
        y_path = settings.y_path
        X, y = torch.load(X_path), [float(x) for x in load_list_from_file(y_path)]
        # Set title for loss evolution with respect to epoch and model name
        model_name = f'best_model_experimental_Dx_regression_{settings.loss_fn}_beta{settings.beta}_batch{settings.batch_size}_epoch{settings.n_epochs}'
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
    model = AngleNet(input_size, settings.n_hidden_layers)  # CHANGE THE STRUCTURE OF THE NETWORK IN THE 'ANGLENET' CLASS

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
            # y_pred_prime = y_pred.clone()
            # y_pred_prime = torch.where(y_pred_prime > settings.threshold_loss, y_pred_prime - 0.5, y_pred_prime)

            loss1 = criterion(y_pred, y_batch)
            loss2 = criterion(resymmetrise_tensor(y_pred), y_batch)

            loss = torch.min(loss1, loss2)
            # loss_prime = criterion(y_pred_prime, y_batch)

            # final_loss = min(loss, loss_prime)
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
        y_pred_test = model(X_test)
        # TODO find another way to calculate the loss
        loss1 = criterion(y_pred, y_batch)
        loss2 = criterion(resymmetrise_tensor(y_pred), y_batch)

        loss = torch.min(loss1, loss2)

        loss = float(loss)
        history.append(loss)
        pbar.set_postfix({name_criterion: loss})

        if loss < best_loss:
            best_loss = loss
            best_weights = copy.deepcopy(model.state_dict())
            y_pred_best = model(X_test)
            # std = min(calculate_std_dev(y_pred_test, y_test), calculate_std_dev(y_pred_prime, y_test))
            # acc = max(accuracy(y_test, y_pred_test), accuracy(y_test, y_pred_prime))
            std = calculate_std_dev(y_pred_test, y_test)
            acc = accuracy(y_test, y_pred_test)

    pbar.close()

    # Restore model and return best accuracy
    model.load_state_dict(best_weights)
    # Save the state dictionary
    # save_model(model, model_name)

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
        r'$\sigma = {{{deviation}}} $'.format(deviation=dec_to_sci(std), ),
        r'$Accuracy = {{{acc}}}$'.format(acc=acc, ),
        f'{settings.n_hidden_layers} hidden layers',
        f'{settings.loss_fn}',
        r'$\beta = {{{beta}}}$'.format(beta=settings.beta, )
    ))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.9, 0.9, textstr, transform=ax.transAxes, fontsize=14, ha='right', va='top', bbox=props)

    # plt.savefig(f".\saved\plot\{model_name}_loss.png")
    plt.show()

    # Plot some lines and patches
    # if torch.cuda.is_available():
    #     y_pred_numpy = y_pred_best.cpu().detach().numpy()
    # else:
    #     y_pred_numpy = y_pred_best.cpu().detach().numpy()

    # fig1, axes1 = create_multiplots(X_test, y_test, y_pred_numpy, number_sample=9, cmap='gray')
    # plt.tight_layout()
    # plt.savefig(f".\saved\plot\{model_name}_patches.png")
    # plt.show()


if __name__ == '__main__':
    main()

