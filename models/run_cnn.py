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
from models.model import loss_fn_dic
from utils.save_model import save_model
from utils.settings import settings
from utils.statistics import calculate_std_dev, accuracy

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})


def main():

    if settings.synthetic:
        # Read Synthetic data
        N = settings.patch_size_x
        n = settings.n_synthetic
        X, y = create_image_set(n, N, background=True, aa=True)  # n images of size NxN

        image_set_test, angles_test = create_image_set(n, N, background=True, aa=True)
        fig, axes = create_multiplots(image_set_test, angles_test, number_sample=n, cmap='copper')
        plt.show()

        # X = X.reshape(n, N*N)
        # # Set title for loss evolution with respect to epoch and model name
        # model_name = f'best_model_experimental_Dx_convolution_{settings.loss_fn}_batch{settings.batch_size}_epoch{settings.n_epochs}_kernel{settings.kernel_size_conv}'
        #
        # ax_title = f'Training on the synthetic patches (convolution) \n Learning rate: {settings.learning_rate} | Epochs: {settings.n_epochs} | Batch: {settings.batch_size} | Kernel: {settings.kernel_size_conv}'

    else:
        X_path = settings.x_path
        y_path = settings.y_path
        X, y = torch.load(X_path), [float(x) for x in load_list_from_file(y_path)]
        # Set title for loss evolution with respect to epoch and model name
        model_name = f'best_model_experimental_Dx_convolution_{settings.loss_fn}_batch{settings.batch_size}_epoch{settings.n_epochs}_kernel{settings.kernel_size_conv}'
        ax_title = f'Training on the experimental patches (convolution + Dx) \n Learning rate: {settings.learning_rate} | Epochs: {settings.n_epochs} | Batch: {settings.batch_size} | Kernel: {settings.kernel_size_conv}'

    # Load data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

    # Convert to 2D PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
    logger.info('Dataset has been set up successfully')

    # Set hyperparameters
    batch_size = settings.batch_size
    learning_rate = settings.learning_rate
    num_epochs = settings.n_epochs

    # Initialize model
    kernel_size_conv = settings.kernel_size_conv
    network = CNN(kernel_size_conv)

    name_criterion = settings.loss_fn
    criterion = loss_fn_dic[name_criterion]

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)  # optimizer
    logger.info('Model has been initialized')

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
    best_loss = inf   # init to infinity
    best_weights = None
    history = []

    # Initialize the progress bar
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
        pbar.update(1)
        pbar.set_postfix({"Loss": mse})

        if mse < best_loss:
            best_loss = mse
            best_weights = copy.deepcopy(network.state_dict())
            acc = accuracy(y_test, y_pred)

    # Close the progress bar
    pbar.close()
    # logger.info('Training is ended')

    # Restore model and return best accuracy
    network.load_state_dict(best_weights)
    y_pred = network(X_test)
    std = calculate_std_dev(y_pred, y_test)

    # if anti_alias:
    #     model_name = f'best_model_cnn_synthetic_gaussian{sigma}_kernel{kernel_size_conv}_aa.pt'
    # save_model(network, model_name, 'cnn')

    # Plot accuracy
    fig, ax = plt.subplots()
    ax.set_title(ax_title, fontsize=12)
    print("MSE: %.4f" % best_loss)
    print("STD: % .4f" % std)
    print(f"Accuracy: {acc}")
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'Loss ({name_criterion})')
    ax.plot(history)

    # Add a text box to the plot
    textstr = '\n'.join((
        r'$Loss = %.4f$' % (best_loss, ) + f' ({settings.loss_fn})',
        r'$\sigma = %.4f$' % (std, ),
        r'$Accuracy = %.4f$' % (acc, )
    ))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.9, 0.9, textstr, transform=ax.transAxes, fontsize=14, ha='right', va='top', bbox=props)
    # plt.savefig(f".\saved\plot\{model_name}.png")
    plt.show()

    # Plot some lines and patches
    # if torch.cuda.is_available():
    #     y_pred_numpy = y_pred.cpu().cpu().detach().numpy()
    # else:
    #     y_pred_numpy = y_pred.cpu().detach().numpy()

    # fig1, axes1 = create_multiplots(X_test.cpu(), y_test.cpu(), y_pred_numpy, number_sample=16)
    # plt.tight_layout()
    # plt.savefig(f".\saved\plot\{model_name}_patches.png")
    # plt.show()


if __name__ == '__main__':
    main()
