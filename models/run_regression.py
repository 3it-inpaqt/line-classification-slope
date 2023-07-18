import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from linegeneration.generate_lines import create_image_set
from models.model import AngleNet
from models.loss import loss_fn_dic
# from plot.lines_visualisation import create_multiplots
from utils.angle_operations import normalize_angle
from utils.create_csv import init_csv
from utils.logger import logger
from utils.save_model import save_model
from utils.statistics import calculate_std_dev, accuracy
from utils.settings import settings
from utils.misc import load_list_from_file, resymmetrise_tensor

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})


def main():

    if settings.synthetic:
        # Read Synthetic data
        N = settings.patch_size_x
        n = settings.n_synthetic
        X, y = create_image_set(n, N, gaussian_blur=settings.sigma, background=settings.background,
                                aa=settings.anti_alias)  # n images of size NxN
        X = X.reshape(n, N * N)
        # Set title for loss evolution with respect to epoch and model name
        model_name = f'best_model_synthetic_regression_{settings.loss_fn}_batch{settings.batch_size}_epoch{settings.n_epochs}'
        # custom_suffix = '_new_loss'
        # if len(custom_suffix) > 0:
        #     model_name += custom_suffix
        ax_title = f'Learning rate: {settings.learning_rate} | Epochs: {settings.n_epochs} | Batch: {settings.batch_size} | Threshold: {settings.threshold_loss}°'

    else:
        X_path = settings.x_path
        y_path = settings.y_path
        X, y = torch.load(X_path), [float(x) for x in load_list_from_file(y_path)]
        # Set title for loss evolution with respect to epoch and model name
        model_name = f'experimental_{settings.research_group}_regression_{settings.loss_fn}'
        if settings.loss_fn == 'SmoothL1Loss' or settings.loss_fn == 'WeightedSmoothL1':
            model_name += f'{settings.beta}'
        elif settings.loss_fn == 'HarmonicFunctionLoss':
            model_name += f'{settings.num_harmonics}'

        model_name += f'_batch{settings.batch_size}_epoch{settings.n_epochs}'

        if settings.dx:
            model_name += '_Dx'

        saving_dir = f'./saved/'

        # train-test split for model evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

        # Convert to 2D PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

        # Move network and data tensors to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(device)
        X_train = X_train.to(device=device)
        y_train = y_train.to(device=device)
        X_test = X_test.to(device=device)
        y_test = y_test.to(device=device)

    # -- Testing different parameters to find correlation -- #
    while settings.beta != 0.11:  # CHANGE TO DIFFERENT PARAMETER IF NEEDED
        best_std = np.inf

        for i in range(settings.run_number):
            # Reset the model
            input_size = settings.patch_size_x * settings.patch_size_y
            model = AngleNet(input_size,
                             settings.n_hidden_layers)  # CHANGE THE STRUCTURE OF THE NETWORK IN THE 'ANGLENET' CLASS
            # Move model to cuda
            model = model.to(device=device)

            # Loss function and optimizer
            learning_rate = settings.learning_rate
            name_criterion = settings.loss_fn
            criterion = loss_fn_dic[name_criterion]
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            n_epochs = settings.n_epochs  # number of epochs to run
            batch_size = settings.batch_size  # size of each batch
            batch_start = torch.arange(0, len(X_train), batch_size)
            # Hold the best model
            best_loss = np.inf  # init to 0.04 to look only at very low loss
            best_weights = None
            history = []

            pbar = tqdm(range(n_epochs), desc="Training Progress", unit="epoch")

            for epoch in range(n_epochs):
                model.train()
                for start in batch_start:
                    # Take a batch
                    X_batch = X_train[start:start + batch_size]
                    y_batch = y_train[start:start + batch_size]
                    X_batch = X_batch.flatten(1)  # flatten array for matrix multiplication
                    # Forward pass
                    y_pred = model(X_batch)

                    # Loss
                    if settings.use_threshold_loss:
                        loss1 = criterion(y_pred, y_batch)
                        loss2 = criterion(resymmetrise_tensor(y_pred, normalize_angle(settings.threshold_loss * 2 * np.pi / 180)),
                                          y_batch)
                        loss = torch.min(loss1, loss2)

                    else:
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

                # Loss
                if settings.use_threshold_loss:
                    loss1 = criterion(y_pred, y_test)
                    loss2 = criterion(resymmetrise_tensor(y_pred, normalize_angle(settings.threshold_loss * 2 * np.pi / 180)),
                                      y_test)
                    loss = torch.min(loss1, loss2)
                else:
                    loss = criterion(y_pred, y_test)
                loss = float(loss)
                history.append(loss)
                pbar.set_postfix({name_criterion: loss})

                if loss < best_loss:
                    best_loss = loss
                    best_weights = copy.deepcopy(model.state_dict())
                    y_pred_best = model(X_test)
                    std = calculate_std_dev(y_pred_best, y_test)
                    acc = accuracy(y_test, y_pred_best)

            pbar.close()

            # Restore model and return best accuracy
            model.load_state_dict(best_weights)

            # Save the state dictionary if the model is better than the previous one
            if std < best_std:
                best_std = std

                print('best std: ', best_std)

                save_model(model, filename=model_name, directory_path=saving_dir, loss_history=history, best_loss=best_loss, accuracy=acc, standard_deviation=best_std)
                init_csv(loss=best_loss, std_dev=best_std)
                logger.info(f'Run {i+1} ended.')

            # Plot some lines and patches
            # if torch.cuda.is_available():
            #     y_pred_numpy = y_pred_best.cpu().detach().numpy()
            # else:
            #     y_pred_numpy = y_pred_best.cpu().detach().numpy()
            #
            # fig1, axes1 = create_multiplots(X_test, y_test, y_pred_numpy, number_sample=9, cmap='copper', normalize=True)
            # plt.tight_layout()
            # # plt.savefig(f".\saved\plot\{model_name}_patches.png")
            # plt.show()

        settings.beta += 0.01  # CHANGE TO DIFFERENT PARAMETER IF NEEDED


if __name__ == '__main__':
    main()
