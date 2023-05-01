from torch.autograd import Variable
import torch


def train(model, optimizer, criterion, train_x, train_y, val_x, val_y, n_epochs):
    # empty list to store training losses
    train_losses = []
    # empty list to store validation losses
    val_losses = []

    for epoch in range(n_epochs):
        model.train()

        # getting the training set
        x_train, y_train = Variable(train_x), Variable(train_y)
        # getting the validation set
        x_val, y_val = Variable(val_x), Variable(val_y)

        # converting the data into GPU format
        if torch.cuda.is_available():
            x_train = x_train.cuda()
            y_train = y_train.cuda()
            x_val = x_val.cuda()
            y_val = y_val.cuda()

        # clearing the Gradients of the model parameters
        optimizer.zero_grad()

        print('x and y shape for train', x_train.shape, y_train.shape)

        # prediction for training and validation set
        output_train = model(x_train)
        output_val = model(x_val)

        # print('output_train shape: ', output_train.shape)
        # print('y_train shape: ', y_train.shape)
        #
        # print('output_train shape: ', output_val.shape)
        # print('y_train shape: ', y_val.shape)

        # computing the training and validation loss
        loss_train = criterion(output_train, y_train)
        loss_val = criterion(output_val, y_val)
        train_losses.append(loss_train)
        val_losses.append(loss_val)

        # computing the updated weights of all the model parameters
        loss_train.backward()
        optimizer.step()
        tr_loss = loss_train.item()
        if epoch % 2 == 0:
            # printing the validation loss
            print('Epoch : ', epoch + 1, '\t', 'loss :', loss_val)

    return train_losses, val_losses
