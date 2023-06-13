from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

from models.cnn import CNN
from utils.misc import load_list_from_file


# Set hyperparameters
batch_size = 100
learning_rate = 0.001
num_epochs = 20

# Initialize model
model = CNN(batch_size, learning_rate, num_epochs)

# Load data
X, y = torch.load('./saved/double_dot_patches_Dx.pt'), [float(x) for x in load_list_from_file('./saved/double_dot_normalized_angles.txt')]
n, N = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# Convert to 2D PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)


# We use the pre-defined number of epochs to determine how many iterations to train the network on
for epoch in range(num_epochs):
    model.train()  # prepare model for training
    # Load in the data in batches using the train_loader object
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))




