import matplotlib.pyplot as plt

from autograd.tensor import Tensor
from autograd.functions import Linear, CrossEntropyLoss, ReLU, ModelArch, SGD, he_normal_init
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def generate_real_data():
    iris = load_iris()
    X, y = iris.data, iris.target

    X = np.pad(X, ((0, 0), (0, 6)), 'constant')

    encoder = OneHotEncoder(sparse_output=False)
    Y = encoder.fit_transform(y.reshape(-1, 1))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    X_autograd_train = [Tensor(x, requires_grad=True) for x in X_train]
    Y_autograd_train = [Tensor(y, requires_grad=False) for y in Y_train]

    X_torch_train = torch.tensor(X_train, dtype=torch.float32)
    Y_torch_train = torch.tensor(np.argmax(Y_train, axis=1), dtype=torch.long)

    return (X_autograd_train, Y_autograd_train), (X_torch_train, Y_torch_train)


class MyAutogradModel(ModelArch):
    def __init__(self):
        self.layers = {
            "fc1": Linear(10, 8, weights=he_normal_init(10, 8)),
            "fc2": Linear(8, 3, weights=he_normal_init(8, 3))
        }
        super().__init__()
        self._register_parameters(self.layers.values())

    def forward(self, x):
        out = ReLU(self.layers["fc1"](x))
        return self.layers["fc2"](out)


class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


(autograd_X_train, autograd_Y_train), (torch_X_train, torch_Y_train) = generate_real_data()
LEARNING_RATE  = 0.01
EPOCHS = 200

auto_model = MyAutogradModel()
auto_optimizer = SGD(lr=LEARNING_RATE, model=auto_model)

torch_model = TorchModel()
torch_optimizer = optim.SGD(torch_model.parameters(), lr=LEARNING_RATE)
torch_criterion = nn.CrossEntropyLoss()

auto_losses = []
torch_losses = []

for epoch in range(EPOCHS):
    auto_total_loss = 0
    for x, y_true in zip(autograd_X_train, autograd_Y_train):
        y_pred = auto_model(x)
        loss = CrossEntropyLoss(y_pred, y_true)
        auto_total_loss += loss.data.sum()
        loss.backward()
        auto_optimizer.step()
        auto_optimizer.zero_grad()

    torch_total_loss = 0
    for i in range(torch_X_train.size(0)):
        x = torch_X_train[i].unsqueeze(0)
        y = torch_Y_train[i].unsqueeze(0)
        output = torch_model(x)
        loss = torch_criterion(output, y)
        torch_total_loss += loss.item()
        torch_optimizer.zero_grad()
        loss.backward()
        torch_optimizer.step()

    auto_avg_loss = auto_total_loss / len(autograd_X_train)
    torch_avg_loss = torch_total_loss / torch_X_train.size(0)

    auto_losses.append(auto_avg_loss)
    torch_losses.append(torch_avg_loss)

    print(f"{epoch+1:>5} | {auto_avg_loss:.6f} | {torch_avg_loss:.6f}")

plt.plot(range(EPOCHS), auto_losses, label='My Autograd Loss')
plt.plot(range(EPOCHS), torch_losses, label='Torch Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss/Epochs torch vs my autograd')
plt.legend()
plt.grid()
plt.show()
