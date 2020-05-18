import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, D, H1, H2, class_count):
        super().__init__()
        self.conv1 = nn.Conv2d(D, H1, kernel_size=5)
        self.conv2 = nn.Conv2d(H1, H2, kernel_size=5)
        self.dropout1 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(320, 50)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(50, class_count)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.max_pool2d(x, 2)
        x = torch.relu(x)

        x = self.conv2(x)
        x = self.dropout1(x)
        x = torch.max_pool2d(x, 2)
        x = torch.relu(x)

        x = x.view(-1, 320)
        x = self.fc1(x)
        x = torch.relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)

        return torch.log_softmax(x, dim=-1)
