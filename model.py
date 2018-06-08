import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):

    def __init__(self, num_classes):
        super(AlexNet, self).__init__()

        self.num_classes = num_classes

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=11, stride=(2, 3), padding=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2)),
            nn.BatchNorm2d(48)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(48, 128, kernel_size=5, stride=(2, 3), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(128)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2)),
            nn.BatchNorm2d(128)
        )

        self.fc1 = nn.Linear(2560, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(F.relu(self.fc1(x)))
        x = F.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class LeNet(nn.Module):

    def __init__(self, num_classes):
        super(LeNet, self).__init__()

        self.num_classes = num_classes

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(20)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(20, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(20)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(20, 40, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(40, 40, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(40)
        )

        self.fc1 = nn.Linear(3520, 1000)
        self.fc2 = nn.Linear(1000, self.num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
