import torch
import torch.nn as nn


class AlexNet2d(nn.Module):

    def __init__(self, num_classes):
        super(AlexNet2d, self).__init__()

        self.num_classes = num_classes

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=(2, 3), padding=5),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2))
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=(2, 3), padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=(2, 2))
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 5 * 4, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, self.num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class AlexNet1d(nn.Module):

    def __init__(self, num_classes):
        super(AlexNet1d, self).__init__()

        self.num_classes = num_classes

        self.block1 = nn.Sequential(
            nn.Conv1d(64, 96, kernel_size=11, stride=3, padding=5),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(96, 256, kernel_size=5, stride=3, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, self.num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ConvLSTM(nn.Module):

    def __init__(self, num_classes):
        super(ConvLSTM, self).__init__()

        self.num_classes = 41

        self.block1 = nn.Sequential(
            nn.Conv1d(64, 96, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.lstm = nn.LSTM(256, 1024)

        self.classifier = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, self.num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # [batch, input_size, seq_len] => [input_size, batch, seq_len]
        # => [seq_len, batch, input_size]
        x = x.transpose(0, 1)
        x = x.transpose(0, 2)
        # (h_0, c_0) を省略するとzero vectorで初期化
        x, hn = self.lstm(x)
        # 最後の系列をLSTMに入れた時の出力のみを使う
        x = x[-1]
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    # dummy test
    # [128, 64, 401]
    model = ConvLSTM(41)
    dummy = torch.zeros(128, 64, 401)
    out = model(dummy)
