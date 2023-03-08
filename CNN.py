import torch.nn as nn


class CNNNet(nn.Module):
    def __init__(self, feature_dim=39, drop_rate=0.2, num_class=7):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Sequential(  # input(batch_size, feature_dim, time_step)
            nn.Conv1d(in_channels=feature_dim, out_channels=64, kernel_size=3, padding="same"),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.ReLU(),
        )  # output(batch_size, feature_dim, time_step/2)
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding="same"),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.ReLU(),
        )  # output(batch_size, feature_dim, time_step/4)
        # self.conv3 = nn.Sequential(
        #     nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding="same"),
        #     nn.MaxPool1d(kernel_size=2),
        #     nn.BatchNorm1d(64),
        #     nn.Dropout(0.2),
        #     nn.ReLU()
        # )
        # self.resample = nn.Sequential(
        #     nn.Conv1d(in_channels=feature_dim, out_channels=64, kernel_size=1, padding="same"),
        #     nn.MaxPool1d(kernel_size=4),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        # )
        self.fc1 = nn.Linear(43 * 64, 500)
        self.dropout = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(500, num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
