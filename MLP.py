import torch.nn as nn


# 不加dropout和L2偏置，训练集上正确率能达到80%，验证集上只有60%
class MLPNet(nn.Module):
    def __init__(self, feature_dim=39, drop_rate=0.2, num_class=7):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(173, 50)
        self.fc11 = nn.Linear(50, 1)
        self.dropout = nn.Dropout(p=drop_rate)
        self.fc2 = nn.Linear(feature_dim, num_class)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc11(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
