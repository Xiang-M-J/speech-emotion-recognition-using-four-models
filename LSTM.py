import torch.nn as nn


class LSTMNet(nn.Module):
    def __init__(self, feature_dim=39, drop_rate=0.2, num_class=7):  # input [batch_size, time_step, feature_dim]
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=256, batch_first=True, bidirectional=True)
        # output [batch_size, time_step, feature_dim]
        # self.pool = nn.MaxPool1d(2)  # input [batch_size,feature_dim,time_step]
        self.batch_norm = nn.BatchNorm1d(
            2 * 256)  # input [batch_size,feature_dim,time_step] or [batch_size,feature_dim]

        self.fc1 = nn.Linear(173, 1)
        self.dropout = nn.Dropout(p=drop_rate)
        self.fc2 = nn.Linear(2 * 256, num_class)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()  # 调用flatten_parameters让parameter的数据存放成连续的块，提高内存的利用率和效率
        RNN_out, (_, _) = self.lstm(x)  # if bidirectional=True，输出将包含序列中每个时间步的正向和反向隐藏状态的串联
        # 添加pool
        # x = self.pool(RNN_out.transpose(1, 2))
        # x = x[:, :, -1]  # 取最后一步的RNN输出
        # x = self.batch_norm(x)
        # x = self.fc1(x)
        # 不加pool
        x = self.batch_norm(RNN_out.transpose(1, 2))
        # x = RNN_out[:, -1, :]  # 取最后一步的RNN输出
        x = self.fc1(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
