import numpy as np
import torch.nn as nn
import torch


class PositionalEncoding(nn.Module):
    def __init__(self, feature_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = torch.from_numpy(np.array([
            [pos / np.power(10000, 2 * i / feature_dim) for i in range(feature_dim)]
            if pos != 0 else np.zeros(feature_dim) for pos in range(max_len)]))

        pos_table[1:, 0::2] = torch.sin(pos_table[1:, 0::2])  # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = torch.cos(pos_table[1:, 1::2])  # 字嵌入维度为奇数时
        self.pos_table = pos_table.float().cuda()  # enc_inputs: [seq_len, feature_dim]

    def forward(self, enc_inputs):  # enc_inputs: [batch_size, seq_len, feature_dim]
        enc_inputs = torch.add(enc_inputs, self.pos_table[:enc_inputs.size(1), :])  # 此处使用+=会导致报错
        # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
        # 此处用到了广播机制，self.pos_table[:enc_inputs.size(1), :]的形状为[seq_len, feature_dim], 
        # 相加相当于在batch_size维度上相加
        return self.dropout(enc_inputs).cuda()


def get_mask(mask, n_head: int):
    batch = mask.shape[0]
    seq_len = mask.shape[1]
    src_attn_mask = torch.tensor(mask == 0).unsqueeze(1)
    src_attn_mask = src_attn_mask.expand(batch, seq_len, seq_len)
    src_attn_mask = src_attn_mask.repeat(n_head, 1, 1)
    return src_attn_mask.cuda()


class TransformerNet(nn.Module):
    def __init__(self, feature_dim=39, drop_rate=0.1, num_class=7):
        super(TransformerNet, self).__init__()  # input shape: [batch_shape, feature_dim, seq_len]
        self.d_model1 = 64
        self.d_model2 = 256
        self.n_head = 8
        self.dim_feedforward = 1024
        self.n_layers = 4
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=feature_dim, out_channels=self.d_model1, kernel_size=1, padding="same"),
            nn.BatchNorm1d(num_features=self.d_model1),
            nn.ReLU(),
            nn.Dropout1d(p=0),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.d_model1, out_channels=self.d_model2, kernel_size=1, padding="same"),
            nn.BatchNorm1d(num_features=self.d_model2),
            nn.ReLU(),
            nn.Dropout1d(p=0),
        )
        self.position = PositionalEncoding(self.d_model2)
        encoder_layer = nn.TransformerEncoderLayer(self.d_model2, self.n_head,
                                                   self.dim_feedforward,  # input: [batch_size, seq_len, feature_dim]
                                                   dropout=0, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        self.drop = nn.Dropout(drop_rate)
        # self.global_pool = nn.AdaptiveAvgPool1d(1)  # input: [batch_size, feature_dim, seq_len]
        self.fc1 = nn.Linear(173, 1)
        self.fc2 = nn.Linear(self.d_model2, num_class)

    def forward(self, x, mask=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.position(x.transpose(1, 2))
        if mask is not None:
            mask = get_mask(mask, self.n_head)
            x = self.encoder(x, mask)
        else:
            x = self.encoder(x)
        # 不用AdaptiveAvgPool1d

        x = x.transpose(1, 2)
        x = self.fc1(x)
        x = x.squeeze(-1)
        x = self.drop(x)
        x = self.fc2(x)

        # 用AdaptiveAvgPool1d
        # x = self.global_pool(x.transpose(2, 1))
        # x = x.squeeze(-1)
        # x = self.drop(x)
        # x = self.fc2(x)
        return x


class TransformerNetV2(nn.Module):
    def __init__(self, feature_dim=39, drop_rate=0.1, num_class=7):
        super(TransformerNetV2, self).__init__()  # input shape: [batch_shape, feature_dim, seq_len]
        self.d_model1 = 64
        self.d_model2 = 256
        self.n_head = 8
        self.dim_feedforward = 1024
        self.n_layers = 2
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=feature_dim, out_channels=self.d_model1, kernel_size=1, padding="same"),
            nn.BatchNorm1d(num_features=self.d_model1),
            nn.Dropout1d(p=0.2),
            nn.ReLU()
        )
        # self.conv2 = nn.Sequential(
        #     nn.Conv1d(in_channels=self.d_model1, out_channels=self.d_model2, kernel_size=1, padding="same"),
        #     nn.BatchNorm1d(num_features=self.d_model2),
        #     nn.Dropout1d(p=0.2),
        #     nn.ReLU()
        # )
        self.position = PositionalEncoding(self.d_model1)
        encoder_layer = nn.TransformerEncoderLayer(self.d_model1, self.n_head,
                                                   self.dim_feedforward,  # input: [batch_size, seq_len, feature_dim]
                                                   dropout=0.2, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        self.drop = nn.Dropout(drop_rate)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # input: [batch_size, feature_dim, seq_len]
        self.fc1 = nn.Linear(173, 1)
        self.fc2 = nn.Linear(self.d_model1, num_class)

    def forward(self, x, mask=None):
        x = self.conv1(x)
        # x = self.conv2(x)
        x = x.transpose(1, 2)
        x = self.position(x)
        if mask is not None:
            mask = get_mask(mask, self.n_head)
            x = self.encoder(x, mask)
        else:
            x = self.encoder(x)
        # 不用AdaptiveAvgPool1d
        # x = x.transpose(1, 2)
        # x = self.fc1(x)
        # x = x.squeeze(-1)
        # x = self.drop(x)
        # x = self.fc2(x)

        # 用AdaptiveAvgPool1d
        x = self.global_pool(x.transpose(2, 1))
        x = x.squeeze(-1)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class TransformerNetV3(nn.Module):
    def __init__(self, feature_dim=39, drop_rate=0.1, num_class=7):
        super(TransformerNetV3, self).__init__()  # input shape: [batch_shape, feature_dim, seq_len]
        # self.d_model1 = 64
        # self.d_model2 = 256
        self.n_head = 3
        self.dim_feedforward = 512
        self.n_layers = 4
        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(in_channels=feature_dim, out_channels=self.d_model1, kernel_size=1, padding="same"),
        #     nn.BatchNorm1d(num_features=self.d_model1),
        #     nn.Dropout1d(p=0.2),
        #     nn.ReLU()
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv1d(in_channels=self.d_model1, out_channels=self.d_model2, kernel_size=1, padding="same"),
        #     nn.BatchNorm1d(num_features=self.d_model2),
        #     nn.Dropout1d(p=0.2),
        #     nn.ReLU()
        # )
        # self.pool = nn.MaxPool1d(2)     # input: [batch_size, feature_dim, seq_len]
        # self.position = PositionalEncoding(self.d_model1)
        encoder_layer = nn.TransformerEncoderLayer(feature_dim, self.n_head,
                                                   self.dim_feedforward,  # input: [batch_size, seq_len, feature_dim]
                                                   dropout=0.4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        self.drop = nn.Dropout(drop_rate)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # input: [batch_size, feature_dim, seq_len]
        self.fc1 = nn.Linear(173, 1)
        self.fc2 = nn.Linear(feature_dim, num_class)

    def forward(self, x, mask=None):
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.pool(x)
        x = x.transpose(1, 2)
        # x = self.position(x)
        if mask is not None:
            mask = get_mask(mask, self.n_head)
            x = self.encoder(x, mask)
        else:
            x = self.encoder(x)
        # 不用AdaptiveAvgPool1d
        # x = x.transpose(1, 2)
        # x = self.fc1(x)
        # x = x.squeeze(-1)
        # x = self.drop(x)
        # x = self.fc2(x)

        # 用AdaptiveAvgPool1d
        x = self.global_pool(x.transpose(2, 1))
        x = x.squeeze(-1)
        x = self.drop(x)
        x = self.fc2(x)
        return x
