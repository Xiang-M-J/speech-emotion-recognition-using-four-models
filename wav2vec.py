import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import dataloader
from transformers import Wav2Vec2ForCTC

from utils import accuracy_cal
from utils import myWavLoader

epochs = 50
lr = 1e-4
beta1 = 0.93
beta2 = 0.98
seq_len = 64000


def cal_hidden(seq_len):
    h1 = math.floor((seq_len - 10) / 5 + 1)
    h2 = math.floor((h1 - 3) / 2 + 1)
    h3 = math.floor((h2 - 3) / 2 + 1)
    h4 = math.floor((h3 - 3) / 2 + 1)
    h5 = math.floor((h4 - 3) / 2 + 1)
    h6 = math.floor((h5 - 2) / 2 + 1)
    h7 = math.floor((h6 - 2) / 2 + 1)
    return h7


class Wav2vec(nn.Module):
    def __init__(self, seq_len):
        super(Wav2vec, self).__init__()
        self.encoder = Wav2Vec2ForCTC.from_pretrained("./pretrained/")
        # output torch.Size([batch_size, seq_len, feature_dim])  feature_dim=32
        self.bn = nn.BatchNorm1d(32)  # input [batch_size feature_dim, seq_len]
        hidden_size = cal_hidden(seq_len=seq_len)
        self.fc1 = nn.Linear(hidden_size, 1)
        self.fc2 = nn.Linear(32, 7)
        # self.act = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.encoder(x).logits
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        # x = self.pool(x)
        # x = x.reshape(x.size(0), 4 * 32)
        x = self.fc1(x)
        # x = x.reshape(x.size(0), 32)
        x = x.squeeze(-1)
        x = self.fc2(x)
        # x = self.act(x)
        return x


x = np.load("preprocess/x_wav.npy")
y = np.load("preprocess/y.npy")
dataset = myWavLoader(x=x, y=y)

data_loader = dataloader.DataLoader(dataset, batch_size=16, shuffle=True)

print(cal_hidden(64000))

model = Wav2vec(seq_len=seq_len)
for name, param in model.named_parameters():
    if "encoder" in name:
        param.requires_grad = False

# optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(beta1, beta2))
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(beta1, beta2))
loss_fn = torch.nn.CrossEntropyLoss()
if torch.cuda.is_available():
    model = model.cuda()
    loss_fn = loss_fn.cuda()
for epoch in range(epochs):
    model.train()
    train_correct = 0
    train_loss = 0
    val_correct = 0
    val_loss = 0
    for step, (bx, by) in enumerate(data_loader):
        if torch.cuda.is_available():
            bx = bx.cuda()
            by = by.cuda()
        bx = bx.squeeze(1)
        by = by.float()
        output = model(bx)
        loss = loss_fn(output, by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_correct += accuracy_cal(output, by)
        train_loss += loss.data.item()
    print("epoch: {}, accuracy: {:.3f}".format(epoch + 1, train_correct / 535))
