import math
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import dataloader
from transformers import Wav2Vec2Model, Wav2Vec2Config

from utils import accuracy_cal
from utils import myWavLoader

epochs = 100
lr = 1e-3
beta1 = 0.93
beta2 = 0.98
seq_len = 64000
batch_size = 16
train_num = 435
val_num = 100
use_amp = True


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
        self.pool = nn.MaxPool1d(2)
        configuration = Wav2Vec2Config(num_attention_heads=6, num_hidden_layers=6)
        self.encoder = Wav2Vec2Model(configuration)
        # output torch.Size([batch_size, seq_len, feature_dim])  feature_dim=32
        self.bn = nn.BatchNorm1d(768)  # input [batch_size feature_dim, seq_len]
        hidden_size = cal_hidden(seq_len=seq_len/2)
        self.fc1 = nn.Linear(hidden_size, 1)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(768, 7)
        # self.act = nn.Softmax(dim=1)

    def forward(self, x):
        # x = self.encoder(x).logits
        # x = x.permute(0, 2, 1)
        x = self.pool(x)
        x = self.encoder(x)
        x = x[0]
        x = self.drop(x)
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

train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths=[train_num, val_num])

train_loader = dataloader.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = dataloader.DataLoader(val_dataset, batch_size=16, shuffle=False)

# print(cal_hidden(64000))

model = Wav2vec(seq_len=seq_len)
for name, param in model.named_parameters():
    if "encoder" in name:
        param.requires_grad = False

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(beta1, beta2))
loss_fn = torch.nn.CrossEntropyLoss()
if use_amp:
    scaler = GradScaler()

if torch.cuda.is_available():
    model = model.cuda()
    loss_fn = loss_fn.cuda()
for epoch in range(epochs):
    model.train()
    train_correct = 0
    train_loss = 0
    val_correct = 0
    val_loss = 0
    for step, (bx, by) in enumerate(train_loader):
        if torch.cuda.is_available():
            bx = bx.cuda()
            by = by.cuda()
        optimizer.zero_grad()
        if use_amp:
            with autocast():
                output = model(bx)
                loss = loss_fn(output, by)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(bx)
            loss = loss_fn(output, by)
            loss.backward()
            optimizer.step()
        train_correct += accuracy_cal(output, by)
        train_loss += loss.data.item()
    model.eval()
    with torch.no_grad():
        for step, (vx, vy) in enumerate(val_loader):
            if torch.cuda.is_available():
                vx = vx.cuda()
                vy = vy.cuda()
            if use_amp:
                with autocast():
                    output = model(vx)
                    loss = loss_fn(output, vy)
            else:
                output = model(vx)
                loss = loss_fn(output, vy)
            val_correct += accuracy_cal(output, vy)
            val_loss += loss.data.item()
    print("epoch: {}, train_accuracy: {:.3f}\t train_loss: {:.4f}; \t val_accuracy: {:.3f}\t val_loss: {:.4f}".format(
        epoch + 1, train_correct / train_num, train_loss / (math.ceil(train_num / batch_size)),
        val_correct / val_num, train_loss / (math.ceil(val_num / batch_size))))
