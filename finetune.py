# 感觉效果一般
import os
from functools import partial
import numpy as np
import torch
from ray.tune import CLIReporter
from torch.utils.data import dataloader
from utils import myLoader, noam, smooth_labels, l2_regularization, accuracy_cal
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from CNN import CNNNet
from LSTM import LSTMNet
from TIM import TIMNet
from Transformer import TransformerNet

config = {
    'lr': tune.loguniform(1e-5, 1e-2),
    'weight_decay': tune.uniform(0, 0.5),
    'dropout': tune.uniform(0, 0.5)
}

model_type = "LSTM"  # 模型选择
data_type = "mfcc"
use_noam = False
use_scheduler = True
warmup = 300
initial_lr = 0.1
augment = False
random_seed = 34
epochs = 100  # 迭代次数
if augment:
    spilt_rate = [0.6, 0.2, 0.2]
else:
    spilt_rate = [0.8, 0.1, 0.1]  # 训练集、验证集、测试集分割比例
feature_dim = 39  # 特征维度
num_class = 7  # 类别数
smooth = True  # 是否平滑标签（True：平滑 False：不平滑）


def train(config, checkpoint_dir=None, train_dataset=None, val_dataset=None, epochs: int = 100, ):
    if augment:
        if data_type == "mel":
            x_a = np.load("D:\\graduate_code\\Model4\\preprocess\\x_mel_a.npy")
        else:
            x_a = np.load("D:\\graduate_code\\Model4\\preprocess\\x_mfcc_a.npy")
        y_a = np.load("D:\\graduate_code\\Model4\\preprocess\\y_a.npy")
        x_a = x_a.transpose([0, 2, 1])
        augment_indices = np.random.permutation(x_a.shape[0])
        x_a = x_a[augment_indices]
        y_a = y_a[augment_indices]
        augment_dataset = myLoader(x_a, y_a)
        train_loader = dataloader.DataLoader(
            dataset=train_dataset + augment_dataset,
            batch_size=8
        )
    else:
        train_loader = dataloader.DataLoader(
            dataset=train_dataset,
            batch_size=8,
        )
    val_loader = dataloader.DataLoader(
        dataset=val_dataset,
        batch_size=8,
    )

    train_num = len(train_loader.dataset)  # 当数据增强时这样能得到正确的训练集数量
    val_num = len(val_dataset)
    if model_type == "CNN":
        model = CNNNet(feature_dim=feature_dim, drop_rate=config['dropout'], num_class=num_class)
    elif model_type == "LSTM":
        model = LSTMNet(feature_dim=feature_dim, drop_rate=config['dropout'], num_class=num_class)
    elif model_type == "TIM":
        model = TIMNet(feature_dim=feature_dim, drop_rate=config['dropout'], num_class=num_class)
    elif model_type == "Transformer":
        model = TransformerNet(feature_dim=feature_dim, drop_rate=config['dropout'], num_class=num_class)
    else:
        print(f"{model_type} is a wrong mode type")
        exit()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['lr'], betas=(0.93, 0.98))
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5, last_epoch=-1)
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    best_val_accuracy = 0
    best_accuracy = [0, 0]
    # if torch.cuda.is_available():
    #     model = model.cuda()
    #     loss_fn = loss_fn.cuda()
    steps = 0
    for epoch in range(epochs):
        model.train()
        train_correct = 0
        train_loss = 0
        val_correct = 0
        val_loss = 0
        for step, (bx, by) in enumerate(train_loader):
            if use_noam and model_type == "Transformer":
                for p in optimizer.param_groups:
                    p['lr'] = initial_lr * noam(d_model=512, step=steps + 1, warmup=warmup)
            # if torch.cuda.is_available():
            #     bx = bx.cuda()
            #     by = by.cuda()
            output = model(bx)
            if smooth:
                by = smooth_labels(by, 0.1)  # 平滑标签
            loss = loss_fn(output, by) + l2_regularization(model, config['weight_decay'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_correct += accuracy_cal(output, by)
            train_loss += loss.data.item()
            steps += 1

        model.eval()
        with torch.no_grad():
            for step, (vx, vy) in enumerate(val_loader):
                # if torch.cuda.is_available():
                #     vx = vx.cuda()
                #     vy = vy.cuda()
                output = model(vx)
                loss = loss_fn(output, vy)
                val_correct += accuracy_cal(output, vy)
                val_loss += loss.data.item()
        if use_scheduler:
            scheduler.step()
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
        train_acc = float(train_correct * 100) / train_num
        val_acc = float(val_correct * 100) / val_num
        tune.report(val_accuracy=val_acc, train_accuracy=train_acc)
        print(f"epoch {epoch+1}")
        # if val_acc > best_val_accuracy:
        #     best_accuracy[0] = best_val_accuracy
        #     best_accuracy[1] = train_acc
        #
        # else:
        #     print(f"val_accuracy did not improve from {best_val_accuracy}")


if data_type == "mel":
    x = np.load("preprocess/x_mel.npy")
    feature_dim = 40
else:
    x = np.load("preprocess/x_mfcc.npy")
y = np.load("preprocess/y.npy")
model_name = "fine-tune"
addition = "fine-tune"
Num = x.shape[0]  # 样本数
if data_type == "mfcc":
    x = x.transpose([0, 2, 1])
dataset = myLoader(x, y)  # input shape of x: [样本数，特征维度，时间步]  input shape of y: [样本数，类别数]
train_num = int(Num * spilt_rate[0])
val_num = int(Num * spilt_rate[1])
test_num = Num - train_num - val_num
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                         [train_num, val_num, test_num],
                                                                         generator=torch.Generator().manual_seed(
                                                                             random_seed))

scheduler = ASHAScheduler(
    metric="val_accuracy",
    mode="max",
    max_t=20,
    grace_period=1,
    reduction_factor=2)
reporter = CLIReporter(
    # parameter_columns=["l1", "l2", "lr", "batch_size"],
    metric_columns=["val_accuracy", "train_accuracy"])
tuner = tune.run(
    partial(
        train, train_dataset=train_dataset, val_dataset=val_dataset, epochs=epochs
    ),
    config=config,
    progress_reporter=reporter,
    scheduler=scheduler,
    resources_per_trial={'gpu': 1},
    fail_fast="raise"
)
best_trial = tuner.get_best_trial("val_accuracy", "max", 'last')
print("Best trial config: {}".format(best_trial.config))
