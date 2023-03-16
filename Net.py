import datetime
import math
import os

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tensorboardX import SummaryWriter
from torch.utils.data import dataloader

from CNN import CNNNet
from LSTM import LSTMNet
from TIM import TIMNet
from GRU import GRUNet
from MLP import MLPNet
from Transformer import TransformerNet, TransformerNetV2
from config import beta1, beta2, step_size, gamma, save, augment, data_type, use_noam, use_scheduler, warmup, initial_lr
from utils import Metric, smooth_labels, accuracy_cal, \
    CLASS_LABELS, plot_matrix, plot, logger, l2_regularization, myLoader, noam


class Net_Instance:
    def __init__(self, model_type: str, model_name: str, feature_dim=39, num_class=7, addition: str = ""):
        self.model_type = model_type
        self.feature_dim = feature_dim
        self.num_class = num_class
        self.model_name = model_name
        self.addition = addition
        self.best_path = f"models/{model_type}/" + model_name + "_best" + ".pt"  # 模型保存路径(max val acc)
        self.last_path = f"models/{model_type}/" + model_name + ".pt"  # 模型保存路径(final)
        self.result_path = f"results/{model_type}/"  # 结果保存路径（分为数据和图片）

        if save:
            self.logger = logger(self.model_name, addition=addition)
            date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            self.writer = SummaryWriter("runs/" + date)

    def train(self, train_dataset, val_dataset, batch_size: int, epochs: int, lr: float, weight_decay: float,
              drop_rate: float = 0.2, smooth: bool = True, is_mask=False):

        if save:
            self.logger.start()
            self.writer.add_text("model name", self.model_name)
            self.writer.add_text('addition', self.addition)
        metric = Metric()
        if augment:
            if data_type == "mel":
                x_a = np.load("preprocess/x_mel_a.npy")
            else:
                x_a = np.load("preprocess/x_mfcc_a.npy")
            y_a = np.load("preprocess/y_a.npy")
            x_a = x_a.transpose([0, 2, 1])
            augment_indices = np.random.permutation(x_a.shape[0])
            x_a = x_a[augment_indices]
            y_a = y_a[augment_indices]
            augment_dataset = myLoader(x_a, y_a)
            train_loader = dataloader.DataLoader(
                dataset=train_dataset + augment_dataset,
                batch_size=batch_size
            )
        else:
            train_loader = dataloader.DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
            )
        val_loader = dataloader.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
        )
        train_num = len(train_loader.dataset)  # 当数据增强时这样能得到正确的训练集数量
        val_num = len(val_dataset)
        if self.model_type == "CNN":
            model = CNNNet(feature_dim=self.feature_dim, drop_rate=drop_rate, num_class=self.num_class)
        elif self.model_type == "LSTM":
            model = LSTMNet(feature_dim=self.feature_dim, drop_rate=drop_rate, num_class=self.num_class)
        elif self.model_type == "GRU":
            model = GRUNet(feature_dim=self.feature_dim, drop_rate=drop_rate, num_class=self.num_class)
        elif self.model_type == "MLP":
            model = MLPNet(feature_dim=self.feature_dim, drop_rate=drop_rate, num_class=self.num_class)
        elif self.model_type == "TIM":
            model = TIMNet(feature_dim=self.feature_dim, drop_rate=drop_rate, num_class=self.num_class)
        elif self.model_type == "Transformer":
            model = TransformerNetV2(feature_dim=self.feature_dim, drop_rate=drop_rate, num_class=self.num_class)
        else:
            print(f"{self.model_type} is a wrong mode type")
            exit()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(beta1, beta2))
        loss_fn = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=-1)

        if is_mask:
            train_indices = train_dataset.indices
            val_indices = val_dataset.indices
            masks = np.load("preprocess/mask.npy")
            if augment:
                masks_a = np.load("preprocess/mask_a.npy")
                mask1 = masks[train_indices]
                mask2 = masks_a[augment_indices]
                train_masks = np.vstack([mask1, mask2])
                val_masks = masks[val_indices]
            else:
                train_masks = masks[train_indices]
                val_masks = masks[val_indices]
        best_val_accuracy = 0
        if torch.cuda.is_available():
            model = model.cuda()
            loss_fn = loss_fn.cuda()
        steps = 0
        for epoch in range(epochs):
            model.train()
            train_correct = 0
            train_loss = 0
            val_correct = 0
            val_loss = 0

            for step, (bx, by) in enumerate(train_loader):
                if use_noam and self.model_type == "Transformer":
                    for p in optimizer.param_groups:
                        p['lr'] = initial_lr * noam(d_model=512, step=steps + 1, warmup=warmup)
                if torch.cuda.is_available():
                    bx = bx.cuda()
                    by = by.cuda()
                if is_mask and self.model_type == "Transformer":
                    mask = train_masks[step * batch_size: (step * batch_size + bx.shape[0])]
                    output = model(bx, mask)
                else:
                    output = model(bx)
                if smooth:
                    by = smooth_labels(by, 0.1)  # 平滑标签
                loss = loss_fn(output, by) + l2_regularization(model, weight_decay)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_correct += accuracy_cal(output, by)
                train_loss += loss.data.item()
                steps += 1

            model.eval()
            with torch.no_grad():
                for step, (vx, vy) in enumerate(val_loader):
                    if torch.cuda.is_available():
                        vx = vx.cuda()
                        vy = vy.cuda()
                    if is_mask and self.model_type == "Transformer":
                        mask = val_masks[step * batch_size: (step * batch_size + vx.shape[0])]
                        output = model(vx, mask)
                    else:
                        output = model(vx)
                    # output = model(vx)
                    loss = loss_fn(output, vy)
                    val_correct += accuracy_cal(output, vy)
                    val_loss += loss.data.item()
            if use_scheduler:
                scheduler.step()

            metric.train_acc.append(float(train_correct * 100) / train_num)
            metric.train_loss.append(train_loss / math.ceil((train_num / batch_size)))
            metric.val_acc.append(float(val_correct * 100) / val_num)
            metric.val_loss.append(val_loss / math.ceil(val_num / batch_size))
            if save:
                self.writer.add_scalar('train accuracy', metric.train_acc[-1], epoch + 1)
                self.writer.add_scalar('train loss', metric.train_loss[-1], epoch + 1)
                self.writer.add_scalar('validation accuracy', metric.val_acc[-1], epoch + 1)
                self.writer.add_scalar('validation loss', metric.val_loss[-1], epoch + 1)
            print(
                'Epoch :{}\t train Loss:{:.4f}\t train Accuracy:{:.3f}\t val Loss:{:.4f} \t val Accuracy:{:.3f}'.format(
                    epoch + 1, metric.train_loss[-1], metric.train_acc[-1], metric.val_loss[-1],
                    metric.val_acc[-1]))
            if metric.val_acc[-1] > best_val_accuracy:
                print(f"val_accuracy improved from {best_val_accuracy} to {metric.val_acc[-1]}")
                best_val_accuracy = metric.val_acc[-1]
                metric.best_val_acc[0] = best_val_accuracy
                metric.best_val_acc[1] = metric.train_acc[-1]
                if save:
                    torch.save(model, self.best_path)
                    print(f"saving model to {self.best_path}")
            elif metric.val_acc[-1] == best_val_accuracy:
                if metric.train_acc[-1] > metric.best_val_acc[1]:
                    metric.best_val_acc[1] = metric.train_acc[-1]
                    if save:
                        torch.save(model, self.best_path)
                        print(f"update train accuracy")
            else:
                print(f"val_accuracy did not improve from {best_val_accuracy}")
        if save:
            torch.save(model, self.last_path)
            print(f"save model(last): {self.last_path}")
            plot(metric.item(), self.model_name, self.result_path)
            np.save(self.result_path + "data/" + self.model_name + "_train_metric.npy", metric.item())
            self.writer.add_text("beat validation accuracy", f"{metric.best_val_acc}")
            dummy_input = torch.rand(16, 39, 173)

            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
            if self.model_type != "Transformer":
                self.writer.add_graph(model, dummy_input)
            else:
                mask = torch.rand(16, 173)
                self.writer.add_graph(model, [dummy_input, mask])
                # else:
                #     self.writer.add_graph(model, [dummy_input])
            self.logger.train(train_metric=metric)
        return metric

    def test(self, test_dataset, batch_size: int, model_path: str = None):
        if model_path is None:
            model_path = self.best_path
        if not os.path.exists(model_path):
            print(f"error! cannot find the model in {model_path}")
            return
        model = torch.load(model_path)
        print(f"load model: {model_path}")
        self.logger.start()
        test_loader = dataloader.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
        )
        test_num = len(test_dataset)
        model.eval()
        metric = Metric(mode="test")
        test_correct = 0
        test_loss = 0
        loss_fn = torch.nn.CrossEntropyLoss()
        y_pred = torch.zeros(test_num)
        y_true = torch.zeros(test_num)
        if torch.cuda.is_available():
            loss_fn = loss_fn.cuda()
        for step, (vx, vy) in enumerate(test_loader):
            if torch.cuda.is_available():
                vx = vx.cuda()
                vy = vy.cuda()
            output = model(vx)
            y_pred[step * batch_size: step * batch_size + vy.shape[0]] = torch.max(output.data, 1)[1]
            y_true[step * batch_size: step * batch_size + vy.shape[0]] = torch.max(vy.data, 1)[1]
            loss = loss_fn(output, vy)
            test_correct += accuracy_cal(output, vy)
            test_loss += loss
        conf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(self.num_class))
        metric.confusion_matrix = conf_matrix
        fig = plot_matrix(conf_matrix, labels_name=CLASS_LABELS, model_name=self.model_name, normalize=False,
                          result_path=self.result_path)
        report = classification_report(y_true, y_pred, target_names=CLASS_LABELS)
        print(report)
        metric.report = report
        metric.test_loss = test_loss / math.ceil(test_num / batch_size)
        metric.test_acc = float((test_correct * 100) / test_num)
        print("{} test Loss:{:.4f} test Accuracy:{:.3f}".format(self.model_name, metric.test_loss, metric.test_acc))
        if save:
            self.writer.add_text("test loss", str(metric.test_loss))
            self.writer.add_text("test accuracy", str(metric.test_acc))
            self.writer.add_figure("confusion matrix", fig)
            self.writer.add_text("classification report", report)
            self.logger.test(test_metric=metric)
            np.save(self.result_path + "data/" + self.model_name + "_test_metric.npy", metric.item())
        return metric
