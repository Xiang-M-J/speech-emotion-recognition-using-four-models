import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import dataset
from torchvision import transforms
import librosa
from transformers import Wav2Vec2Processor

plt.rcParams['font.sans-serif'] = ['Simhei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

CLASS_LABELS = ["angry", "boredom", "disgust", "fear", "happy", "neutral", "sad"]
dpi = 300


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model.txt when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model.md ...')
        path = os.path.join(self.save_path, 'best_network.pth')
        torch.save(model.state_dict(), path)  # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss


def accuracy_cal_numpy(y_pred, y_true):
    predict = np.argmax(y_pred.numpy(), 1)
    label = np.argmax(y_true.numpy(), 1)
    true_num = (predict == label).sum()
    return true_num


def accuracy_cal(y_pred, y_true):
    predict = torch.max(y_pred.data, 1)[1]  # torch.max()返回[values, indices]，torch.max()[1]返回indices
    label = torch.max(y_true.data, 1)[1]
    true_num = (predict == label).sum()
    return true_num


# 更新混淆矩阵
def confusion_matrix(pred, labels, conf_matrix):
    pred = torch.max(pred, 1)[1]
    labels = torch.max(labels, 1)[1]
    for p, t in zip(pred, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def plot_matrix(cm, labels_name, model_name: str, title='Confusion matrix', normalize=False,
                result_path: str = "results/"):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        Input
        - cm : 计算出的混淆矩阵的值
        - classes : 混淆矩阵中每一行每一列对应的列
        - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    fig = plt.figure(figsize=(4, 4), dpi=120)
    # 画图，如果希望改变颜色风格，可以改变此部分的cmap=plt.get_cmap('Blues')处
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()  # 绘制图例
    # 图像标题
    plt.title(title)
    # 绘制坐标
    num_local = np.array(range(len(labels_name)))
    axis_labels = labels_name
    plt.xticks(num_local, axis_labels, rotation=45)  # 将标签印在x轴坐标上， 并倾斜45度
    plt.yticks(num_local, axis_labels)  # 将标签印在y轴坐标上
    # x,y轴长度一致(问题1解决办法）
    plt.axis("equal")
    # x轴处理一下，如果x轴或者y轴两边有空白的话(问题2解决办法）
    ax = plt.gca()  # 获得当前axis
    left, right = plt.xlim()  # 获得x轴最大最小值
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")
    thresh = cm.max() / 2.
    # 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if normalize:
                plt.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd') + '%',
                         ha="center", va="center",
                         color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行
            else:
                plt.text(j, i, int(cm[i][j]),
                         ha="center", va="center",
                         color="white" if cm[i][j] > thresh else "black")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(result_path + "images/" + model_name + "_confusion_matrix.jpg", dpi=dpi)
    # 显示
    # plt.show()
    return fig


class myLoader(dataset.Dataset):
    def __init__(self, x, y, train=True):
        super(myLoader, self).__init__()
        self.train = train  # 训练和测试时对数据的处理可能不同
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.x = x
        self.y = y

    def __getitem__(self, index):
        data = self.x[index]
        label = self.y[index]
        return data, label

    def __len__(self):
        return len(self.y)


class myWavLoader(dataset.Dataset):
    def __init__(self, x, y) -> None:
        super(myWavLoader, self).__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        data = self.x[index]
        label = self.y[index]
        return data, label

    def __len__(self):
        return len(self.y)


class Metric:
    def __init__(self, mode="train"):
        if mode == "train":
            self.mode = "train"
            self.train_acc = []
            self.train_loss = []
            self.val_acc = []
            self.val_loss = []
            self.best_val_acc = [0, 0]  # [val_acc, train_acc]
        elif mode == "test":
            self.mode = "test"
            self.test_acc = 0
            self.test_loss = 0
            self.confusion_matrix = None
            self.report = None
        else:
            print("wrong mode !!! use default mode train")
            self.mode = "train"
            self.train_acc = []
            self.train_loss = []
            self.val_acc = []
            self.val_loss = []

    def item(self) -> dict:
        if self.mode == "train":
            data = {"train_acc": self.train_acc, "train_loss": self.train_loss,
                    "val_acc": self.val_acc, "val_loss": self.val_loss}
        else:
            data = {"test_acc": self.test_acc, "test_loss": self.test_loss}
        return data


def plot(metric: dict, model_name: str, result_path: str = "results/", ):
    train_acc = metric['train_acc']
    train_loss = metric['train_loss']
    val_loss = metric["val_loss"]
    val_acc = metric['val_acc']
    epoch = np.arange(len(train_acc)) + 1

    plt.figure()
    plt.plot(epoch, train_acc)
    plt.plot(epoch, val_acc)
    plt.legend(["train accuracy", "validation accuracy"])
    plt.xlabel("epoch")
    plt.ylabel("accuracy(%)")
    plt.title("train accuracy and validation accuracy")
    plt.savefig(result_path + "images/" + model_name + "_accuracy.png", dpi=dpi)
    # plt.show()

    plt.figure()
    plt.plot(epoch, train_loss)
    plt.plot(epoch, val_loss)
    plt.legend(["train loss", "validation loss"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("train loss and validation loss")
    plt.savefig(result_path + "images/" + model_name + "_loss.png", dpi=dpi)
    # plt.show()


class logger:
    def __init__(self, model_name: str, addition: str, filename: str = "log.txt"):
        self.model_name = model_name
        self.addition = addition
        self.time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.filename = filename
        self.is_start = False

    def start(self):
        if not self.is_start:
            with open(self.filename, 'a') as f:
                f.write("\n========================\t" + self.time + "\t========================\n")
                f.write(f"model name: \t{self.model_name}\n")
                f.write(f"addition: \t{self.addition}\n")
            self.is_start = True

    def train(self, train_metric: Metric):
        with open(self.filename, 'a') as f:
            f.write("\n========================\t" + "train begin" + "\t========================\n")
            f.write("train(final): \t\t" + "train loss: {:.4f}\t train accuracy: {:.3f}\t validation loss: {:.4f}\t "
                                           "validation accuracy: {:.3f} \n".format(train_metric.train_loss[-1],
                                                                                   train_metric.train_acc[-1],
                                                                                   train_metric.val_loss[-1],
                                                                                   train_metric.val_acc[-1]))
            f.write("train(max_min): \t" + "train loss: {:.4f}\t train accuracy: {:.3f}\t validation loss: {:.4f}\t "
                                           "validation accuracy: {:.3f} \n".format(min(train_metric.train_loss),
                                                                                   max(train_metric.train_acc),
                                                                                   min(train_metric.val_loss),
                                                                                   max(train_metric.val_acc)))
            f.write("best val accuracy: {:3f} \t corresponding train accuracy: {:3f}\n"
                    .format(train_metric.best_val_acc[0], train_metric.best_val_acc[1]))
            f.write("========================\t" + "train end" + "\t========================\n")

    def test(self, test_metric: Metric):
        with open(self.filename, 'a') as f:
            f.write("\n========================\t" + "test begin" + "\t========================\n")
            f.write("test: \t\t\t" + "test loss: \t{:.4f} \t test accuracy:\t {:.3f} \n".format(test_metric.test_loss,
                                                                                                test_metric.test_acc))
            f.write("confusion matrix: \n")
            for i in range(len(test_metric.confusion_matrix)):
                f.write(str(test_metric.confusion_matrix[i]) + '\n')
            f.write("\n")
            f.write("classification report: \n")
            f.write(test_metric.report)
            f.write("\n")
            f.write("========================\t" + "test end" + "\t========================\n")


def log(train_metric: Metric, test_metric: Metric, model_name: str, addition: str):
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("log.txt", 'a') as f:
        f.write("========================\t" + date + "\t========================\n")
        f.write(f"model.md name: \t{model_name}\n")
        f.write(f"addition: \t{addition}\n")
        f.write("train(final): \t\t" + "train loss: {:.4f}\t train accuracy: {:.3f}\t validation loss: {:.4f}\t "
                                       "validation accuracy: {:.3f} \n".format(train_metric.train_loss[-1],
                                                                               train_metric.train_acc[-1],
                                                                               train_metric.val_loss[-1],
                                                                               train_metric.val_acc[-1]))
        f.write("train(max_min): \t" + "train loss: {:.4f}\t train accuracy: {:.3f}\t validation loss: {:.4f}\t "
                                       "validation accuracy: {:.3f} \n".format(min(train_metric.train_loss),
                                                                               max(train_metric.train_acc),
                                                                               min(train_metric.val_loss),
                                                                               max(train_metric.val_acc)))

        f.write("test: \t\t\t\t" + "test loss: \t{:.4f} \t test accuracy:\t {:.3f} \n".format(test_metric.test_loss,
                                                                                              test_metric.test_acc))
        f.write("\n")


def l2_regularization(model, alpha: float):
    l2_loss = []
    for module in model.modules():
        if type(module) is torch.nn.Conv2d or type(module) is torch.nn.Linear:
            l2_loss.append((module.weight ** 2).sum() / 2.0)
    return alpha * sum(l2_loss)


def smooth_labels(labels, factor=0.1):
    # smooth the labels
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels


def noam(d_model, step, warmup):
    fact = min(step ** (-0.5), step * warmup ** (-1.5))
    return fact * (d_model ** (-0.5))
