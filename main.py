import numpy as np
from utils import myLoader
import torch
from torch.utils.data import dataloader
from Net import Net_Instance
from config import beta1, beta2, gamma, step_size, random_seed, data_type, save, augment, use_noam, use_scheduler, \
    initial_lr, warmup

model_type = "Transformer"  # 模型选择
model_index = 6  # 模型编号
epochs = 100  # 迭代次数
lr = 8e-5  # 学习率
batch_size = 16  # 批次大小
if augment:
    spilt_rate = [0.6, 0.2, 0.2]
else:
    spilt_rate = [0.8, 0.1, 0.1]  # 训练集、验证集、测试集分割比例
drop_rate = 0.25  # dropout
feature_dim = 39  # 特征维度
num_class = 7  # 类别数
weight_decay = 0.1  # l2正则化参数
smooth = True  # 是否平滑标签（True：平滑 False：不平滑）

if __name__ == "__main__":
    if data_type == "mel":
        x = np.load("preprocess/x_mel.npy")
        feature_dim = 40
    else:
        x = np.load("preprocess/x_mfcc.npy")
    y = np.load("preprocess/y.npy")
    model_name = f"{model_type}-{model_index}_{augment}_drop{str(drop_rate).split('.')[-1]}_{data_type}" \
                 f"_smooth{smooth}_epoch{epochs}_l2re{str(weight_decay).split('.')[-1]}_lr{str(lr).split('.')[-1]}"
    addition = f"{model_type}, scheduler({use_scheduler}, gamma: {gamma}, step_size: {step_size})," \
               f" adam(beta1: {beta1}, beta2: {beta2}), random_seed: {random_seed}, augment: {augment} " \
               f" data_type: {data_type}, use_noam: ({use_noam}, initial_lr: {initial_lr}, warmup: {warmup})"
    option = input(f"{save} 模型名与最后保存的结果有关，请确定模型名为：{model_name}, (y(default)/n):")
    if option == '' or option == 'y' or option == 'yes' or option is None:
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
        instance = Net_Instance(model_type=model_type, model_name=model_name, addition=addition,
                                feature_dim=feature_dim, num_class=num_class)
        instance.train(train_dataset, val_dataset, batch_size, epochs, lr, weight_decay, drop_rate, smooth,
                       is_mask=False)
        instance.test(test_dataset, batch_size=batch_size)
    else:
        print("请修改模型名后再次执行")
