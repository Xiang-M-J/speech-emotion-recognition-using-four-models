## 预处理
22050Hz采样率, 50ms, 4s

## CNN
****
| 丢弃 | 数据类型 | 标签平滑 |       迭代次数       | 正则化参数 | 学习率 | 验证/训练准确率 | 测试准确率 |
| :--: | :------: | :------: | :------------------: | :--------: | :----: | :-------------: | :--------: |
| 0.2  |   mfcc   |   true   |         100          |    0.1     |  5e-4  |   84.91,  100   |   77.78    |
| 0.4  |   mfcc   |   true   | 400(300时已经收敛了) |    0.3     |  1e-4  |   86.91,98.32   |   82.24    |
| 0.4  |   mfcc   |   true   |         300          |    0.3     |  1e-4  |  85.98, 99.44   |   85.98    |
| 0.4  |   mfcc   |   true   |         100          |    0.3     |  5e-4  |  88.68, 99.77   |   85.18    |
|      |          |          |                      |            |        |                 |            |
|      |          |          |                      |            |        |                 |            |

CNN, scheduler(gamma: 0.5, step_size: 50), adam(beta1: 0.93, beta2: 0.98), random_seed: 34, data_type: mfcc

CNN, scheduler(True, gamma: 0.5, step_size: 100), adam(beta1: 0.93, beta2: 0.98), random_seed: 34, augment: True data_type: mfcc, use_noam: (False, initial_lr: 0.1, warmup: 500) CNN结构中仅对全连接层设置了特殊的dropout参数，两个卷积层的dropout都为0.2

CNN, scheduler(True, gamma: 0.5, step_size: 100), adam(beta1: 0.93, beta2: 0.98), random_seed: 34, augment: Truedata_type: mfcc, use_noam: (False, initial_lr: 0.1, warmup: 500) 两个卷积层的dropout设置为0.1

CNN, scheduler(True, gamma: 0.5, step_size: 100), adam(beta1: 0.93, beta2: 0.98), random_seed: 34, augment: False data_type: mfcc, use_noam: (False, initial_lr: 0.1, warmup: 500) 两个卷积层的dropout设置为0.1

过多的卷积层如3层好像不利于网络泛化

不需要加入残差结构，加入的效果并不好

数据增强后，测试集的准确率有所提高，与验证集最高能达到同一水平，验证集上的准确率变化更加平稳。

加入l2正则化参数会帮助模型泛化，过大的l2正则化参数不利于训练集的训练，但升高学习率可以在一定程度上加速训练集的训练，但对泛化能力没有太大贡献。

升高学习率(如5e-3)，模型的泛化能力会有较大的衰减，降低学习率(如1e-4)，模型的泛化能力会有一定提升，但所需的迭代次数会比较大。

（卷积层）dropout增大好像会影响到模型的测试准确率

## LSTM

****

| 丢弃 | 序号 | 数据类型 | 标签平滑 | 迭代次数 | 正则化参数 | 学习率 | 验证/训练准确率 | 测试准确率 | 日期                |
| :--: | ---- | :------: | :------: | :------: | :--------: | :----: | :-------------: | :--------: | ------------------- |
| 0.3  | 1    |   mfcc   |   true   |   100    |    0.2     |  5e-4  |  86.79, 97.43   |   62.96    |                     |
| 0.3  | 1    |   mfcc   |   true   |   100    |    0.3     |  5e-4  |  88.70, 99.53   |   72.22    |                     |
| 0.3  | 2    |   mfcc   |   true   |   100    |    0.3     |  5e-4  |   88.68, 100    |   83.33    |                     |
| 0.4  | 2    |   mfcc   |   true   |   100    |    0.3     |  5e-4  |    90.57,100    |   79.63    | 2023-03-09 15:40:31 |
| 0.5  | 2    |   mfcc   |   true   |   100    |    0.3     |  5e-4  |    94.34,100    |   77.78    | 2023-03-09 15:42:19 |
| 0.5  | 2    |   mfcc   |   true   |   100    |    0.2     |  5e-4  |    88.68,100    |   85.19    | 2023-03-09 16:08:46 |
| 0.5  | 2    |   mfcc   |   true   |   100    |    0.4     |  5e-4  |    92.45,100    |   79.63    | 2023-03-09 16:11:49 |
| 0.5  | 2    |   mfcc   |   true   |   100    |    0.5     |  5e-4  |   90.57, 100    |   88.89    | 2023-03-09 16:13:14 |
| 0.5  | 2    |   mfcc   |   true   |   100    |    0.5     |  5e-4  |  88.68, 99.53   |   79.63    | 2023-03-09 16:19:13 |
| 0.5  | 2    |   mfcc   |   true   |   100    |    0,5     |  5e-4  |  94.34, 100.0   |   81.48    | 2023_03_09 16:24:37 |

1 ：LSTM, scheduler(gamma: 0.5, step_size: 50), adam(beta1: 0.93, beta2: 0.98), random_seed: 34

2：LSTM, scheduler(True, gamma: 0.5, step_size: 50), adam(beta1: 0.93, beta2: 0.98), random_seed: 34, augment: False data_type: mfcc, use_noam: (False, initial_lr: 0.1, warmup: 500) 不取LSTM的最后一步，直接用全连接层将时间步转成1，之后加入dropout，LSTM只用一层。

如果只取最后一步的输出模型的泛化能力不会太好。

增加dropout会让验证集的准确率上升，但是会影响到测试准确率

过小的L2正则化参数不利于模型的泛化，过大也不利于模型泛化（结果不够稳定），不过可能对测试集上的性能有所帮助

## TIM

| 丢弃 | 编号 | 数据类型 | 标签平滑 | 迭代次数 | 正则化参数 | 学习率 | 验证/训练准确率 | 测试准确率 | 日期                |
| ---- | ---- | -------- | -------- | -------- | ---------- | :----: | --------------- | ---------- | ------------------- |
| 0.1  | 1    | mfcc     | true     | 100      | 0          |  3e-3  | 92.45,98.36     | 70.37      |                     |
| 0.1  | 1    | mfcc     | true     | 100      | 0.1        |  3e-3  | 90.57, 92.52    | 74.07      |                     |
| 0.25 | 2    | mfcc     | true     | 100      | 0          |  5e-3  | 94.34, 100      | 88.89      | 2023-03-09 20:33:51 |
|      |      |          |          |          |            |        |                 |            |                     |
|      |      |          |          |          |            |        |                 |            |                     |

1: TIM, scheduler(gamma: 0.5, step_size: 50), adam(beta1: 0.93, beta2: 0.98), random_seed: 34

2: TIM, scheduler(True, gamma: 0.4, step_size: 50), adam(beta1: 0.93, beta2: 0.98), random_seed: 34, augment: False data_type: mfcc, use_noam: (False, initial_lr: 0.1, warmup: 500) 不在temporal-aware-block中加入SpatialDropout，将filters增加至128，在WeightLayer后加入dropout层

加入正则会影响训练准确率，还是不加比较好，

模型比较复杂，需要较大的学习率来驱动

## Transformer

| 丢弃 | 数据类型 | 编号 | 标签平滑 | 迭代次数 | 正则化参数 |     学习率     | 验证/训练准确率 | 测试准确率 |        日期         |
| :--: | :------: | :--: | :------: | :------: | :--------: | :------------: | :-------------: | :--------: | :-----------------: |
| 0.2  |   mfcc   |  1   |   true   |   100    |    0.2     | noam(0.1, 500) |   83.02, 100    |   81.48    | 2023-03-08 10:07:51 |
| 0.2  |   mfcc   |  2   |   true   |   100    |    0.2     |      6e-5      |    84.91,100    |   85.19    | 2023-03-08 10:31:44 |
| 0.1  |   mfcc   |  4   |   true   |   100    |     0      | noam(0.1, 300) |  [88.68, 100]   |   75.93    | 2023-03-10 13:10:34 |
| 0.2  |   mfcc   |  4   |   true   |   100    |    0.1     |      6e-5      |  [92.45,96.03]  |   87.04    | 2023-03-10 13:42:52 |
| 0.1  |   mfcc   |  4   |   true   |   100    |    0.1     |      6e-5      | [86.79, 99.53]  |   87.04    | 2023-03-10 13:24:51 |
| 0.1  |   mfcc   |  4   |   true   |   100    |    0.1     |      6e-5      | [90.57, 99.53]  |   79.63    | 2023-03-10 15:28:33 |

1: Transformer, scheduler(False, gamma: 0.5, step_size: 50), adam(beta1: 0.93, beta2: 0.98), random_seed: 34, data_type: mfcc, use_noam: (True, initial_lr: 0.1, warmup: 500)

2: Transformer, scheduler(True, gamma: 0.5, step_size: 25), adam(beta1: 0.93, beta2: 0.98), random_seed: 34, data_type: mfcc, use_noam: (False, initial_lr: 0.1, warmup: 500)

4: Transformer, scheduler(False, gamma: 0.8, step_size: 50), adam(beta1: 0.93, beta2: 0.98), random_seed: 34, augment: False data_type: mfcc, use_noam: (True, initial_lr: 0.1, warmup: 300) 两层卷积卷到256维，再输入transformer中,transformer: n_head=8, n_layers=4, dim_feedforward=1024，卷积层和transformer的dropout设为0，不加mask。

1/2 ：一层卷积，d_model=512, dim_feedforward, n_layers = 3, n_head=8, 用AdaptiveAvgPool1d

当模型比较大的时候，最好选择小一点的学习率

不加mask，相当于加噪声了，模型的泛化能力可能更好一点



前向后向因果卷积，将结果送到transformer中？

**召回率：**样本中的正例有多少被预测正确了

**精确率：**预测为正的样本中有多少是真正的正样本