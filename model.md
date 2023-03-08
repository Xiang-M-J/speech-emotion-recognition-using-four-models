## 预处理
22050Hz采样率, 25ms, 4s

## CNN
****
| 丢弃 | 数据类型 | 标签平滑 |       迭代次数       | 正则化参数 | 学习率 | 验证/训练准确率 | 测试准确率 |
| :--: | :------: | :------: | :------------------: | :--------: | :----: | :-------------: | :--------: |
| 0.2  |   mfcc   |   true   |         100          |    0.1     |  5e-4  |   84.91,  100   |   77.78    |
| 0.4  |   mfcc   |   true   | 400(300时已经收敛了) |    0.3     |  1e-4  |   86.91,98.32   |   82.24    |
| 0.4  |   mfcc   |   true   |         300          |    0.3     |  1e-4  |  85.98, 99.44   |   85.98    |
|      |          |          |                      |            |        |                 |            |
|      |          |          |                      |            |        |                 |            |
|      |          |          |                      |            |        |                 |            |

CNN, scheduler(gamma: 0.5, step_size: 50), adam(beta1: 0.93, beta2: 0.98), random_seed: 34, data_type: mfcc

CNN, scheduler(True, gamma: 0.5, step_size: 100), adam(beta1: 0.93, beta2: 0.98), random_seed: 34, augment: True data_type: mfcc, use_noam: (False, initial_lr: 0.1, warmup: 500) CNN结构中仅对全连接层设置了特殊的dropout参数，两个卷积层的dropout都为0.2

CNN, scheduler(True, gamma: 0.5, step_size: 100), adam(beta1: 0.93, beta2: 0.98), random_seed: 34, augment: Truedata_type: mfcc, use_noam: (False, initial_lr: 0.1, warmup: 500) 两个卷积层的dropout设置为0.1

过多的卷积层如3层好像不利于网络泛化

不需要加入残差结构，加入的效果并不好

数据增强后，测试集的准确率有所提高，与验证集最高能达到同一水平，验证集上的准确率变化更加平稳。

加入l2正则化参数会帮助模型泛化，过大的l2正则化参数不利于训练集的训练，但升高学习率可以在一定程度上加速训练集的训练，但对泛化能力没有太大贡献。

升高学习率(如5e-3)，模型的泛化能力会有较大的衰减，降低学习率(如1e-4)，模型的泛化能力会有一定提升，但所需的迭代次数会比较大。

（卷积层）dropout增大好像会影响到模型的测试准确率

## LSTM

****

| 丢弃 | 数据类型 | 标签平滑 | 迭代次数 | 正则化参数 | 学习率 | 验证/训练准确率 | 测试准确率 |
| :--: | :------: | :------: | :------: | :--------: | :----: | :-------------: | :--------: |
| 0.3  |   mfcc   |   true   |   100    |    0.2     |  5e-4  |  86.79, 97.43   |   62.96    |
| 0.3  |   mfcc   |   true   |   100    |    0.3     |  5e-4  |  88.70, 99.53   |   72.22    |
|      |          |          |          |            |        |                 |            |
|      |          |          |          |            |        |                 |            |
|      |          |          |          |            |        |                 |            |

LSTM, scheduler(gamma: 0.5, step_size: 50), adam(beta1: 0.93, beta2: 0.98), random_seed: 34

## TIM

| 丢弃  | 数据类型 | 标签平滑 | 迭代次数 | 正则化参数 | 学习率  | 验证/训练准确率     | 测试准确率 |
|-----|------|------|------|-------|:----:|--------------|-------|
| 0.1 | mfcc | true | 100  | 0     | 3e-3 | 92.45,98.36  | 70.37 |
| 0.1 | mfcc | true | 100  | 0.1   | 3e-3 | 90.57, 92.52 | 74.07 |
|     |      |      |      |       |      |              |       |
|     |      |      |      |       |      |              |       |
|     |      |      |      |       |      |              |       |

TIM, scheduler(gamma: 0.5, step_size: 50), adam(beta1: 0.93, beta2: 0.98), random_seed: 34

加入正则会影响训练准确率，还是不加比较好

## Transformer

| 丢弃 | 数据类型 | 编号 | 标签平滑 | 迭代次数 | 正则化参数 |     学习率     | 验证/训练准确率 | 测试准确率 |        日期         |
| :--: | :------: | :--: | :------: | :------: | :--------: | :------------: | :-------------: | :--------: | :-----------------: |
| 0.2  |   mfcc   |  1   |   true   |   100    |    0.2     | noam(0.1, 500) |   83.02, 100    |   81.48    | 2023-03-08 10:07:51 |
| 0.2  |   mfcc   |  2   |   true   |   100    |    0.2     |      6e-5      |    84.91,100    |   85.19    | 2023-03-08 10:31:44 |

Transformer, scheduler(False, gamma: 0.5, step_size: 50), adam(beta1: 0.93, beta2: 0.98), random_seed: 34, data_type: mfcc, use_noam: (True, initial_lr: 0.1, warmup: 500)

Transformer, scheduler(True, gamma: 0.5, step_size: 25), adam(beta1: 0.93, beta2: 0.98), random_seed: 34, data_type: mfcc, use_noam: (False, initial_lr: 0.1, warmup: 500)

1/2 ：一层卷积，d_model=512, dim_feedforward, n_layers = 3, n_head=8, 用AdaptiveAvgPool1d

两层卷积的效果好像没有一层卷积的效果好

当模型比较大的时候，最好选择小一点的学习率

