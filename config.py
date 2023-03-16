beta1 = 0.93
beta2 = 0.98
gamma = 0.3
step_size = 50
random_seed = 34
data_type = "mfcc"  # ["mfcc", "mel"]
save = False  # 是否保存结果与模型
augment = False  # 是否使用增强后的数据
use_scheduler = False
use_noam = False  # 用于Transformer
warmup = 300  # noam参数
initial_lr = 0.08  # 初始学习率(noam)
