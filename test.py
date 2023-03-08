from utils import noam
import matplotlib.pyplot as plt
from config import warmup, initial_lr
# lr = 0.1
noam_lr = []
d_model = 512
# warmup = 300

for i in range(2700):
    noam_lr.append(initial_lr * noam(d_model=d_model, step=i + 1, warmup=warmup))

plt.plot(noam_lr)
plt.show()

# for p in optimizer.param_groups:
#     p['lr'] *= 0.9
