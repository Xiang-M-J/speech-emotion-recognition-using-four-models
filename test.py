# from utils import noam
# import matplotlib.pyplot as plt
# from config import warmup, initial_lr
# from ray import tune
# # lr = 0.1
# noam_lr = []
# d_model = 39
# # # warmup = 300
#
# for i in range(2700):
#     noam_lr.append(initial_lr * noam(d_model=d_model, step=i + 1, warmup=warmup))
#
# plt.plot(noam_lr)
# plt.show()

# a = tune.loguniform(1e-4, 1e-1)
# b = tune.choice([2,3,4,5])
# print(int(b))
# for p in optimizer.param_groups:
#     p['lr'] *= 0.9
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

#### Scale the training data ####
# store shape so we can transform it back

N,C,H,W = X_train.shape
# Reshape to 1D because StandardScaler operates on a 1D array
# tell numpy to infer shape of 1D array with '-1' argument
X_train = np.reshape(X_train, (N,-1))
X_train = scaler.fit_transform(X_train)