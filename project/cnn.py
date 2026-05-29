import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# ========= 全局参数 =========
k_n = 4
k_s = 3
f_c = 64
ir = 0.0001
num_epochs = 20000

# ========= 读取数据 =========
data = pd.read_csv('../1.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# ========= 数据增强（跳过对称序列）=========
X_flip, y_flip = [], []
for i in range(len(X)):
    seq = X[i]
    flipped = np.flip(seq, axis=0)
    if not np.array_equal(seq, flipped):
        X_flip.append(flipped)
        y_flip.append(y[i])

X_aug = np.concatenate((X, X_flip), axis=0)
y_aug = np.concatenate((y, y_flip), axis=0)

indices = np.arange(len(X_aug))
np.random.shuffle(indices)
X_aug = X_aug[indices]
y_aug = y_aug[indices]

# ========= 数据集划分 =========
X_train, X_val, y_train, y_val = train_test_split(
    X_aug, y_aug, test_size=0.2, random_state=43, shuffle=True
)

X_train = torch.Tensor(X_train.reshape(-1, 1, 20))
y_train = torch.Tensor(y_train.reshape(-1, 1))
X_val = torch.Tensor(X_val.reshape(-1, 1, 20))
y_val = torch.Tensor(y_val.reshape(-1, 1))

# 保存标签
np.savetxt(f'y-train_kn_{k_n}_ks_{k_s}.txt', y_train.numpy(), fmt='%.4f')
np.savetxt(f'y-test_kn_{k_n}_ks_{k_s}.txt', y_val.numpy(), fmt='%.4f')

# ========= 模型定义 =========
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, k_n, kernel_size=k_s)

        # 第一个卷积核全 1
        self.conv1.weight.data[0] = torch.ones((1, k_s))

        # 其他卷积核随机 0/1
        if k_n > 1:
            rand_k = torch.randint(0, 2, (k_n - 1, 1, k_s)).float()
            self.conv1.weight.data[1:] = rand_k

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(k_n * (20 - k_s  + 1), f_c)  ##20-s+1
        self.fc2 = nn.Linear(f_c, 1)
        self.activation = nn.SiLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


model = CNN()
optimizer = optim.Adam(model.parameters(), lr=ir)
criterion = nn.MSELoss()

# 保存初始卷积权重
with open('ori_weight.txt', 'w') as f:
    f.write(str(model.conv1.weight.data.cpu().numpy()))

# ========= 前向钩子：只保存最终一次 activation =========
def get_last_activation(model, X):
    acts = []

    def hook_fn(module, inp, out):
        arr = out.detach().cpu().numpy()
        arr2 = arr.reshape(arr.shape[0], -1)
        acts.append(arr2)

    hook = model.conv1.register_forward_hook(hook_fn)
    _ = model(X)
    hook.remove()

    return acts

# ========= 训练函数（增加每个epoch的指标输出）=========
def train_fast(model, X_train, y_train, X_val, y_val):
    model.train()
    
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每个epoch计算训练集和验证集指标
        model.eval()
        with torch.no_grad():
            # 训练集预测
            train_pred = model(X_train)
            train_mse = mean_squared_error(y_train.numpy(), train_pred.numpy())
            train_r2 = r2_score(y_train.numpy(), train_pred.numpy())
            
            # 验证集预测
            val_pred = model(X_val)
            val_mse = mean_squared_error(y_val.numpy(), val_pred.numpy())
            val_r2 = r2_score(y_val.numpy(), val_pred.numpy())
        
        # 打印epoch信息
        if (epoch + 1) % 1 == 0:  # 每个epoch都输出
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, "
                  f"Train MSE: {train_mse:.4f}, Train R2: {train_r2:.4f}, "
                  f"Val MSE: {val_mse:.4f}, Val R2: {val_r2:.4f}")
        
        # 切换回训练模式
        model.train()

    # ======= 训练结束后，只保存最后一次结果 =======
    model.eval()

    # 最终 activation（对 X_train）
    final_acts = get_last_activation(model, X_train)

    with open(f'final_activation_kn_{k_n}_ks_{k_s}.txt', 'w') as f:
        for act in final_acts:
            # act 是 (num_samples, features)
            for row in act:
                f.write(", ".join(map(lambda v: f"{v:.6f}", row)) + "\n")

    # 最终卷积权重
    with open(f'final_weights_kn_{k_n}_ks_{k_s}.txt', 'w') as f:
        w = model.conv1.weight.detach().cpu().numpy()
        for k in w:
            f.write(", ".join(map(lambda x: f"{x:.6f}", k.flatten())) + "\n")

    # 最终训练/验证性能与预测
    pred_tr = model(X_train).detach().cpu().numpy()   # shape (N,1)
    pred_val = model(X_val).detach().cpu().numpy()

    ytr = y_train.detach().cpu().numpy()
    yv = y_val.detach().cpu().numpy()

    mse_tr = mean_squared_error(ytr, pred_tr)
    mse_v = mean_squared_error(yv, pred_val)
    r2_tr = r2_score(ytr, pred_tr)
    r2_v = r2_score(yv, pred_val)

    # 保存最终的性能
    with open(f'final_r2_mse_kn_{k_n}_ks_{k_s}.txt', 'w') as f:
        f.write(f"Train MSE: {mse_tr:.6f}\n")
        f.write(f"Train R2: {r2_tr:.6f}\n")
        f.write(f"Val MSE: {mse_v:.6f}\n")
        f.write(f"Val R2: {r2_v:.6f}\n")

    # ====== 保存最终预测 ======
    np.savetxt(f'train-pre_kn_{k_n}_ks_{k_s}.txt', pred_tr.reshape(-1), fmt='%.6f')
    np.savetxt(f'test-pre_kn_{k_n}_ks_{k_s}.txt', pred_val.reshape(-1), fmt='%.6f')

    # 终端打印前10个预测以便快速检查
    print("\nFirst 10 train predictions:", pred_tr.reshape(-1)[:10])
    print("First 10 val predictions:", pred_val.reshape(-1)[:10])
    print(f"\nFinal Train MSE: {mse_tr:.6f}, Train R2: {r2_tr:.6f}")
    print(f"Final Val   MSE: {mse_v:.6f},   Val R2: {r2_v:.6f}")

    return model, (mse_tr, r2_tr, mse_v, r2_v)


# ========= 开始训练 =========
model, final_metrics = train_fast(model, X_train, y_train, X_val, y_val)

# 保存模型
torch.save(model.state_dict(), "model.pth")

# ========= 保存 X_train（与你原来一致）=========
X_train_np = X_train.numpy()
with open(f'x_train_kn_{k_n}_ks_{k_s}.txt', 'w') as f:
    for row in X_train_np.reshape(X_train_np.shape[0], -1):
        f.write(" ".join(map(lambda v: f"{v:.6f}", row)) + "\n")

# 列格式保存
with open(f'x_train_ver_kn_{k_n}_ks_{k_s}.txt', 'w') as f:
    for row in X_train_np.reshape(X_train_np.shape[0], -1):
        for v in row:
            f.write(f"{v:.6f}\n")
            
# ============================================================
# 合并绘图：Train 与 Test 的真实值 vs 预测值（同一张图）
# ============================================================
import matplotlib.pyplot as plt

# 转 numpy（确保在 CPU 上）
y_tr = y_train.detach().cpu().numpy().reshape(-1)
y_te = y_val.detach().cpu().numpy().reshape(-1)
pred_tr = model(X_train).detach().cpu().numpy().reshape(-1)
pred_te = model(X_val).detach().cpu().numpy().reshape(-1)

# 计算指标
mse_tr = mean_squared_error(y_tr, pred_tr)
r2_tr = r2_score(y_tr, pred_tr)
mse_te = mean_squared_error(y_te, pred_te)
r2_te = r2_score(y_te, pred_te)

# 画图
plt.figure(figsize=(7, 6))

# 两组散点
plt.scatter(y_tr, pred_tr, s=30, alpha=0.6, label=f'Train (n={len(y_tr)})', marker='o')
plt.scatter(y_te, pred_te, s=40, alpha=0.7, label=f'Test (n={len(y_te)})', marker='^', color='orange')

# 对角线 y=x
ymin = min(y_tr.min(), y_te.min(), pred_tr.min(), pred_te.min())
ymax = max(y_tr.max(), y_te.max(), pred_tr.max(), pred_te.max())
pad = (ymax - ymin) * 0.05
plt.plot([ymin - pad, ymax + pad], [ymin - pad, ymax + pad], 'k--', linewidth=1)

plt.xlabel("True values", fontsize=12)
plt.ylabel("Predicted values", fontsize=12)
plt.title("True vs Predicted (Train & Test)", fontsize=14)
plt.legend(framealpha=0.9)

# 在图上显示两个箱子，分别列出 Train 与 Test 的 R2 和 MSE
textstr = (
    f"Train:  R²={r2_tr:.4f}\nTrain MSE={mse_tr:.4e}\n\n"
    f"Test:   R²={r2_te:.4f}\nTest MSE={mse_te:.4e}"
)
plt.gca().text(
    0.02, 0.98, textstr,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
)

plt.xlim(ymin - pad, ymax + pad)
plt.ylim(ymin - pad, ymax + pad)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("combined_train_test_pred.png", dpi=300)
# plt.show()
print("Saved combined_train_test_pred.png")


# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib.pyplot as plt
# import os

# # ========= 全局参数 =========
# k_n = 4
# k_s = 3
# f_c = 64
# ir = 0.0001
# num_epochs = 20000

# # ========= 读取数据 =========
# data = pd.read_csv('../1.csv')
# X = data.iloc[:, :-1].values
# y = data.iloc[:, -1].values

# # ========= 数据增强（跳过对称序列）=========
# X_flip, y_flip = [], []
# for i in range(len(X)):
#     seq = X[i]
#     flipped = np.flip(seq, axis=0)
#     if not np.array_equal(seq, flipped):
#         X_flip.append(flipped)
#         y_flip.append(y[i])

# X_aug = np.concatenate((X, X_flip), axis=0)
# y_aug = np.concatenate((y, y_flip), axis=0)

# indices = np.arange(len(X_aug))
# np.random.shuffle(indices)
# X_aug = X_aug[indices]
# y_aug = y_aug[indices]

# # ========= 数据集划分 =========
# X_train, X_val, y_train, y_val = train_test_split(
#     X_aug, y_aug, test_size=0.2, random_state=43, shuffle=True
# )

# X_train = torch.Tensor(X_train.reshape(-1, 1, 20))
# y_train = torch.Tensor(y_train.reshape(-1, 1))
# X_val = torch.Tensor(X_val.reshape(-1, 1, 20))
# y_val = torch.Tensor(y_val.reshape(-1, 1))

# # 保存标签
# np.savetxt(f'y-train_kn_{k_n}_ks_{k_s}.txt', y_train.numpy(), fmt='%.4f')
# np.savetxt(f'y-test_kn_{k_n}_ks_{k_s}.txt', y_val.numpy(), fmt='%.4f')

# # ========= 模型定义 =========
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv1d(1, k_n, kernel_size=k_s)

#         # 第一个卷积核全 1
#         self.conv1.weight.data[0] = torch.ones((1, k_s))

#         # 其他卷积核随机 0/1
#         if k_n > 1:
#             rand_k = torch.randint(0, 2, (k_n - 1, 1, k_s)).float()
#             self.conv1.weight.data[1:] = rand_k

#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(k_n * 18, f_c)
#         self.fc2 = nn.Linear(f_c, 1)
#         self.activation = nn.SiLU()

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.activation(x)
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.activation(x)
#         x = self.fc2(x)
#         return x


# model = CNN()
# optimizer = optim.Adam(model.parameters(), lr=ir)
# criterion = nn.MSELoss()

# # 保存初始卷积权重
# with open('ori_weight.txt', 'w') as f:
#     f.write(str(model.conv1.weight.data.cpu().numpy()))

# # ========= 前向钩子：只保存最终一次 activation =========
# def get_last_activation(model, X):
#     acts = []

#     def hook_fn(module, inp, out):
#         arr = out.detach().cpu().numpy()
#         arr2 = arr.reshape(arr.shape[0], -1)
#         acts.append(arr2)

#     hook = model.conv1.register_forward_hook(hook_fn)
#     _ = model(X)
#     hook.remove()

#     return acts

# # ========= 训练函数（极大提速版）=========
# def train_fast(model, X_train, y_train, X_val, y_val):
#     model.train()
#     for epoch in range(num_epochs):
#         # 前向
#         outputs = model(X_train)
#         loss = criterion(outputs, y_train)

#         # 反向
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if (epoch + 1) % 2000 == 0:
#             print(f"Epoch {epoch+1}/{num_epochs}, Loss={loss.item():.6f}")

#     # ======= 训练结束后，只保存最后一次结果 =======
#     model.eval()

#     # 最终 activation（对 X_train）
#     final_acts = get_last_activation(model, X_train)

#     with open(f'final_activation_kn_{k_n}_ks_{k_s}.txt', 'w') as f:
#         for act in final_acts:
#             # act 是 (num_samples, features)
#             for row in act:
#                 f.write(", ".join(map(lambda v: f"{v:.6f}", row)) + "\n")

#     # 最终卷积权重
#     with open(f'final_weights_kn_{k_n}_ks_{k_s}.txt', 'w') as f:
#         w = model.conv1.weight.detach().cpu().numpy()
#         for k in w:
#             f.write(", ".join(map(lambda x: f"{x:.6f}", k.flatten())) + "\n")

#     # 最终训练/验证性能与预测
#     pred_tr = model(X_train).detach().cpu().numpy()   # shape (N,1)
#     pred_val = model(X_val).detach().cpu().numpy()

#     ytr = y_train.detach().cpu().numpy()
#     yv = y_val.detach().cpu().numpy()

#     mse_tr = mean_squared_error(ytr, pred_tr)
#     mse_v = mean_squared_error(yv, pred_val)
#     r2_tr = r2_score(ytr, pred_tr)
#     r2_v = r2_score(yv, pred_val)

#     # 保存最终的性能
#     with open(f'final_r2_mse_kn_{k_n}_ks_{k_s}.txt', 'w') as f:
#         f.write(f"Train MSE: {mse_tr:.6f}\n")
#         f.write(f"Train R2: {r2_tr:.6f}\n")
#         f.write(f"Val MSE: {mse_v:.6f}\n")
#         f.write(f"Val R2: {r2_v:.6f}\n")

#     # ====== 保存最终预测（你指出缺少的部分） ======
#     np.savetxt(f'train-pre_kn_{k_n}_ks_{k_s}.txt', pred_tr.reshape(-1), fmt='%.6f')
#     np.savetxt(f'test-pre_kn_{k_n}_ks_{k_s}.txt', pred_val.reshape(-1), fmt='%.6f')

#     # 终端打印前10个预测以便快速检查
#     print("\nFirst 10 train predictions:", pred_tr.reshape(-1)[:10])
#     print("First 10 val predictions:", pred_val.reshape(-1)[:10])
#     print(f"\nFinal Train MSE: {mse_tr:.6f}, Train R2: {r2_tr:.6f}")
#     print(f"Final Val   MSE: {mse_v:.6f},   Val R2: {r2_v:.6f}")

#     return model, (mse_tr, r2_tr, mse_v, r2_v)


# # ========= 开始训练 =========
# model, final_metrics = train_fast(model, X_train, y_train, X_val, y_val)

# # 保存模型
# torch.save(model.state_dict(), "model.pth")

# # ========= 保存 X_train（与你原来一致）=========
# X_train_np = X_train.numpy()
# with open(f'x_train_kn_{k_n}_ks_{k_s}.txt', 'w') as f:
#     for row in X_train_np.reshape(X_train_np.shape[0], -1):
#         f.write(" ".join(map(lambda v: f"{v:.6f}", row)) + "\n")

# # 列格式保存
# with open(f'x_train_ver_kn_{k_n}_ks_{k_s}.txt', 'w') as f:
#     for row in X_train_np.reshape(X_train_np.shape[0], -1):
#         for v in row:
#             f.write(f"{v:.6f}\n")
# # ============================================================
# # 合并绘图：Train 与 Test 的真实值 vs 预测值（同一张图）
# # ============================================================
# import matplotlib.pyplot as plt

# # 转 numpy（确保在 CPU 上）
# y_tr = y_train.detach().cpu().numpy().reshape(-1)
# y_te = y_val.detach().cpu().numpy().reshape(-1)
# pred_tr = model(X_train).detach().cpu().numpy().reshape(-1)
# pred_te = model(X_val).detach().cpu().numpy().reshape(-1)

# # 计算指标
# mse_tr = mean_squared_error(y_tr, pred_tr)
# r2_tr = r2_score(y_tr, pred_tr)
# mse_te = mean_squared_error(y_te, pred_te)
# r2_te = r2_score(y_te, pred_te)

# # 画图
# plt.figure(figsize=(7, 6))

# # 两组散点
# plt.scatter(y_tr, pred_tr, s=30, alpha=0.6, label=f'Train (n={len(y_tr)})', marker='o')
# plt.scatter(y_te, pred_te, s=40, alpha=0.7, label=f'Test (n={len(y_te)})', marker='^', color='orange')

# # 对角线 y=x
# ymin = min(y_tr.min(), y_te.min(), pred_tr.min(), pred_te.min())
# ymax = max(y_tr.max(), y_te.max(), pred_tr.max(), pred_te.max())
# pad = (ymax - ymin) * 0.05
# plt.plot([ymin - pad, ymax + pad], [ymin - pad, ymax + pad], 'k--', linewidth=1)

# plt.xlabel("True values", fontsize=12)
# plt.ylabel("Predicted values", fontsize=12)
# plt.title("True vs Predicted (Train & Test)", fontsize=14)
# plt.legend(framealpha=0.9)

# # 在图上显示两个箱子，分别列出 Train 与 Test 的 R2 和 MSE
# textstr = (
#     f"Train:  R²={r2_tr:.4f}\nTrain MSE={mse_tr:.4e}\n\n"
#     f"Test:   R²={r2_te:.4f}\nTest MSE={mse_te:.4e}"
# )
# plt.gca().text(
#     0.02, 0.98, textstr,
#     transform=plt.gca().transAxes,
#     fontsize=10,
#     verticalalignment='top',
#     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
# )

# plt.xlim(ymin - pad, ymax + pad)
# plt.ylim(ymin - pad, ymax + pad)
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig("combined_train_test_pred.png", dpi=300)
# # plt.show()
# print("Saved combined_train_test_pred.png")

