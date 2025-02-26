import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
# 1. 生成数据
# 创建一些模拟数据，假设 y = 3x + 2 + 噪声
torch.manual_seed(30)  # 设置随机种子
x = torch.linspace(0, 10, 100).unsqueeze(1)  # 输入数据 (100, 1)
y = 3 * x + 2 + torch.randn(100, 1) * 2  # 输出数据带噪声 (100, 1)

# 2. 定义一个简单的线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入特征维度1，输出特征维度1

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.SGD(model.parameters(), lr=0.02)  # 随机梯度下降优化器

# 4. 训练模型
epochs = 100  # 迭代次数
losses = []  # 记录损失

for epoch in range(epochs):
    model.train()  # 设置为训练模式
    optimizer.zero_grad()  # 清空梯度

    predictions = model(x)  # 模型预测
    loss = criterion(predictions, y)  # 计算损失
    loss.backward()  # 反向传播计算梯度
    optimizer.step()  # 更新模型参数

    losses.append(loss.item())  # 记录损失

    # 每100次输出一次训练信息
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 5. 可视化训练结果
# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(range(epochs), losses)
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.show()

# 绘制预测结果
model.eval()  # 设置为评估模式
with torch.no_grad():
    predicted = model(x)  # 模型预测

plt.figure(figsize=(10, 5))
plt.scatter(x.numpy(), y.numpy(), label="Ground Truth")  # 原始数据
plt.plot(x.numpy(), predicted.numpy(), color="red", label="Predicted")  # 预测结果
plt.legend()
plt.title("Linear Regression Result")
plt.show()
