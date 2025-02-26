# 导包
import torch
import torch.nn as nn                          #神经网络
import torch.optim as optim                    #定义优化器
from torchvision import datasets,transforms    #数据集    transforms完成对数据的处理
import matplotlib.pyplot as plt
from rich import print
from rich.progress import track
# 定义超参数
input_size = 28 * 28  #输入大小
hidden_size_1 = 512  # 第一个隐藏层的大小
hidden_size_2 = 512
# hidden_size_3 = 128
# hidden_size_4 = 64
# hidden_size_5 = 32
num_classes = 10 #输出大小（类别数）
batch_size = 100#批大小
learning_rate = 0.0001 #学习率
num_epochs = 8 #训练轮数

# 加载 MNIST 数据集
train_dataset = datasets.MNIST(root='../data/mnist',train=True,transform=transforms.ToTensor(),download=True)
test_dataset = datasets.MNIST(root='../data/mnist',train=False,transform=transforms.ToTensor(),download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)   #一批数据为100个
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 定义 MLP 网络
class MLP(nn.Module):
    # 初始化方法
    # input_size 输入数据的维度
    # hidden_size 隐藏层的大小
    # num_classes 输出分类的数量
    def __init__(self,input_size,hidden_size_1,num_classes):
        # 调用父类的初始化方法
        super(MLP,self).__init__()
        # 定义第1个全连接层
        self.fc1 = nn.Linear(input_size,hidden_size_1)
        # 定义ReLu激活函数
        self.relu = nn.ReLU()
        # 定义第2个全连接层 # 第二个激活函数



        self.fc6 = nn.Linear(hidden_size_1, num_classes)  # 输出层
        
    def forward(self,x):
        out = self.fc1(x)
        out = self.relu(out)


        out = self.fc6(out)
        return out

# 实例化 MLP 网络
model = MLP(input_size, hidden_size_1, num_classes).to(device)


# 现在我们已经定义了 MLP 网络并加载了 MNIST 数据集，接下来使用 PyTorch 的自动求导功能和优化器进行训练。首先，定义损失函数和优化器；然后迭代训练数据并使用优化器更新网络参数。

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
# CrossEntropyLoss = Softmax + log + nllloss
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
# optimizer = optim.SGD(model.parameters(),0.2)
epoch_losses = []
# 训练网络
# 外层for循环控制训练的次数
# 内层for循环控制从DataLoader中循环读取数据
# 训练网络
plt.imshow(train_dataset[0][0][0],cmap='gray')   # 显示第一个训练数据        
plt.show()
for epoch in track(range(num_epochs), description="Training..."):
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 28 * 28).to(device)  # 将images转换成向量并移动到GPU
        labels = labels.to(device)  # 将labels移动到GPU
        
        outputs = model(images)  # 将数据送到网络中
        loss = criterion(outputs, labels)  # 计算损失
        
        optimizer.zero_grad()  # 首先将梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        
        total_loss += loss.item()  # 累加损失
        
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step[{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    # 计算每个epoch的平均损失并存储
    epoch_loss = total_loss / len(train_loader)
    epoch_losses.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}] Average Loss: {epoch_loss:.4f}')

# 绘制损失曲线


# 测试网络
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.view(-1, 28 * 28).to(device)  # 将images转换成向量并移动到GPU
        labels = labels.to(device)  # 将labels移动到GPU
        outputs = model(images)  # 将数据传送到网络
        _, predicted = torch.max(outputs.data, 1)  # 返回一个元组：第一个为最大值，第二个是最大值的下标
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'{num_epochs}: Accuracy of the network on the 10000 test images: {100 * correct / total}%')
# 保存模型
torch.onnx.export(model, images,  "mlp_only_1.onnx", export_params=True,verbose=True)
plt.figure(figsize=(20, 10))
plt.plot(range(1, num_epochs + 1), epoch_losses)
plt.title("Loss Curve ")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

