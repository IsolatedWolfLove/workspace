import torch
import torch.nn as nn
import torch.optim as optim                    #定义优化器
from torchvision import datasets,transforms    #数据集    transforms完成对数据的处理
import matplotlib.pyplot as plt
from rich import print
from rich.progress import track
from torchsummary import summary
num_classes = 10 #输出大小（类别数）
batch_size = 600#批大小
learning_rate = 0.002 #学习率
num_epochs = 150 #训练轮数
input_size = 28 * 28 #输入大小（图片大小）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = datasets.MNIST(root='../data/mnist',train=True,transform=transforms.ToTensor(),download=True)
test_dataset = datasets.MNIST(root='../data/mnist',train=False,transform=transforms.ToTensor(),download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)   #一批数据为100个
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        # 卷积层1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        # 池化(汇聚)层1
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # 卷积层2

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        # 池化(汇聚)层2
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # 展开
        self.flatten = nn.Flatten()

        # 三个全连接层
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
        # 声明两个激活函数
        self.act = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.pool1(y)
        y = self.act(self.conv2(y))
        y = self.pool2(y)
        y = self.flatten(y)

        y = self.act(self.fc1(y))

        y = self.act(self.fc2(y))

        y = self.fc3(y)
        
        y = self.softmax(y)
        
        return y


# 创建模型实例
model = LeNet(num_classes=10)

# 将模型放到设备上 (CPU 或 GPU)

model.to(device)
print(device)
# 使用 torchsummary 显示模型摘要
summary(model, input_size=(1, 28, 28))
criterion = nn.CrossEntropyLoss()
# CrossEntropyLoss = Softmax + log + nllloss
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
# optimizer = optim.SGD(model.parameters(),0.2)
epoch_losses = []
for epoch in track(range(num_epochs), description="Training..."):
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)  # 将images转换成向量并移动到GPU
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
        images = images.to(device)  # 将images转换成向量并移动到GPU
        labels = labels.to(device)  # 将labels移动到GPU
        outputs = model(images)  # 将数据传送到网络
        _, predicted = torch.max(outputs.data, 1)  # 返回一个元组：第一个为最大值，第二个是最大值的下标
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'{num_epochs}: Accuracy of the network on the 10000 test images: {100 * correct / total}%')
# 保存模型
torch.onnx.export(model, images,  "cnn.onnx", export_params=True,verbose=True)
plt.figure(figsize=(20, 10))
plt.plot(range(1, num_epochs + 1), epoch_losses)
plt.title("Loss Curve ")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

'''
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F

"""
卷积运算 使用mnist数据集，和10-4，11类似的，只是这里：1.输出训练轮的acc 2.模型上使用torch.nn.Sequential
"""
# Super parameter ------------------------------------------------------------------------------------
batch_size = 64
learning_rate = 0.01
momentum = 0.5
EPOCH = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Prepare dataset ------------------------------------------------------------------------------------
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# softmax归一化指数函数(https://blog.csdn.net/lz_peter/article/details/84574716),其中0.1307是mean均值和0.3081是std标准差

train_dataset = datasets.MNIST(root='./data/mnist', train=True, transform=transform)  # 本地没有就加上download=True
test_dataset = datasets.MNIST(root='./data/mnist', train=False, transform=transform)  # train=True训练集，=False测试集
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

fig = plt.figure()
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.tight_layout()
    plt.imshow(train_dataset.train_data[i], cmap='gray', interpolation='none')
    plt.title("Labels: {}".format(train_dataset.train_labels[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()


# 训练集乱序，测试集有序
# Design model using class ------------------------------------------------------------------------------
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)  # 一层卷积层,一层池化层,一层激活层(图是先卷积后激活再池化，差别不大)
        x = self.conv2(x)  # 再来一次
        x = x.view(batch_size, -1)  # flatten 变成全连接网络需要的输入 (batch, 20,4,4) ==> (batch,320), -1 此处自动算出的是320
        x = self.fc(x)
        return x  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）


model = Net()


# Construct loss and optimizer ------------------------------------------------------------------------------
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)  # lr学习率，momentum冲量


# Train and Test CLASS --------------------------------------------------------------------------------------
# 把单独的一轮一环封装在函数类里
def train(epoch):
    running_loss = 0.0  # 这整个epoch的loss清零
    running_total = 0
    running_correct = 0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        # inputs, target = inputs.to(device), target.to(device)  # 转到GPU上
        optimizer.zero_grad()

        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()

        # 把运行中的loss累加起来，为了下面300次一除
        running_loss += loss.item()
        # 把运行中的准确率acc算出来
        _, predicted = torch.max(outputs.data, dim=1)
        running_total += inputs.shape[0]
        running_correct += (predicted == target).sum().item()

        if batch_idx % 300 == 299:  # 不想要每一次都出loss，浪费时间，选择每300次出一个平均损失,和准确率
            print('[%d, %5d]: loss: %.3f , acc: %.2f %%'
                  % (epoch + 1, batch_idx + 1, running_loss / 300, 100 * running_correct / running_total))
            running_loss = 0.0  # 这小批300的loss清零
            running_total = 0
            running_correct = 0  # 这小批300的acc清零

        # torch.save(model.state_dict(), './model_Mnist.pth')
        # torch.save(optimizer.state_dict(), './optimizer_Mnist.pth')


def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 测试集不用算梯度
        for data in test_loader:
            images, labels = data
            # images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # dim = 1 列是第0个维度，行是第1个维度，沿着行(第1个维度)去找1.最大值和2.最大值的下标
            total += labels.size(0)  # 张量之间的比较运算
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print('[%d / %d]: Accuracy on test set: %.1f %% ' % (epoch+1, EPOCH, 100 * acc))  # 求测试的准确率，正确数/总数
    return acc


# Start train and Test --------------------------------------------------------------------------------------
if __name__ == '__main__':
    acc_list_test = []
    for epoch in range(EPOCH):
        train(epoch)
        # if epoch % 10 == 9:  #每训练10轮 测试1次
        acc_test = test()
        acc_list_test.append(acc_test)

    plt.plot(acc_list_test)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy On TestSet')
    plt.show()
'''
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyper parameters
num_epochs = 10
num_classes = 10
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)  # Flatten the output
        out = self.fc(out)
        return out

model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Function to visualize images
def show_images(images):
    plt.figure(figsize=(10,10))
    for i, img in enumerate(images):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img.squeeze().numpy(), cmap='gray')
    plt.show()

# Function to train the model
def train(model, device, train_loader, optimizer, epoch, num_epochs):
    model.train()  # Set the model to training mode
    total_step = len(train_loader)
    losses = []
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    return losses

# Function to test the model
def test(model, device, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Function to plot training progress
def plot_progress(epochs, train_losses, test_accuracies):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for i in range(epochs):
        plt.plot(train_losses[i], label=f'Epoch {i+1}')
    plt.title('Training Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()

# Visualize a few images
dataiter = iter(train_loader)
images, _ = next(dataiter)
show_images(images[:25])  # Visualize 25 images

# Initialize lists to monitor loss and accuracy
train_losses = []
test_accuracies = []

# Train the model
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    train_loss = train(model, device, train_loader, optimizer, epoch, num_epochs)
    train_losses.append(train_loss)
    test_accuracy = test(model, device, test_loader)
    test_accuracies.append(test_accuracy)

    print(f'Epoch {epoch+1}/{num_epochs} - Average Loss: {sum(train_loss)/len(train_loss):.4f}, Accuracy: {test_accuracy:.2f}%')

# Plot training progress
plot_progress(num_epochs, train_losses, test_accuracies)

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
'''
# import torch
# import torchvision
# from torchvision import datasets
# from torchvision import transforms
# from torch.utils.data import DataLoader
 
# batch_size = 64
# device = torch.device("cuda:0")
 
# class CNN(torch.nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.convl1 = torch.nn.Conv2d(1,10,5)
#         self.convl2 = torch.nn.Conv2d(10,20,5)
#         self.pooling = torch.nn.MaxPool2d(2)
#         self.activate = torch.nn.ReLU()
#         self.linear = torch.nn.Linear(320,10)
 
#     def forward(self,x):
#         x = x.view(-1,1,28,28)
#         x = self.convl1(x)
#         x = self.pooling(x)
#         x = self.activate(x)
#         x = self.convl2(x)
#         x = self.pooling(x)
#         x = x.view(-1,320)
#         x = self.linear(x)
#         return x
 
# def train(train_loader, model, criterion, optimizer, epoch):
#     loss_sum = 0.0
#     for index, (x, y) in enumerate(train_loader):
#         x = x.to(device)
#         y = y.to(device)
#         y_hat = model(x)
#         loss = criterion(y_hat, y)
#         loss_sum += loss.item()
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
 
#         if (index % 300 == 299):
#             print(epoch,",",index//300, ":", loss_sum/300)
#             loss_sum = 0.0
 
 
# def test(test_loader, model):
#     total = 0
#     correct = 0
#     for x,y in test_loader:
#         x = x.to(device)
#         y = y.to(device)
#         y_hat = model(x)
#         _,guess = torch.max(y_hat,1)
#         correct += (guess == y).sum().item()
#         total += y.size(0)
#     print("ACC == ", correct / total)
 
 
# if __name__ == '__main__':
#     transformer=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(0.1307, 0.3018)])
#     train_data = datasets.MNIST('MNIST',True,transformer,download=True)
#     test_data=datasets.MNIST('MNIST',True,transformer,download=True)
 
#     train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True, num_workers=2)
#     test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=True,num_workers=2)
#     model = CNN()
#     model.to(device)
#     criterion = torch.nn.CrossEntropyLoss()
#     criterion.to(device)
#     optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)
#     for epoch in range(10):
#         train(train_loader,model,criterion,optimizer,epoch)
 
# test(test_loader,model)