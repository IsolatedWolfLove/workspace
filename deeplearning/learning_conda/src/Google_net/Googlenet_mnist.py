''' coding:utf-8 '''
"""
作者:shiyi
日期:年 09月 09日
通过pytorch模块实现Googlenet
"""


import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from rich.progress import track

# 参数设置
batch_size = 64


# 将数据类型转化为tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.137, ), (0.3081, ))
])
# 构建训练集数据
train_dataset = datasets.MNIST(root='../dataset/minist/',
                               train=True,
                               download=True,
                               transform=transform)
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)
# 构建测试集数据
test_dataset = datasets.MNIST(root='../dataset/minist',
                              train=False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)


class InceptionA(torch.nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        # 构建第一卷积层分支
        self.branch1x1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        # 构建第二卷积层分支
        self.branch5x5_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding=2)
        # 构造第三卷积层分支
        self.branch3x3_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1)
        # 构造池化层分支
        self.branch_pool = torch.nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        # debug
        # print("brach1x1 shape ", branch1x1.shape,
        #       "branch5x5 shape ", branch5x5.shape,
        #       "branch3x3 shape ", branch3x3.shape,
        #       "branch_pool shape ", branch_pool.shape)
        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)        # 将返回值进行拼接，注意，dim=1意味着将数据沿channel方向进行拼接，要保证b，w，h的参数都相同


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=5)

        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)

        self.mp = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(1408, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))      # 按照计算图构造前向函数结构1
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))      # 按照计算图构造前向函数结构2
        x = self.incep2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x


model = Net()       # 实例化

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# 将训练移植到GUP上进行并行运算加速
device = torch.device("cuda:0" if torch.cuda.is_available() else "CPU")     # 定义显卡设备0
model.to(device)    # 将模型迁移搭配device0 及显卡0上


# 定义训练函数
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):       # 将索引从1开始
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)       # 将训练过程的数据也迁移至GPU上
        optimizer.zero_grad()

        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch+1, batch_idx+1, running_loss / 2000))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)  # 将测试过程的数据也迁移至GPU上
            # print("input shape", input.shape)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print('Accuracy on test set: %.3f %% [%.3f/%.3f]' % (100 * correct/total, correct, total))


if __name__ == '__main__':
    for epoch in track(range(28)):
        train(epoch)
        test()
    torch.onnx.export(model,
                      torch.randn(1, 1, 28, 28).to(device),
                      "googlenet_300bitch.onnx",
                      verbose=True)