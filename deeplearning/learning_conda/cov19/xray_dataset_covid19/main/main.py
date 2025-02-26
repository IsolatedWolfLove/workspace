'''
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn import svm
from rich import print
from rich.progress import track
num_epochs = 20
# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 加载数据集
train_dataset = datasets.ImageFolder(root='path_to_train_data', transform=transform)
test_dataset = datasets.ImageFolder(root='path_to_test_data', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 构建CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)  # 假设输入图像大小为28x28

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # 展平特征图
        x = torch.relu(self.fc1(x))
        return x

# 实例化模型
model = CNN()

# 训练CNN模型
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in track(range(num_epochs)):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 特征提取
features = []
labels = []
with torch.no_grad():
    for images, labels_batch in test_loader:
        outputs = model(images)
        features.extend(outputs.numpy())
        labels.extend(labels_batch.numpy())

# SVM分类器训练
class LinearSVM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearSVM, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
# 实例化SVM分类器

clf = svm.SVC(kernel='linear')
clf.fit(features, labels)

# 模型评估
# 这里可以使用测试集上的特征和标签来评估SVM分类器的性能
'''





# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from rich import print
# from rich.progress import track
# # 数据预处理
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # 调整图片大小以适应CNN输入
#     transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485], std=[0.229])  # 灰度图的均值和标准差
# ])

# # 加载训练数据集
# train_dataset = datasets.ImageFolder(root='/home/ccy/workspace/deeplearning/learning_conda/cov19/archive(2)/chest_xray/train', transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# # 加载测试数据集
# test_dataset = datasets.ImageFolder(root='/home/ccy/workspace/deeplearning/learning_conda/cov19/archive(2)/chest_xray/test', transform=transform)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# # 定义CNN模型
# class CNNFeatureExtractor(nn.Module):
#     def __init__(self):
#         super(CNNFeatureExtractor, self).__init__()
#         self.features = nn.Sequential(
#             # 第一个卷积层，输入通道为1（灰度图像），输出通道为32
#             nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 池化层，下采样

#             # 第二个卷积层，输入通道为32，输出通道为64
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 池化层，下采样

#             # 第三个卷积层，输入通道为64，输出通道为128
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 池化层，下采样

#             # 展平层，将多维的特征图展平成一维向量
#             nn.Flatten()
#         )
        
#         # 全连接层，将展平后的特征向量映射到128维特征空间
#         self.classifier = nn.Linear(128 * 28 * 28, 128)  # 修改这里的尺寸

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x

# # 实例化CNN模型并转移到GPU
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# cnn = CNNFeatureExtractor().to(device)

# # 训练CNN模型
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(cnn.parameters(), lr=0.001)

# # 训练CNN
# num_epochs = 10  # 为了示例，我们只训练10个epoch
# for epoch in track(range(num_epochs)):
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)  # 将数据转移到GPU
#         optimizer.zero_grad()
#         outputs = cnn(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
# torch.save(cnn.state_dict(), 'conv_10_epoch_cnn_model.pth')  # 保存模型参数
# # 特征提取
# features = []
# labels = []
# cnn.eval()  # 设置为评估模式
# with torch.no_grad():
#     for images, labels_batch in train_loader:  
#         images = images.to(device)
#         outputs = cnn(images)
#         features.append(outputs.to(device))  # 保持为Tensor
#         labels.append(labels_batch.to(device))

# features = torch.cat(features, dim=0)
# labels = torch.cat(labels, dim=0)

# # 定义SVM模型
# class LinearSVM(nn.Module):
#     def __init__(self, input_dim, num_classes):
#         super(LinearSVM, self).__init__()
#         self.fc = nn.Linear(input_dim, num_classes)

#     def forward(self, x):
#         return self.fc(x)

# # 实例化SVM模型并转移到GPU
# svm = LinearSVM(input_dim=128, num_classes=len(torch.unique(labels))).to(device)

# # 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(svm.parameters(), lr=0.01)

# # 训练SVM模型
# num_epochs_svm = 8888
# for epoch in track(range(num_epochs_svm)):
#     optimizer.zero_grad()
#     features_tensor = features.to(device)  # 将特征合并并转移到GPU
#     labels_tensor = labels.to(device)  # 将标签合并并转移到GPU
#     outputs = svm(features_tensor)
#     loss = criterion(outputs, labels_tensor)
#     loss.backward()
#     optimizer.step()

# # 评估SVM模型
# svm.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for images, labels_batch in test_loader:
#         images = images.to(device)
#         labels_batch = labels_batch.to(device)  # 将标签也转移到GPU上
#         features_test = cnn(images)
#         outputs = svm(features_test)
#         _, predicted = torch.max(outputs, 1)
#         total += labels_batch.size(0)
#         correct += (predicted == labels_batch).sum().item()

# accuracy = 100 * correct / total
# print(f'Accuracy of the SVM model on the test set: {accuracy}%')



# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms, models
# from torch.utils.data import DataLoader
# from rich import print
# from rich.progress import track

# # 数据预处理
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # 调整图片大小以适应CNN输入
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet的均值和标准差
# ])

# # 加载训练数据集
# train_dataset = datasets.ImageFolder(root='/home/ccy/workspace/deeplearning/learning_conda/cov19/archive(2)/chest_xray/train', transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# # 加载测试数据集
# test_dataset = datasets.ImageFolder(root='/home/ccy/workspace/deeplearning/learning_conda/cov19/archive(2)/chest_xray/test', transform=transform)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# # 定义GoogLeNet模型
# def get_googlenet(num_classes):
#     model = models.googlenet(pretrained=True)
#     # 冻结所有参数
#     for param in model.parameters():
#         param.requires_grad = False
#     # 修改最后的全连接层
#     num_features = model.fc.in_features
#     model.fc = nn.Linear(num_features, num_classes)
#     return model

# # 获取类别数量
# targets_tensor = torch.tensor(train_dataset.targets)
# num_classes = len(torch.unique(targets_tensor))

# # 实例化GoogLeNet模型并转移到GPU
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# googlenet = get_googlenet(num_classes).to(device)

# # 训练GoogLeNet模型
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(googlenet.fc.parameters(), lr=0.001)  # 只训练最后的全连接层

# # 训练GoogLeNet
# num_epochs = 10  # 为了示例，我们只训练10个epoch
# for epoch in track(range(num_epochs)):
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)  # 将数据转移到GPU
#         optimizer.zero_grad()
#         outputs = googlenet(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

# torch.save(googlenet.state_dict(), 'googlenet_model.pth')  # 保存模型参数

# # 特征提取
# features = []
# labels = []
# googlenet.eval()  # 设置为评估模式
# with torch.no_grad():
#     for images, labels_batch in train_loader:  
#         images = images.to(device)
#         outputs = googlenet(images)
#         features.append(outputs.cpu())  # 保持为Tensor并移回CPU
#         labels.append(labels_batch.cpu())

# features = torch.cat(features, dim=0)
# labels = torch.cat(labels, dim=0)

# # 定义SVM模型
# class LinearSVM(nn.Module):
#     def __init__(self, input_dim, num_classes):
#         super(LinearSVM, self).__init__()
#         self.fc = nn.Linear(input_dim, num_classes)

#     def forward(self, x):
#         return self.fc(x)

# # 实例化SVM模型并转移到GPU
# svm = LinearSVM(input_dim=1024, num_classes=num_classes).to(device)  # 确保input_dim与GoogleNet输出匹配

# # 定义损失函数和优化器
# criterion_svm = nn.CrossEntropyLoss()
# optimizer_svm = optim.SGD(svm.parameters(), lr=0.01)

# # 训练SVM模型
# num_epochs_svm = 8888
# for epoch in track(range(num_epochs_svm)):
#     optimizer_svm.zero_grad()
#     features_tensor = features.to(device)  # 将特征合并并转移到GPU
#     labels_tensor = labels.to(device)  # 将标签合并并转移到GPU
#     outputs = svm(features_tensor)
#     loss = criterion_svm(outputs, labels_tensor)
#     loss.backward()
#     optimizer_svm.step()

# # 评估SVM模型
# svm.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for images, labels_batch in test_loader:
#         images = images.to(device)
#         labels_batch = labels_batch.to(device)  # 将标签也转移到GPU上
#         features_test = googlenet(images)
#         outputs = svm(features_test)
#         _, predicted = torch.max(outputs, 1)
#         total += labels_batch.size(0)
#         correct += (predicted == labels_batch).sum().item()

# accuracy = 100 * correct / total
# print(f'Accuracy of the SVM model on the test set: {accuracy}%')



# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms, models
# from torch.utils.data import DataLoader
# from rich import print
# from rich.progress import track

# # 数据预处理
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # 调整图片大小以适应CNN输入
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet的均值和标准差
# ])

# # 加载训练数据集
# train_dataset = datasets.ImageFolder(root='/home/ccy/workspace/deeplearning/learning_conda/cov19/archive(2)/chest_xray/train', transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# # 加载测试数据集
# test_dataset = datasets.ImageFolder(root='/home/ccy/workspace/deeplearning/learning_conda/cov19/archive(2)/chest_xray/test', transform=transform)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# # 定义GoogLeNet模型
# def get_googlenet(num_classes):
#     model = models.googlenet(pretrained=True)
#     # 冻结所有参数
#     for param in model.parameters():
#         param.requires_grad = False
#     # 修改最后的全连接层
#     num_features = model.fc.in_features
#     model.fc = nn.Linear(num_features, num_classes)
#     return model

# # 获取类别数量
# targets_tensor = torch.tensor(train_dataset.targets)
# num_classes = len(torch.unique(targets_tensor))

# # 实例化GoogLeNet模型并转移到GPU
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# googlenet = get_googlenet(num_classes).to(device)

# # 训练GoogLeNet模型
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(googlenet.fc.parameters(), lr=0.001)  # 只训练最后的全连接层

# # 训练GoogLeNet
# num_epochs = 50  # 为了示例，我们只训练10个epoch
# for epoch in track(range(num_epochs)):
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)  # 将数据转移到GPU
#         optimizer.zero_grad()
#         outputs = googlenet(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

# torch.save(googlenet.state_dict(), 'googlenet_50_model.pth')  # 保存模型参数

# # 特征提取
# features = []
# labels = []
# googlenet.eval()  # 设置为评估模式
# with torch.no_grad():
#     for images, labels_batch in train_loader:  
#         images = images.to(device)
#         outputs = googlenet(images)
#         # 获取卷积层的输出
#         features.append(outputs.cpu())  # 保持为Tensor并移回CPU
#         labels.append(labels_batch.cpu())

# features = torch.cat(features, dim=0)
# labels = torch.cat(labels, dim=0)

# print(features.shape)  # 打印特征的维度

# # 定义SVM模型
# class LinearSVM(nn.Module):
#     def __init__(self, input_dim, num_classes):
#         super(LinearSVM, self).__init__()
#         self.fc = nn.Linear(input_dim, num_classes)

#     def forward(self, x):
#         return self.fc(x)

# # 实例化SVM模型并转移到GPU
# input_dim = features.shape[1]  # 动态获取特征维度
# svm = LinearSVM(input_dim=input_dim, num_classes=num_classes).to(device)  # 确保input_dim与GoogleNet输出匹配

# # 定义损失函数和优化器
# criterion_svm = nn.CrossEntropyLoss()
# optimizer_svm = optim.SGD(svm.parameters(), lr=0.01)

# # 训练SVM模型
# num_epochs_svm = 8888
# for epoch in track(range(num_epochs_svm)):
#     optimizer_svm.zero_grad()
#     features_tensor = features.to(device)  # 将特征合并并转移到GPU
#     labels_tensor = labels.to(device)  # 将标签合并并转移到GPU
#     outputs = svm(features_tensor)
#     loss = criterion_svm(outputs, labels_tensor)
#     loss.backward()
#     optimizer_svm.step()

# # 评估SVM模型
# svm.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for images, labels_batch in test_loader:
#         images = images.to(device)
#         labels_batch = labels_batch.to(device)  # 将标签也转移到GPU上
#         features_test = googlenet(images)
#         outputs = svm(features_test)
#         _, predicted = torch.max(outputs, 1)
#         total += labels_batch.size(0)
#         correct += (predicted == labels_batch).sum().item()

# accuracy = 100 * correct / total
# print(f'Accuracy of the SVM model on the test set: {accuracy}%')


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms, models
# from torch.utils.data import DataLoader
# from rich import print
# from rich.progress import track

# # 数据预处理
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # 调整图片大小以适应CNN输入
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet的均值和标准差
# ])

# # 加载训练数据集
# train_dataset = datasets.ImageFolder(root='/home/ccy/workspace/deeplearning/learning_conda/cov19/archive(2)/chest_xray/train', transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# # 加载测试数据集
# test_dataset = datasets.ImageFolder(root='/home/ccy/workspace/deeplearning/learning_conda/cov19/archive(2)/chest_xray/test', transform=transform)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# # 定义GoogLeNet模型
# def get_googlenet(num_classes):
#     model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)  # 使用weights参数
#     # 冻结所有参数
#     for param in model.parameters():
#         param.requires_grad = False
#     # 修改最后的全连接层
#     num_features = model.fc.in_features
#     model.fc = nn.Linear(num_features, num_classes)
#     return model

# # 获取类别数量
# targets_tensor = torch.tensor(train_dataset.targets)
# num_classes = len(torch.unique(targets_tensor))

# # 实例化GoogLeNet模型并转移到GPU
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# googlenet = get_googlenet(num_classes).to(device)

# # 加载模型参数
# model_path = 'googlenet_model.pth'
# googlenet.load_state_dict(torch.load(model_path, map_location=device))

# # 特征提取
# features = []
# labels = []
# googlenet.eval()  # 设置为评估模式
# with torch.no_grad():
#     for images, labels_batch in train_loader:  
#         images = images.to(device)
#         outputs = googlenet(images)
#         features.append(outputs.cpu())  # 保持为Tensor并移回CPU
#         labels.append(labels_batch.cpu())

# features = torch.cat(features, dim=0)
# labels = torch.cat(labels, dim=0)

# # 定义SVM模型
# class LinearSVM(nn.Module):
#     def __init__(self, input_dim, num_classes):
#         super(LinearSVM, self).__init__()
#         self.fc = nn.Linear(input_dim, num_classes)

#     def forward(self, x):
#         return self.fc(x)

# # 实例化SVM模型并转移到GPU
# input_dim = features.shape[1]  # 动态获取特征维度
# svm = LinearSVM(input_dim=input_dim, num_classes=num_classes).to(device)  # 确保input_dim与GoogleNet输出匹配

# # 定义损失函数和优化器
# criterion_svm = nn.CrossEntropyLoss()
# optimizer_svm = optim.SGD(svm.parameters(), lr=0.01)

# # 训练SVM模型
# num_epochs_svm = 88
# for epoch in track(range(num_epochs_svm)):
#     optimizer_svm.zero_grad()
#     features_tensor = features.to(device)  # 将特征合并并转移到GPU
#     labels_tensor = labels.to(device)  # 将标签合并并转移到GPU
#     outputs = svm(features_tensor)
#     loss = criterion_svm(outputs, labels_tensor)
#     loss.backward()
#     optimizer_svm.step()

# # 评估SVM模型
# svm.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for images, labels_batch in test_loader:
#         images = images.to(device)
#         labels_batch = labels_batch.to(device)  # 将标签也转移到GPU上
#         features_test = googlenet(images)
#         outputs = svm(features_test)
#         _, predicted = torch.max(outputs, 1)
#         total += labels_batch.size(0)
#         correct += (predicted == labels_batch).sum().item()

# accuracy = 100 * correct / total
# print(f'Accuracy of the SVM model on the test set: {accuracy}%')




import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from rich import print
from rich.progress import track

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图片大小以适应CNN输入
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet的均值和标准差
])

# 加载训练数据集
train_dataset = datasets.ImageFolder(root='/home/ccy/workspace/deeplearning/learning_conda/cov19/archive(2)/chest_xray/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 加载测试数据集
test_dataset = datasets.ImageFolder(root='/home/ccy/workspace/deeplearning/learning_conda/cov19/archive(2)/chest_xray/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# 定义ResNet模型
def get_resnet(num_classes):
    model = models.resnet50(pretrained=True)
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    # 修改最后的全连接层
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

# 获取类别数量
targets_tensor = torch.tensor(train_dataset.targets)
num_classes = len(torch.unique(targets_tensor))

# 实例化ResNet模型并转移到GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet = get_resnet(num_classes).to(device)

# 训练ResNet模型
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.fc.parameters(), lr=0.001)  # 只训练最后的全连接层

# 训练ResNet
num_epochs = 50  # 为了示例，我们只训练10个epoch
for epoch in track(range(num_epochs)):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # 将数据转移到GPU
        optimizer.zero_grad()
        outputs = resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

torch.save(resnet.state_dict(), 'all_resnet50__model.pth')  # 保存模型参数

# 特征提取
features = []
labels = []
resnet.eval()  # 设置为评估模式
with torch.no_grad():
    for images, labels_batch in train_loader:  
        images = images.to(device)
        outputs = resnet(images)
        # 获取全连接层之前的输出
        features.append(outputs.cpu())  # 保持为Tensor并移回CPU
        labels.append(labels_batch.cpu())

features = torch.cat(features, dim=0)
labels = torch.cat(labels, dim=0)

print(features.shape)  # 打印特征的维度

# 定义SVM模型
class LinearSVM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearSVM, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# 实例化SVM模型并转移到GPU
input_dim = features.shape[1]  # 动态获取特征维度
svm = LinearSVM(input_dim=input_dim, num_classes=num_classes).to(device)  # 确保input_dim与ResNet输出匹配

# 定义损失函数和优化器
criterion_svm = nn.CrossEntropyLoss()
optimizer_svm = optim.SGD(svm.parameters(), lr=0.01)

# 训练SVM模型
num_epochs_svm = 8888
for epoch in track(range(num_epochs_svm)):
    optimizer_svm.zero_grad()
    features_tensor = features.to(device)  # 将特征合并并转移到GPU
    labels_tensor = labels.to(device)  # 将标签合并并转移到GPU
    outputs = svm(features_tensor)
    loss = criterion_svm(outputs, labels_tensor)
    loss.backward()
    optimizer_svm.step()

# 评估SVM模型
svm.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels_batch in test_loader:
        images = images.to(device)
        labels_batch = labels_batch.to(device)  # 将标签也转移到GPU上
        features_test = resnet(images)
        outputs = svm(features_test)
        _, predicted = torch.max(outputs, 1)
        total += labels_batch.size(0)
        correct += (predicted == labels_batch).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the SVM model on the test set: {accuracy}%')