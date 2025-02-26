import torch
import torch.nn as nn
import torch.optim as optim                    #定义优化器
from torchvision import datasets,transforms    #数据集    transforms完成对数据的处理
import matplotlib.pyplot as plt
from rich import print
from rich.progress import track
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
num_classes = 5 #输出大小（类别数）
batch_size = 28#批大小
learning_rate = 0.0005 #学习率
num_epochs = 50#训练轮数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
datapath = "/home/ccy/workspace/deeplearning/learning_conda/src/flower_det/data/flowers"
txtpath = "/home/ccy/workspace/deeplearning/learning_conda/src/flower_det/data/label.txt"
class MyDataset(Dataset):
    def __init__(self,txtpath, transform=None):
        #创建一个list用来储存图片和标签信息
        imgs = []
        self.transforms = transform
        #打开第一步创建的txt文件，按行读取，将结果以元组方式保存在imgs里
        datainfo = open(txtpath,'r')
        for line in datainfo:
            # print(line)
            line = line.strip('\n')
            words = line.split()
            # print(words)
            if(words[1] == 'daisy'):
                words[1] = 0
            elif(words[1] == 'dandelion'):
                words[1] = 1
            elif(words[1] == 'rose'):
                words[1] = 2
            elif(words[1] == 'sunflower'):
                words[1] = 3
            elif(words[1] == 'tulip'):
                words[1] = 4    
            # print(words)
            imgs.append((words[0],words[1]))

        self.imgs = imgs
	#返回数据集大小
    def __len__(self):
        return len(self.imgs)
	#打开index对应图片进行预处理后return回处理后的图片和标签
    def __getitem__(self, index):
        pic,label = self.imgs[index]
        pic = Image.open(datapath+'/'+pic)
        if self.transforms is not None:
            pic = self.transforms(pic)
        return pic,int(label)
#实例化对象
transform = transforms.Compose([
    transforms.RandomVerticalFlip(),  # 随机垂直翻转
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((512, 512)),  # 改为 512x512
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 创建数据集实例
# data = SVHNDataset(train_path, train_label,
# 			transforms.Compose([
# 				# 缩放到固定尺寸
# 	 			transforms.Resize((64, 128)),
			
# 			  	# 随机颜色变换
# 	 		  	transforms.ColorJitter(0.2, 0.2, 0.2),
	
# 			  	# 加⼊随机旋转
# 	 	     	transforms.RandomRotation(5),
	 	   
# 	 	     	# 将图⽚转换为pytorch 的tesntor
# 		     	# transforms.ToTensor(),
		   
# 		     	# 对图像像素进⾏归⼀化
# 		     	# transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
# 	    	 ]))
data = MyDataset(txtpath, transform=transform)

# 将数据集导入DataLoader
data_loader = DataLoader(data, batch_size,shuffle=True, num_workers=12)
# for i, (images, labels) in enumerate(data_loader):
#     print(images.shape)
#     for user in labels:
# 	    print(user)


class LeNet(nn.Module):
    def __init__(self, num_classes=5):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        
        # 计算 fc1 的正确输入大小
        # 假设输入图像大小为 512x512
        # Conv1: 512x512 -> 512x512 (padding=2)
        # Pool1: 512x512 -> 256x256
        # Conv2: 256x256 -> 252x252 (no padding)
        # Pool2: 252x252 -> 126x126
        # 因此最终特征图大小为 126x126x16
        self.fc1 = nn.Linear(in_features=16 * 126 * 126, out_features=120)  # 调整 in_features
        self.fc1_0 = nn.Linear(in_features=120, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc2_0 = nn.Linear(in_features=84, out_features=84)
        self.fc2_1 = nn.Linear(in_features=84, out_features=42)
        self.fc2_2 = nn.Linear(in_features=42, out_features=18)
        self.fc3 = nn.Linear(in_features=18, out_features=num_classes)
        self.act = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.pool1(self.act(self.conv1(x)))
        y = self.pool2(self.act(self.conv2(y)))
        y = self.flatten(y)
        y = self.act(self.fc1(y))
        y = self.act(self.fc1_0(y))
        y = self.act(self.fc2(y))
        y = self.act(self.fc2_0(y))
        y = self.act(self.fc2_1(y))
        y = self.act(self.fc2_2(y))
        y = self.fc3(y)
        y = self.softmax(y)
        return y


# 创建模型实例
model = LeNet(num_classes=5)

# 将模型放到设备上 (CPU 或 GPU)

model.to(device)

# 使用 torchsummary 显示模型摘要
# summary(model, input_size=(2, 28, 28))
criterion = nn.CrossEntropyLoss()
# CrossEntropyLoss = Softmax + log + nllloss
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
# optimizer = optim.SGD(model.parameters(),0.2)
epoch_losses = []
# 定义累积步数
accumulation_steps = 4  # 累积4个batch的梯度再更新
optimizer.zero_grad()   # 确保开始时梯度为零

for epoch in track(range(num_epochs), description="Training..."):
    total_loss = 0
    for i, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 将损失除以累积步数
        loss = loss / accumulation_steps
        
        # 反向传播
        loss.backward()
        
        # 每 accumulation_steps 次迭代，更新一次参数
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps  # 恢复原始损失大小用于打印
                         
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step[{i+1}/{len(data_loader)}], Loss: {loss.item()*accumulation_steps:.4f}')
    
    # 处理最后不足 accumulation_steps 的批次
    if (i + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    epoch_loss = total_loss / len(data_loader)
    epoch_losses.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}] Average Loss: {epoch_loss:.4f}')
# 预测
model.eval()  # 切换到评估模式
with torch.no_grad():
    for i, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        # print(f'Predicted: {predicted}, Actual: {labels}')
        print(f'Accuracy of the network on the {i+1}th batch: {100 * (predicted == labels).sum().item() / labels.size(0):.2f}%')

# 在模型评估之后，导出模型之前
model.eval()  # 确保模型在评估模式

# 创建一个示例输入
dummy_input = torch.randn(1, 3, 512, 512).to(device)  # 更新为 512x512

# 导出模型
try:
    torch.onnx.export(
        model,                  # 要导出的模型
        dummy_input,           # 模型的示例输入
        "flower_model_ok.onnx",   # 输出文件名
        export_params=True,    # 存储训练好的参数权重
        opset_version=11,      # ONNX版本
        do_constant_folding=True,  # 是否执行常量折叠优化
        input_names=['input'],     # 输入名
        output_names=['output'],   # 输出名
        dynamic_axes={             # 动态尺寸
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print("模型导出成功！保存为 'flower_model_ok.onnx'")
except Exception as e:
    print(f"模型导出失败，错误信息：{str(e)}")

# 然后再画损失曲线
plt.figure(figsize=(20, 10))
plt.plot(range(1, num_epochs + 1), epoch_losses)
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
