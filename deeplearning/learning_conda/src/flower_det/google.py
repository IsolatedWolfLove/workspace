import torch.nn as nn
import torch.optim as optim 
import torch
from torchvision import datasets,transforms    #数据集    transforms完成对数据的处理
import matplotlib.pyplot as plt
from rich import print
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from rich.progress import track
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
batch_size = 100


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
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=5)

        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)

        self.mp = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(1408, 5)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))      # 按照计算图构造前向函数结构1
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))      # 按照计算图构造前向函数结构2
        x = self.incep2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x
epoch_losses = []
num_classes = 5 #输出大小（类别数）
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
    # transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.25, 1.3333333333333333)),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((28, 28)), 
    transforms.ToTensor(),  # 将图片转换为Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
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
data_loader = DataLoader(data, batch_size,shuffle=True, num_workers=15)
model = Net()       # 实例化

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.5)


# 将训练移植到GUP上进行并行运算加速
device = torch.device("cuda:0" if torch.cuda.is_available() else "CPU")     # 定义显卡设备0
model.to(device)    # 将模型迁移搭配device0 及显卡0上
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(data_loader, 0):       # 将索引从1开始
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)       # 将训练过程的数据也迁移至GPU上
        optimizer.zero_grad()

        # forward + backward + update                 
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 10 == 9:
            print('[%d, %5d] loss: %.3f' % (epoch+1, batch_idx+1, running_loss / 2000))
            running_loss = 0.0
        return running_loss


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)  # 将测试过程的数据也迁移至GPU上
            # print("input shape", input.shape)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print('Accuracy on test set: %.3f %% [%.3f/%.3f]' % (100 * correct/total, correct, total))


if __name__ == '__main__':
    for epoch in track(range(num_epochs)):
        train(epoch)
        test()
    torch.onnx.export(model,torch.randn(1, 3, 28, 28).to(device),"googlenet_100_50picibitch.onnx", verbose=True)