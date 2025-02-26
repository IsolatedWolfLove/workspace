import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.progress import track
console = Console()

datapath = "/home/ccy/workspace/deeplearning/learning_conda/src/flower_det/data/flowers"
txtpath = "/home/ccy/workspace/deeplearning/learning_conda/src/flower_det/data/label.txt"

# 数据预处理
transform_train = transforms.Compose([
    transforms.Resize((384, 384)),  # 统一大小
    transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
    transforms.RandomRotation(10),  # 小角度旋转
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色调整
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 数据集类
class FlowerDataset(Dataset):
    def __init__(self, txtpath, datapath, transform=None, train=True):
        self.transform = transform
        self.train = train
        self.data = []
        self.datapath = datapath
        
        with open(txtpath, 'r') as f:
            for line in f:
                img_path, label = line.strip().split()
                if label == 'daisy':
                    label = 0
                elif label == 'dandelion':
                    label = 1
                elif label == 'rose':
                    label = 2
                elif label == 'sunflower':
                    label = 3
                elif label == 'tulip':
                    label = 4
                self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(self.datapath + '/' + img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Mixup 数据增强
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# 模型类
class FlowerClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(FlowerClassifier, self).__init__()
        # 使用 EfficientNet B0 替代 EfficientNetV2
        self.model = timm.create_model('efficientnet_b0', pretrained=True)
        
        # 冻结部分层
        for param in list(self.model.parameters())[:-30]:  # 只训练最后几层
            param.requires_grad = False
            
        # 修改最后的分类层
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# 训练设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 20
batch_size = 64
learning_rate = 0.001
    
# 数据加载
train_dataset = FlowerDataset(txtpath, datapath=datapath, transform=transform_train, train=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=18)

# 模型初始化
model = FlowerClassifier(num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# 在训练循环开始前添加两个列表来存储损失值和准确率
epoch_losses = []
epoch_accs = []

# 训练循环
for epoch in track(range(num_epochs), description="Training"):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Mixup
        images, targets_a, targets_b, lam = mixup_data(images, labels)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # 计算准确率
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (lam * predicted.eq(targets_a).sum().float()
                    + (1 - lam) * predicted.eq(targets_b).sum().float())
        
        if (i + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
    scheduler.step()
    
    # 每个 epoch 结束后打印统计信息
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')

    # 在每个 epoch 结束时添加损失值
    epoch_losses.append(epoch_loss)
    epoch_accs.append(epoch_acc)

# 损失曲线
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), epoch_losses, 'b-', label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 准确率曲线
# plt.figure(figsize=(10, 5))
# plt.plot(range(1, num_epochs + 1), epoch_accs, 'r-', label='Training Accuracy')
# plt.title('Training Accuracy Over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy (%)')
# plt.legend()
# plt.grid(True)
# plt.show()

# 保存模型
torch.save(model.state_dict(), 'flower_classifier_1.pth')
