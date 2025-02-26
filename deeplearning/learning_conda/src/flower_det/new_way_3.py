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
from IPython.display import clear_output

console = Console()

# 使用您原有的数据路径
datapath = "/home/ccy/workspace/deeplearning/learning_conda/src/flower_det/data/flowers"
txtpath = "/home/ccy/workspace/deeplearning/learning_conda/src/flower_det/data/label.txt"

# 数据预处理
transform_train = transforms.Compose([
    transforms.Resize((384, 384)),  # 保持较大的输入尺寸
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAutocontrast(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 保持您原有的数据集类
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

# 新的高级分类器
class AdvancedFlowerClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # 使用 ConvNeXt Large 作为骨干网络
        self.model = timm.create_model('convnext_large_in22k', pretrained=True)
        
        # 获取特征维度
        num_ftrs = self.model.head.fc.in_features
        
        # 修改分类头
        self.model.head.fc = nn.Sequential(
            nn.LayerNorm(num_ftrs),
            nn.Linear(num_ftrs, 1024),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # 冻结部分层
        self._freeze_layers()
        
    def _freeze_layers(self):
        # 冻结前面的层
        for param in list(self.model.parameters())[:-30]:
            param.requires_grad = False
            
    def forward(self, x):
        return self.model(x)

# 训练设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 20
batch_size = 48  # 由于模型更大，减小batch size
learning_rate = 5e-5  # 使用更小的学习率

# 数据加载
train_dataset = FlowerDataset(txtpath, datapath=datapath, transform=transform_train, train=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# 模型初始化
model = AdvancedFlowerClassifier(num_classes=5).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 添加标签平滑
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# 记录训练过程
epoch_losses = []
epoch_accs = []

def plot_progress(train_losses, train_accs, val_losses=None, val_accs=None, clear=True):
    if clear:
        clear_output(wait=True)
    
    plt.figure(figsize=(15, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Training Loss')
    if val_losses:
        plt.plot(range(1, len(val_losses) + 1), val_losses, 'r--', label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accs) + 1), train_accs, 'b-', label='Training Accuracy')
    if val_accs:
        plt.plot(range(1, len(val_accs) + 1), val_accs, 'r--', label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    plt.close()

# 训练循环
for epoch in track(range(num_epochs), description="Training"):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
    scheduler.step()
    
    # 记录每个epoch的统计信息
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    epoch_losses.append(epoch_loss)
    epoch_accs.append(epoch_acc)
    
    print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
    
    # 每个epoch结束后更新图表
    if (epoch + 1) % 1 == 0:  # 每个epoch都更新
        plot_progress(epoch_losses, epoch_accs)

# 保存模型
torch.save(model.state_dict(), 'ConvNeXt_Large_flower_classifier.pth')