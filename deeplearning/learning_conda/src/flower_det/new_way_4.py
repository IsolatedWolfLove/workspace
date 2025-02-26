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
import json

console = Console()

# 使用您原有的数据路径
datapath = "/home/ccy/workspace/deeplearning/learning_conda/src/flower_det/archive(3)/files"
txtpath = "/home/ccy/workspace/deeplearning/learning_conda/src/flower_det/archive(3)/labels.csv"

# 数据预处理
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT 标准输入尺寸
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAutocontrast(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class FlowerDataset(Dataset):
    def __init__(self, txtpath, datapath, transform=None, train=True):
        self.transform = transform
        self.train = train
        self.data = []
        self.datapath = datapath
        
        with open(txtpath, 'r') as f:
            for line in f.readlines():
                if line.strip():  # 跳过空行
                    img_name, label = line.strip().split(',')
                    img_path = datapath + '/' + img_name + '.jpg'
                    # 将标签转换为整数，并减1使其从0开始
                    label = int(label) - 1  # 原始标签从1开始，减1使其从0开始
                    if 0 <= label < 102:  # 验证标签范围（0-101）
                        self.data.append((img_path, label))
                    else:
                        print(f"Warning: Invalid label {label+1} found in {img_name}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(label, dtype=torch.long)
        return image, label

class VisionTransformerClassifier(nn.Module):
    def __init__(self, num_classes=102):
        super().__init__()
        # 使用 ViT Large 作为骨干网络
        self.model = timm.create_model('vit_large_patch16_224', pretrained=True)
        
        # 获取特征维度
        num_ftrs = self.model.head.in_features
        
        # 修改分类头
        self.model.head = nn.Sequential(
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

# 训练设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 10 # 新的训练轮数
batch_size = 32
learning_rate = 1e-5  # 可以使用更小的学习率继续微调


# 数据加载
train_dataset = FlowerDataset(txtpath, datapath=datapath, transform=transform_train, train=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)

# 初始化模型
model = VisionTransformerClassifier(num_classes=102).to(device)

# 删除加载旧权重的部分，直接初始化优化器
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
epoch_losses = []
epoch_accs = []

scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

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
    if (epoch + 1) % 1 == 0:
        plot_progress(epoch_losses, epoch_accs)
    if((epoch+1) % 5 == 0):
    # 保存检查点
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_losses[-1],
            'accuracy': epoch_accs[-1]
        }, f'vit_102_flower_classifier_continued_{epoch+1}.pth')

# 保存最终模型和训练历史
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': epoch_losses[-1],
    'accuracy': epoch_accs[-1]
}, 'vit_102_flower_classifier_final.pth')

# 保存训练历史数据
history = {
    'loss': epoch_losses,
    'accuracy': epoch_accs
}
with open('vit_102_training_history_continued.json', 'w') as f:
    json.dump(history, f)
