# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# import timm
# from torch.optim.lr_scheduler import CosineAnnealingLR
# import numpy as np
# import matplotlib.pyplot as plt
# from rich.console import Console
# from rich.progress import track
# from IPython.display import clear_output
# import json

# console = Console()

# # 使用您原有的数据路径
# datapath = "/home/ccy/workspace/deeplearning/learning_conda/src/fruit_det/fruit_141/fruits-360_dataset_100x100/fruits-360/Training"
# txtpath = "/home/ccy/workspace/deeplearning/learning_conda/src/fruit_det/fruit_141/fruits-360_dataset_100x100/fruits-360/label_1.txt"

# # 数据预处理
# transform_train = transforms.Compose([
#     transforms.Resize((224, 224)),  # ViT 标准输入尺寸
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomVerticalFlip(p=0.3),
#     transforms.RandomRotation(30),
#     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     transforms.RandomAutocontrast(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# class FlowerDataset(Dataset):
#     def __init__(self, txtpath, datapath, transform=True, train=True):
#         self.transform = transform
#         self.train = train
#         self.data = []
#         self.datapath = datapath
        
#         with open(txtpath, 'r') as f:
#             for line in f.readlines():
#                 if line.strip():  # 跳过空行
#                     img_name, label = line.strip().split(',')
#                     # print(img_name, label)
#                     img_path = datapath + '/' + img_name 
#                     # 将标签转换为整数，并减1使其从0开始
#                     if label=="Cocos 1":
#                         label=0
#                     if label=="Apple 6":
#                         label=1
#                     if label=="Apple Golden 3":
#                         label=2
#                     if label=="Zucchini 1":
#                         label=3
#                     if label=="Granadilla 1":
#                         label=4
#                     if label=="Eggplant 1":
#                         label=5
#                     if label=="Physalis 1":
#                         label=6
#                     if label=="Banana 1":
#                         label=7
#                     if label=="Peach Flat 1":
#                         label=8
#                     if label=="Tamarillo 1":
#                         label=9
#                     if label=="Huckleberry 1":
#                         label=10
#                     if label=="Apple Crimson Snow 1":
#                         label=11
#                     if label=="Carambula 1":
#                         label=12
#                     if label=="Eggplant long 1":
#                         label=13
#                     if label=="Physalis with Husk 1":
#                         label=14
#                     if label=="Pear Kaiser 1":
#                         label=15
#                     if label=="Pear 2":
#                         label=16
#                     if label=="Strawberry Wedge 1":
#                         label=17
#                     if label=="Kiwi 1":
#                         label=18
#                     if label=="Walnut 1":
#                         label=19
#                     if label=="Maracuja 1":
#                         label=20
#                     if label=="Carrot 1":
#                         label=21
#                     if label=="Potato Sweet 1":
#                         label=22
#                     if label=="Guava 1":
#                         label=23
#                     if label=="Mango 1":
#                         label=24
#                     if label=="Pear Abate 1":
#                         label=25
#                     if label=="Strawberry 1":
#                         label=26
#                     if label=="Avocado 1":
#                         label=27
#                     if label=="Apple hit 1":
#                         label=28
#                     if label=="Kaki 1":
#                         label=29
#                     if label=="Grape White 4":
#                         label=30
#                     if label=="Pineapple 1":
#                         label=31
#                     if label=="Grape White 2":
#                         label=32
#                     if label=="Grape Blue 1":
#                         label=33
#                     if label=="Melon Piel de Sapo 1":
#                         label=34
#                     if label=="Watermelon 1":
#                         label=35
#                     if label=="Tomato 2":
#                         label=36
#                     if label=="Tomato not Ripened 1":
#                         label=37
#                     if label=="Corn Husk 1":
#                         label=38
#                     if label=="Onion Red Peeled 1":
#                         label=39
#                     if label=="Apple Braeburn 1":
#                         label=40
#                     if label=="Pepino 1":
#                         label=41
#                     if label=="Mango Red 1":
#                         label=42
#                     if label=="Kumquats 1":
#                         label=43
#                     if label=="Corn 1":
#                         label=44
#                     if label=="Pomelo Sweetie 1":
#                         label=45
#                     if label=="Rambutan 1":
#                         label=46
#                     if label=="Chestnut 1":
#                         label=47
#                     if label=="Grapefruit Pink 1":
#                         label=48
#                     if label=="Pitahaya Red 1":
#                         label=49
#                     if label=="Onion Red 1":
#                         label=50
#                     if label=="Apple Red 1":
#                         label=51
#                     if label=="Kohlrabi 1":
#                         label=52
#                     if label=="Tomato Maroon 1":
#                         label=53
#                     if label=="Cabbage white 1":
#                         label=54
#                     if label=="Plum 2":
#                         label=55
#                     if label=="Nut Forest 1":
#                         label=56
#                     if label=="Cherry Rainier 1":
#                         label=57
#                     if label=="Lemon Meyer 1":
#                         label=58
#                     if label=="Pepper Green 1":
#                         label=59
#                     if label=="Tomato Heart 1":
#                         label=60
#                     if label=="Pepper Yellow 1":
#                         label=61
#                     if label=="Salak 1":
#                         label=62
#                     if label=="Banana Red 1":
#                         label=63
#                     if label=="Pomegranate 1":
#                         label=64
#                     if label=="Tomato Yellow 1":
#                         label=65
#                     if label=="Cucumber Ripe 2":
#                         label=66
#                     if label=="Potato Red 1":
#                         label=67
#                     if label=="Apple Red 2":
#                         label=68
#                     if label=="Pear Stone 1":
#                         label=69
#                     if label=="Ginger Root 1":
#                         label=70
#                     if label=="Cactus fruit 1":
#                         label=71
#                     if label=="Apple Red 3":
#                         label=72
#                     if label=="Quince 1":
#                         label=73
#                     if label=="Grape White 1":
#                         label=74
#                     if label=="Cherry 1":
#                         label=75
#                     if label=="Cucumber 3":
#                         label=76
#                     if label=="Lychee 1":
#                         label=77
#                     if label=="Cherry 2":
#                         label=78
#                     if label=="Redcurrant 1":
#                         label=79
#                     if label=="Apricot 1":
#                         label=80
#                     if label=="Banana Lady Finger 1":
#                         label=81
#                     if label=="Cauliflower 1":
#                         label=82
#                     if label=="Cherry Wax Red 1":
#                         label=83
#                     if label=="Apple Red Delicious 1":
#                         label=84
#                     if label=="Passion Fruit 1":
#                         label=85
#                     if label=="Hazelnut 1":
#                         label=86
#                     if label=="Papaya 1":
#                         label=87
#                     if label=="Tomato 3":
#                         label=88
#                     if label=="Fig 1":
#                         label=89
#                     if label=="Mangostan 1":
#                         label=90
#                     if label=="Potato White 1":
#                         label=91
#                     if label=="Avocado ripe 1":
#                         label=92
#                     if label=="Apple Golden 1":
#                         label=93
#                     if label=="Peach 1":
#                         label=94
#                     if label=="Apple Red Yellow 2":
#                         label=95
#                     if label=="Limes 1":
#                         label=96
#                     if label=="Cherry Wax Yellow 1":
#                         label=97
#                     if label=="Tangelo 1":
#                         label=98
#                     if label=="Lemon 1":
#                         label=99
#                     if label=="Cherry Wax Black 1":
#                         label=100
#                     if label=="Orange 1":
#                         label=101
#                     if label=="Onion White 1":
#                         label=102
#                     if label=="Plum 1":
#                         label=103
#                     if label=="Pineapple Mini 1":
#                         label=104
#                     if label=="Pear Williams 1":
#                         label=105
#                     if label=="Zucchini dark 1":
#                         label=106
#                     if label=="Raspberry 1":
#                         label=107
#                     if label=="Pepper Red 1":
#                         label=108
#                     if label=="Tomato 1":
#                         label=109
#                     if label=="Dates 1":
#                         label=110
#                     if label=="Pear 1":
#                         label=111
#                     if label=="Grape Pink 1":
#                         label=112
#                     if label=="Cantaloupe 2":
#                         label=113
#                     if label=="Mandarine 1":
#                         label=114
#                     if label=="Grape White 3":
#                         label=115
#                     if label=="Grapefruit White 1":
#                         label=116
#                     if label=="Nut Pecan 1":
#                         label=117
#                     if label=="Apple Golden 2":
#                         label=118
#                     if label=="Apple Pink Lady 1":
#                         label=119
#                     if label=="Peach 2":
#                         label=120
#                     if label=="Tomato 4":
#                         label=121
#                     if label=="Nectarine Flat 1":
#                         label=122
#                     if label=="Plum 3":
#                         label=123
#                     if label=="Tomato Cherry Red 1":
#                         label=124
#                     if label=="Pear Red 1":
#                         label=125
#                     if label=="Blueberry 1":
#                         label=126
#                     if label=="Cucumber Ripe 1":
#                         label=127
#                     if label=="Pear Monster 1":
#                         label=128
#                     if label=="Pear Forelle 1":
#                         label=129
#                     if label=="Cucumber 1":
#                         label=130
#                     if label=="Beetroot 1":
#                         label=131
#                     if label=="Apple Red Yellow 1":
#                         label=132
#                     if label=="Pear 3":
#                         label=133
#                     if label=="Apple Granny Smith 1":
#                         label=134
#                     if label=="Pepper Orange 1":
#                         label=135
#                     if label=="Cantaloupe 1":
#                         label=136
#                     if label=="Mulberry 1":
#                         label=137
#                     if label=="Nectarine 1":
#                         label=138
#                     if label=="Potato Red Washed 1":
#                         label=139
#                     if label=="Clementine 1":
#                         label=140
#                     if 0 <= label < 141:  # 验证标签范围（0-101）
#                         self.data.append((img_path, label))
#                     else:
#                         print(f"Warning: Invalid label {label+1} found in {img_name}")

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         img_path, label = self.data[idx]
#         image = Image.open(img_path).convert('RGB')
        
#         if self.transform:
#             image = self.transform(image)
        
#         label = torch.tensor(label, dtype=torch.long)
#         return image, label

# class VisionTransformerClassifier(nn.Module):
#     def __init__(self, num_classes=141):
#         super().__init__()
#         # 使用 ViT Large 作为骨干网络
#         self.model = timm.create_model('vit_large_patch16_224', pretrained=True)
        
#         # 获取特征维度
#         num_ftrs = self.model.head.in_features
        
#         # 修改分类头
#         self.model.head = nn.Sequential(
#             nn.LayerNorm(num_ftrs),
#             nn.Linear(num_ftrs, 1024),
#             nn.GELU(),
#             nn.Dropout(0.4),
#             nn.Linear(1024, 512),
#             nn.GELU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, num_classes)
#         )
        
#         # 冻结部分层
#         self._freeze_layers()
        
#     def _freeze_layers(self):
#         # 冻结前面的层
#         for param in list(self.model.parameters())[:-30]:
#             param.requires_grad = False
            
#     def forward(self, x):
#         return self.model(x)

# def plot_progress(train_losses, train_accs, val_losses=None, val_accs=None, clear=True):
#     if clear:
#         clear_output(wait=True)
    
#     plt.figure(figsize=(15, 5))
    
#     # 损失曲线
#     plt.subplot(1, 2, 1)
#     plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Training Loss')
#     if val_losses:
#         plt.plot(range(1, len(val_losses) + 1), val_losses, 'r--', label='Validation Loss')
#     plt.title('Loss Over Epochs')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
    
#     # 准确率曲线
#     plt.subplot(1, 2, 2)
#     plt.plot(range(1, len(train_accs) + 1), train_accs, 'b-', label='Training Accuracy')
#     if val_accs:
#         plt.plot(range(1, len(val_accs) + 1), val_accs, 'r--', label='Validation Accuracy')
#     plt.title('Accuracy Over Epochs')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy (%)')
#     plt.legend()
#     plt.grid(True)
    
#     plt.tight_layout()
#     plt.show()
#     plt.close()

# # 训练设置
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# num_epochs = 2 # 新的训练轮数
# batch_size = 24
# learning_rate = 1e-4  # 可以使用更小的学习率继续微调


# # 数据加载
# train_dataset = FlowerDataset(txtpath, datapath=datapath, transform=transform_train, train=True)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=24)

# # 初始化模型
# model = VisionTransformerClassifier(num_classes=141).to(device)

# # 删除加载旧权重的部分，直接初始化优化器
# optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
# epoch_losses = []
# epoch_accs = []

# scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
# criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# # 训练循环
# for epoch in track(range(num_epochs), description="Training"):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
    
#     for i, (images, labels) in enumerate(train_loader):
#         images, labels = images.to(device), labels.to(device)
        
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()
        
#         if (i + 1) % 10 == 0:
#             print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
#                   f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
#     scheduler.step()
    
#     # 记录每个epoch的统计信息
#     epoch_loss = running_loss / len(train_loader)
#     epoch_acc = 100. * correct / total
#     epoch_losses.append(epoch_loss)
#     epoch_accs.append(epoch_acc)
    
#     print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
    
#     # 每个epoch结束后更新图表
#     if (epoch + 1) % 2 == 0:
#         plot_progress(epoch_losses, epoch_accs)    



        

# # 保存最终模型和训练历史
# torch.save({
#     'epoch': num_epochs,
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'loss': epoch_losses[-1],
#     'accuracy': epoch_accs[-1]
# }, 'vit_fruit_classifier_final.pth')

# # 保存训练历史数据
# history = {
#     'loss': epoch_losses,
#     'accuracy': epoch_accs
# }
# with open('vit_fruit_classifier_training_history_continued.json', 'w') as f:
#     json.dump(history, f)





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
datapath = "/home/ccy/workspace/deeplearning/learning_conda/src/fruit_det/fruit_141/fruits-360_dataset_100x100/fruits-360/Training"
txtpath = "/home/ccy/workspace/deeplearning/learning_conda/src/fruit_det/fruit_141/fruits-360_dataset_100x100/fruits-360/label_1.txt"

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
    def __init__(self, txtpath, datapath, transform=True, train=True):
        self.transform = transform
        self.train = train
        self.data = []
        self.datapath = datapath
        
        with open(txtpath, 'r') as f:
            for line in f.readlines():
                if line.strip():  # 跳过空行
                    img_name, label = line.strip().split(',')
                    img_path = datapath + '/' + img_name 
                    label_map = {
                        "Cocos 1": 0,
                        "Apple 6": 1,
                        "Apple Golden 3": 2,
                        "Zucchini 1": 3,
                        "Granadilla 1": 4,
                        "Eggplant 1": 5,
                        "Physalis 1": 6,
                        "Banana 1": 7,
                        "Peach Flat 1": 8,
                        "Tamarillo 1": 9,
                        "Huckleberry 1": 10,
                        "Apple Crimson Snow 1": 11,
                        "Carambula 1": 12,
                        "Eggplant long 1": 13,
                        "Physalis with Husk 1": 14,
                        "Pear Kaiser 1": 15,
                        "Pear 2": 16,
                        "Strawberry Wedge 1": 17,
                        "Kiwi 1": 18,
                        "Walnut 1": 19,
                        "Maracuja 1": 20,
                        "Carrot 1": 21,
                        "Potato Sweet 1": 22,
                        "Guava 1": 23,
                        "Mango 1": 24,
                        "Pear Abate 1": 25,
                        "Strawberry 1": 26,
                        "Avocado 1": 27,
                        "Apple hit 1": 28,
                        "Kaki 1": 29,
                        "Grape White 4": 30,
                        "Pineapple 1": 31,
                        "Grape White 2": 32,
                        "Grape Blue 1": 33,
                        "Melon Piel de Sapo 1": 34,
                        "Watermelon 1": 35,
                        "Tomato 2": 36,
                        "Tomato not Ripened 1": 37,
                        "Corn Husk 1": 38,
                        "Onion Red Peeled 1": 39,
                        "Apple Braeburn 1": 40,
                        "Pepino 1": 41,
                        "Mango Red 1": 42,
                        "Kumquats 1": 43,
                        "Corn 1": 44,
                        "Pomelo Sweetie 1": 45,
                        "Rambutan 1": 46,
                        "Chestnut 1": 47,
                        "Grapefruit Pink 1": 48,
                        "Pitahaya Red 1": 49,
                        "Onion Red 1": 50,
                        "Apple Red 1": 51,
                        "Kohlrabi 1": 52,
                        "Tomato Maroon 1": 53,
                        "Cabbage white 1": 54,
                        "Plum 2": 55,
                        "Nut Forest 1": 56,
                        "Cherry Rainier 1": 57,
                        "Lemon Meyer 1": 58,
                        "Pepper Green 1": 59,
                        "Tomato Heart 1": 60,
                        "Pepper Yellow 1": 61,
                        "Salak 1": 62,
                        "Banana Red 1": 63,
                        "Pomegranate 1": 64,
                        "Tomato Yellow 1": 65,
                        "Cucumber Ripe 2": 66,
                        "Potato Red 1": 67,
                        "Apple Red 2": 68,
                        "Pear Stone 1": 69,
                        "Ginger Root 1": 70,
                        "Cactus fruit 1": 71,
                        "Apple Red 3": 72,
                        "Quince 1": 73,
                        "Grape White 1": 74,
                        "Cherry 1": 75,
                        "Cucumber 3": 76,
                        "Lychee 1": 77,
                        "Cherry 2": 78,
                        "Redcurrant 1": 79,
                        "Apricot 1": 80,
                        "Banana Lady Finger 1": 81,
                        "Cauliflower 1": 82,
                        "Cherry Wax Red 1": 83,
                        "Apple Red Delicious 1": 84,
                        "Passion Fruit 1": 85,
                        "Hazelnut 1": 86,
                        "Papaya 1": 87,
                        "Tomato 3": 88,
                        "Fig 1": 89,
                        "Mangostan 1": 90,
                        "Potato White 1": 91,
                        "Avocado ripe 1": 92,
                        "Apple Golden 1": 93,
                        "Peach 1": 94,
                        "Apple Red Yellow 2": 95,
                        "Limes 1": 96,
                        "Cherry Wax Yellow 1": 97,
                        "Tangelo 1": 98,
                        "Lemon 1": 99,
                        "Cherry Wax Black 1": 100,
                        "Orange 1": 101,
                        "Onion White 1": 102,
                        "Plum 1": 103,
                        "Pineapple Mini 1": 104,
                        "Pear Williams 1": 105,
                        "Zucchini dark 1": 106,
                        "Raspberry 1": 107,
                        "Pepper Red 1": 108,
                        "Tomato 1": 109,
                        "Dates 1": 110,
                        "Pear 1": 111,
                        "Grape Pink 1": 112,
                        "Cantaloupe 2": 113,
                        "Mandarine 1": 114,
                        "Grape White 3": 115,
                        "Grapefruit White 1": 116,
                        "Nut Pecan 1": 117,
                        "Apple Golden 2": 118,
                        "Apple Pink Lady 1": 119,
                        "Peach 2": 120,
                        "Tomato 4": 121,
                        "Nectarine Flat 1": 122,
                        "Plum 3": 123,
                        "Tomato Cherry Red 1": 124,
                        "Pear Red 1": 125,
                        "Blueberry 1": 126,
                        "Cucumber Ripe 1": 127,
                        "Pear Monster 1": 128,
                        "Pear Forelle 1": 129,
                        "Cucumber 1": 130,
                        "Beetroot 1": 131,
                        "Apple Red Yellow 1": 132,
                        "Pear 3": 133,
                        "Apple Granny Smith 1": 134,
                        "Pepper Orange 1": 135,
                        "Cantaloupe 1": 136,
                        "Mulberry 1": 137,
                        "Nectarine 1": 138,
                        "Potato Red Washed 1": 139,
                        "Clementine 1": 140
                    }
                    
                    # 使用标签映射
                    label = label_map.get(label, -1)
                    if 0 <= label < 141:  # 验证标签范围（0-140）
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
    def __init__(self, num_classes=141):
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
num_epochs = 1  # 新的训练轮数
batch_size = 32
learning_rate = 1e-7  # 可以使用更小的学习率继续微调

# 数据加载
train_dataset = FlowerDataset(txtpath, datapath=datapath, transform=transform_train, train=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=24)

# 初始化模型
model = VisionTransformerClassifier(num_classes=141).to(device)

# 尝试加载之前的模型和优化器状态
try:
    checkpoint = torch.load('/home/ccy/workspace/deeplearning/learning_conda/src/fruit_det/fruit_141/fruits-360_dataset_100x100/fruits-360/vit_fruit_classifier_4epochs_final.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']  # 获取开始训练的轮数
    
    # 确保 epoch_losses 和 epoch_accs 是列表
    if isinstance(checkpoint['loss'], list):
        epoch_losses = checkpoint['loss']
    else:
        epoch_losses = []

    if isinstance(checkpoint['accuracy'], list):
        epoch_accs = checkpoint['accuracy']
    else:
        epoch_accs = []
    
    print(f"成功加载之前保存的模型，开始训练第 {start_epoch + 1} 轮。")
except FileNotFoundError:
    print("没有找到之前保存的模型，开始新的训练。")
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    start_epoch = 0
    epoch_losses = []  # 初始化为列表
    epoch_accs = []    # 初始化为列表

scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 训练循环
for epoch in track(range(start_epoch, start_epoch + num_epochs), description="Training"):
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
            print(f'Epoch [{epoch + 1}/{start_epoch + num_epochs}], Step [{i + 1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}, Acc: {100. * correct / total:.2f}%')
    
    scheduler.step()
    
    # 记录每个epoch的统计信息
    try:
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        epoch_losses.append(epoch_loss)  # 将损失添加到列表
        epoch_accs.append(epoch_acc) 
    except:
        pass# 将准确率添加到列表

# 保存最终模型和训练历史
torch.save({
    'epoch': start_epoch + num_epochs,  # 保存当前的总训练轮数
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': epoch_losses,
    'accuracy': epoch_accs
}, 'vit_fruit_classifier_5epochs_final.pth')

# 保存训练历史数据
history = {
    'loss': epoch_losses,
    'accuracy': epoch_accs
}
with open('vit_fruit_classifier_training_history_5epochs_continued.json', 'w') as f:
    json.dump(history, f)


