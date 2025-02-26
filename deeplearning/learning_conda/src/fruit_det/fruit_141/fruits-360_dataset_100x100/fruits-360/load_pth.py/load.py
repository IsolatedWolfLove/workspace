import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import timm
import matplotlib.pyplot as plt

# 数据预处理
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT 标准输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 自定义数据集类
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))  # 获取所有类别文件夹
        self.class_to_idx = {
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
        }  # 使用你提供的索引
        self.samples = self._make_dataset()  # 获取所有样本路径和标签

    def _make_dataset(self):
        samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                samples.append((img_path, self.class_to_idx[cls_name]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, img_path

# 模型定义
class VisionTransformerClassifier(nn.Module):
    def __init__(self, num_classes=141):
        super().__init__()
        self.model = timm.create_model('vit_large_patch16_224', pretrained=False)
        num_ftrs = self.model.head.in_features
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

    def forward(self, x):
        return self.model(x)

# 加载模型
def load_model(model_path, num_classes=141):
    model = VisionTransformerClassifier(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 设置为评估模式
    return model

# 测试函数（逐张检测并输出结果）
def test_model(model, test_loader, device):
    with torch.no_grad():  # 禁用梯度计算
        for batch_idx, (images, labels, img_paths) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            # 逐张输出结果
            for i in range(len(images)):
                img_path = img_paths[i]
                predicted_class = list(test_loader.dataset.class_to_idx.keys())[list(test_loader.dataset.class_to_idx.values()).index(predicted[i].item())]
                true_class = list(test_loader.dataset.class_to_idx.keys())[list(test_loader.dataset.class_to_idx.values()).index(labels[i].item())]
                confidence = torch.max(probabilities[i]).item() * 100

                # 打印结果
                print(f"Image: {img_path}")
                print(f"Predicted Class: {predicted_class}, True Class: {true_class}")
                print(f"Confidence: {confidence:.2f}%")
                print("-" * 50)

                # 显示图片
                image = Image.open(img_path)
                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                plt.axis('off')
                plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}%")
                plt.show()

# 主函数
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载测试数据集
    test_dir = "/home/ccy/workspace/deeplearning/learning_conda/src/fruit_det/fruit_141/fruits-360_dataset_100x100/fruits-360/Test"
    test_dataset = TestDataset(root_dir=test_dir, transform=transform_test)

    # 打乱数据
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)

    # 加载模型
    model_path = "/home/ccy/workspace/deeplearning/learning_conda/src/fruit_det/fruit_141/fruits-360_dataset_100x100/fruits-360/vit_fruit_classifier_5epochs_final.pth"  # 替换为你的模型路径
    model = load_model(model_path, num_classes=141)
    model.to(device)

    # 测试模型（逐张检测并输出结果）
    test_model(model, test_loader, device)

if __name__ == "__main__":
    main()
