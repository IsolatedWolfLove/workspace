# import torch
# from torchvision import transforms
# from PIL import Image
# import timm
# import torch.nn as nn
# import os
# from pathlib import Path
# import matplotlib.pyplot as plt
# import random

# class FlowerClassifier(nn.Module):
#     def __init__(self, num_classes=141):
#         super().__init__()
#         # 使用 ViT Large 作为骨干网络
#         self.model = timm.create_model('vit_large_patch16_224', pretrained=False)
        
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

#     def forward(self, x):
#         return self.model(x)

# # 图像预处理
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # ViT 的标准输入尺寸
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # 类别映射
# class_names = {
# 0: "Cocos 1",
# 1: "Apple 6",
# 2: "Apple Golden 3",
# 3: "Zucchini 1",
# 4: "Granadilla 1",
# 5: "Eggplant 1",
# 6: "Physalis 1",
# 7: "Banana 1",
# 8: "Peach Flat 1",
# 9: "Tamarillo 1",
# 10: "Huckleberry 1",
# 11: "Apple Crimson Snow 1",
# 12: "Carambula 1",
# 13: "Eggplant long 1",
# 14: "Physalis with Husk 1",
# 15: "Pear Kaiser 1",
# 16: "Pear 2",
# 17: "Strawberry Wedge 1",
# 18: "Kiwi 1",
# 19: "Walnut 1",
# 20: "Maracuja 1",
# 21: "Carrot 1",
# 22: "Potato Sweet 1",
# 23: "Guava 1",
# 24: "Mango 1",
# 25: "Pear Abate 1",
# 26: "Strawberry 1",
# 27: "Avocado 1",
# 28: "Apple hit 1",
# 29: "Kaki 1",
# 30: "Grape White 4",
# 31: "Pineapple 1",
# 32: "Grape White 2",
# 33: "Grape Blue 1",
# 34: "Melon Piel de Sapo 1",
# 35: "Watermelon 1",
# 36: "Tomato 2",
# 37: "Tomato not Ripened 1",
# 38: "Corn Husk 1",
# 39: "Onion Red Peeled 1",
# 40: "Apple Braeburn 1",
# 41: "Pepino 1",
# 42: "Mango Red 1",
# 43: "Kumquats 1",
# 44: "Corn 1",
# 45: "Pomelo Sweetie 1",
# 46: "Rambutan 1",
# 47: "Chestnut 1",
# 48: "Grapefruit Pink 1",
# 49: "Pitahaya Red 1",
# 50: "Onion Red 1",
# 51: "Apple Red 1",
# 52: "Kohlrabi 1",
# 53: "Tomato Maroon 1",
# 54: "Cabbage white 1",
# 55: "Plum 2",
# 56: "Nut Forest 1",
# 57: "Cherry Rainier 1",
# 58: "Lemon Meyer 1",
# 59: "Pepper Green 1",
# 60: "Tomato Heart 1",
# 61: "Pepper Yellow 1",
# 62: "Salak 1",
# 63: "Banana Red 1",
# 64: "Pomegranate 1",
# 65: "Tomato Yellow 1",
# 66: "Cucumber Ripe 2",
# 67: "Potato Red 1",
# 68: "Apple Red 2",
# 69: "Pear Stone 1",
# 70: "Ginger Root 1",
# 71: "Cactus fruit 1",
# 72: "Apple Red 3",
# 73: "Quince 1",
# 74: "Grape White 1",
# 75: "Cherry 1",
# 76: "Cucumber 3",
# 77: "Lychee 1",
# 78: "Cherry 2",
# 79: "Redcurrant 1",
# 80: "Apricot 1",
# 81: "Banana Lady Finger 1",
# 82: "Cauliflower 1",
# 83: "Cherry Wax Red 1",
# 84: "Apple Red Delicious 1",
# 85: "Passion Fruit 1",
# 86: "Hazelnut 1",
# 87: "Papaya 1",
# 88: "Tomato 3",
# 89: "Fig 1",
# 90: "Mangostan 1",
# 91: "Potato White 1",
# 92: "Avocado ripe 1",
# 93: "Apple Golden 1",
# 94: "Peach 1",
# 95: "Apple Red Yellow 2",
# 96: "Limes 1",
# 97: "Cherry Wax Yellow 1",
# 98: "Tangelo 1",
# 99: "Lemon 1",
# 100: "Cherry Wax Black 1",
# 101: "Orange 1",
# 102: "Onion White 1",
# 103: "Plum 1",
# 104: "Pineapple Mini 1",
# 105: "Pear Williams 1",
# 106: "Zucchini dark 1",
# 107: "Raspberry 1",
# 108: "Pepper Red 1",
# 109: "Tomato 1",
# 110: "Dates 1",
# 111: "Pear 1",
# 112: "Grape Pink 1",
# 113: "Cantaloupe 2",
# 114: "Mandarine 1",
# 115: "Grape White 3",
# 116: "Grapefruit White 1",
# 117: "Nut Pecan 1",
# 118: "Apple Golden 2",
# 119: "Apple Pink Lady 1",
# 120: "Peach 2",
# 121: "Tomato 4",
# 122: "Nectarine Flat 1",
# 123: "Plum 3",
# 124: "Tomato Cherry Red 1",
# 125: "Pear Red 1",
# 126: "Blueberry 1",
# 127: "Cucumber Ripe 1",
# 128: "Pear Monster 1",
# 129: "Pear Forelle 1",
# 130: "Cucumber 1",
# 131: "Beetroot 1",
# 132: "Apple Red Yellow 1",
# 133: "Pear 3",
# 134: "Apple Granny Smith 1",
# 135: "Pepper Orange 1",
# 136: "Cantaloupe 1",
# 137: "Mulberry 1",
# 138: "Nectarine 1",
# 139: "Potato Red Washed 1",
# 140: "Clementine 1",
# }

# def predict_image(image_path, model_path):
#     try:
#         # 加载模型
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model = FlowerClassifier(num_classes=141)
        
#         # 加载训练好的权重
#         checkpoint = torch.load(model_path, map_location=device)
#         if 'model_state_dict' in checkpoint:
#             model.load_state_dict(checkpoint['model_state_dict'])
#         else:
#             model.load_state_dict(checkpoint)
        
#         model.to(device)
#         model.eval()

#         # 加载并预处理图片
#         image = Image.open(image_path).convert('RGB')
#         image_tensor = transform(image).unsqueeze(0).to(device)

#         # 预测
#         with torch.no_grad():
#             outputs = model(image_tensor)
#             _, predicted = torch.max(outputs, 1)
#             probability = torch.nn.functional.softmax(outputs, dim=1)
#             confidence = torch.max(probability).item() * 100

#         # 获取预测结果
#         predicted_class = class_names[predicted.item()]
        
#         return predicted_class, confidence
        
#     except Exception as e:
#         print(f"错误详情: {str(e)}")
#         raise e

# if __name__ == "__main__":
#     # 设置模型路径和图片文件夹路径
#     model_path = "/home/ccy/workspace/deeplearning/learning_conda/src/fruit_det/fruit_141/fruits-360_dataset_100x100/fruits-360/vit_fruit_classifier_5epochs_final.pth"
#     image_folder = "/home/ccy/workspace/deeplearning/learning_conda/src/fruit_det/fruit_141/fruits-360_dataset_100x100/fruits-360/fruit"
    
#     # 支持的图片格式
#     image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    
#     try:
#         # 获取所有符合条件的图片路径并转换为列表
#         image_paths = [img_path for img_path in Path(image_folder).glob('*') 
#                       if img_path.suffix.lower() in image_extensions]
        
#         # 随机打乱图片顺序
#         random.shuffle(image_paths)
        
#         # 遍历打乱后的图片列表
#         for img_path in image_paths:
#             print(f"\n处理图片: {img_path.name}")
            
#             # 预测
#             predicted_class, confidence = predict_image(str(img_path), model_path)
#             print(f"预测结果: {predicted_class}")
#             print(f"置信度: {confidence:.2f}%")
            
#             # 显示图片和预测结果
#             import matplotlib.pyplot as plt
#             image = Image.open(img_path)
#             plt.figure(figsize=(10, 10))
#             plt.imshow(image)
#             plt.axis('off')
#             plt.title(f'predicted class: {predicted_class} (confidence: {confidence:.2f}%)')
#             plt.show()
#             plt.close()
                
#     except Exception as e:
#         print(f"处理过程中出现错误: {str(e)}")










import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import timm

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
                # print(f"Image: {img_path}, Label: {self.class_to_idx[cls_name]}")  # 打印样本和标签
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            # print(f"Transformed Image Shape: {image.shape}")  # 打印处理后的图像形状
        return image, label

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

# 测试函数
def test_model(model, test_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():  # 禁用梯度计算
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 打印模型输出和预测结果
            # print(f"Model Outputs: {outputs}")
            # print(f"Predicted: {predicted}, Labels: {labels}")

            # 每隔 200 张图片显示一次准确度
            if (batch_idx + 1) * test_loader.batch_size % 200 == 0:
                accuracy = 100. * correct / total
                print(f'Processed {total} images, Current Accuracy: {accuracy:.2f}%')

    # 最终准确度
    accuracy = 100. * correct / total
    print(f'Final Test Accuracy: {accuracy:.2f}%')

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

    # 测试模型
    test_model(model, test_loader, device)

if __name__ == "__main__":
    main()


