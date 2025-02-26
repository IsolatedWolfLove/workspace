import torch
from torchvision import transforms
from PIL import Image
import timm
import torch.nn as nn
import os
from pathlib import Path
import matplotlib.pyplot as plt
import random

class FlowerClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # 使用 ViT Large 作为骨干网络
        self.model = timm.create_model('vit_large_patch16_224', pretrained=False)
        
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

    def forward(self, x):
        return self.model(x)

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT 的标准输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 类别映射
class_names = {
    0: 'daisy',
    1: 'dandelion',
    2: 'rose',
    3: 'sunflower',
    4: 'tulip'
}

def predict_image(image_path, model_path):
    try:
        # 加载模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FlowerClassifier(num_classes=5)
        
        # 加载训练好的权重
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()

        # 加载并预处理图片
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        # 预测
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            probability = torch.nn.functional.softmax(outputs, dim=1)
            confidence = torch.max(probability).item() * 100

        # 获取预测结果
        predicted_class = class_names[predicted.item()]
        
        return predicted_class, confidence
        
    except Exception as e:
        print(f"错误详情: {str(e)}")
        raise e

if __name__ == "__main__":
    # 设置模型路径和图片文件夹路径
    model_path = "/home/ccy/workspace/deeplearning/learning_conda/vit_flower_classifier.pth"
    image_folder = "/home/ccy/workspace/deeplearning/learning_conda/src/flower_det/flower"
    
    # 支持的图片格式
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    
    try:
        # 获取所有符合条件的图片路径并转换为列表
        image_paths = [img_path for img_path in Path(image_folder).glob('*') 
                      if img_path.suffix.lower() in image_extensions]
        
        # 随机打乱图片顺序
        random.shuffle(image_paths)
        
        # 遍历打乱后的图片列表
        for img_path in image_paths:
            print(f"\n处理图片: {img_path.name}")
            
            # 预测
            predicted_class, confidence = predict_image(str(img_path), model_path)
            print(f"预测结果: {predicted_class}")
            print(f"置信度: {confidence:.2f}%")
            
            # 显示图片和预测结果
            import matplotlib.pyplot as plt
            image = Image.open(img_path)
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.axis('off')
            plt.title(f'predicted class: {predicted_class} (confidence: {confidence:.2f}%)')
            plt.show()
            plt.close()
                
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
