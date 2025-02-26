import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms
import timm
import os

# 数据预处理
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT 标准输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 类别映射
class_names = {
    0: "Cocos 1",
    1: "Apple 6",
    2: "Apple Golden 3",
    3: "Zucchini 1",
    4: "Granadilla 1",
    5: "Eggplant 1",
    6: "Physalis 1",
    7: "Banana 1",
    8: "Peach Flat 1",
    9: "Tamarillo 1",
    10: "Huckleberry 1",
    11: "Apple Crimson Snow 1",
    12: "Carambula 1",
    13: "Eggplant long 1",
    14: "Physalis with Husk 1",
    15: "Pear Kaiser 1",
    16: "Pear 2",
    17: "Strawberry Wedge 1",
    18: "Kiwi 1",
    19: "Walnut 1",
    20: "Maracuja 1",
    21: "Carrot 1",
    22: "Potato Sweet 1",
    23: "Guava 1",
    24: "Mango 1",
    25: "Pear Abate 1",
    26: "Strawberry 1",
    27: "Avocado 1",
    28: "Apple hit 1",
    29: "Kaki 1",
    30: "Grape White 4",
    31: "Pineapple 1",
    32: "Grape White 2",
    33: "Grape Blue 1",
    34: "Melon Piel de Sapo 1",
    35: "Watermelon 1",
    36: "Tomato 2",
    37: "Tomato not Ripened 1",
    38: "Corn Husk 1",
    39: "Onion Red Peeled 1",
    40: "Apple Braeburn 1",
    41: "Pepino 1",
    42: "Mango Red 1",
    43: "Kumquats 1",
    44: "Corn 1",
    45: "Pomelo Sweetie 1",
    46: "Rambutan 1",
    47: "Chestnut 1",
    48: "Grapefruit Pink 1",
    49: "Pitahaya Red 1",
    50: "Onion Red 1",
    51: "Apple Red 1",
    52: "Kohlrabi 1",
    53: "Tomato Maroon 1",
    54: "Cabbage white 1",
    55: "Plum 2",
    56: "Nut Forest 1",
    57: "Cherry Rainier 1",
    58: "Lemon Meyer 1",
    59: "Pepper Green 1",
    60: "Tomato Heart 1",
    61: "Pepper Yellow 1",
    62: "Salak 1",
    63: "Banana Red 1",
    64: "Pomegranate 1",
    65: "Tomato Yellow 1",
    66: "Cucumber Ripe 2",
    67: "Potato Red 1",
    68: "Apple Red 2",
    69: "Pear Stone 1",
    70: "Ginger Root 1",
    71: "Cactus fruit 1",
    72: "Apple Red 3",
    73: "Quince 1",
    74: "Grape White 1",
    75: "Cherry 1",
    76: "Cucumber 3",
    77: "Lychee 1",
    78: "Cherry 2",
    79: "Redcurrant 1",
    80: "Apricot 1",
    81: "Banana Lady Finger 1",
    82: "Cauliflower 1",
    83: "Cherry Wax Red 1",
    84: "Apple Red Delicious 1",
    85: "Passion Fruit 1",
    86: "Hazelnut 1",
    87: "Papaya 1",
    88: "Tomato 3",
    89: "Fig 1",
    90: "Mangostan 1",
    91: "Potato White 1",
    92: "Avocado ripe 1",
    93: "Apple Golden 1",
    94: "Peach 1",
    95: "Apple Red Yellow 2",
    96: "Limes 1",
    97: "Cherry Wax Yellow 1",
    98: "Tangelo 1",
    99: "Lemon 1",
    100: "Cherry Wax Black 1",
    101: "Orange 1",
    102: "Onion White 1",
    103: "Plum 1",
    104: "Pineapple Mini 1",
    105: "Pear Williams 1",
    106: "Zucchini dark 1",
    107: "Raspberry 1",
    108: "Pepper Red 1",
    109: "Tomato 1",
    110: "Dates 1",
    111: "Pear 1",
    112: "Grape Pink 1",
    113: "Cantaloupe 2",
    114: "Mandarine 1",
    115: "Grape White 3",
    116: "Grapefruit White 1",
    117: "Nut Pecan 1",
    118: "Apple Golden 2",
    119: "Apple Pink Lady 1",
    120: "Peach 2",
    121: "Tomato 4",
    122: "Nectarine Flat 1",
    123: "Plum 3",
    124: "Tomato Cherry Red 1",
    125: "Pear Red 1",
    126: "Blueberry 1",
    127: "Cucumber Ripe 1",
    128: "Pear Monster 1",
    129: "Pear Forelle 1",
    130: "Cucumber 1",
    131: "Beetroot 1",
    132: "Apple Red Yellow 1",
    133: "Pear 3",
    134: "Apple Granny Smith 1",
    135: "Pepper Orange 1",
    136: "Cantaloupe 1",
    137: "Mulberry 1",
    138: "Nectarine 1",
    139: "Potato Red Washed 1",
    140: "Clementine 1"
}

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

# 预测图片
def predict_image(image_path, model, device):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform_test(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        confidence = torch.max(probabilities).item() * 100
        predicted_class = class_names[predicted.item()]
    return predicted_class, confidence, image

# Tkinter 应用
class App:
    def __init__(self, root, model, device):
        self.root = root
        self.model = model
        self.device = device

        # 设置窗口
        self.root.title("Fruit Classifier")
        self.root.geometry("800x600")

        # 按钮：选择文件夹
        self.folder_button = tk.Button(root, text="Select Folder", command=self.select_folder, font=("Arial", 30), height=2, width=20)
        self.folder_button.pack(pady=20)

        # 模式选择
        self.mode_var = tk.IntVar(value=1)  # 默认选择模式一
        self.mode_frame = tk.Frame(root)
        self.mode_frame.pack(pady=10)
        self.mode1_radio = tk.Radiobutton(self.mode_frame, text="Mode 1: Show Image and Result", variable=self.mode_var, value=1, font=("Arial", 20))
        self.mode1_radio.pack(anchor=tk.W)
        self.mode2_radio = tk.Radiobutton(self.mode_frame, text="Mode 2: Generate Result File", variable=self.mode_var, value=2, font=("Arial", 20))
        self.mode2_radio.pack(anchor=tk.W)

        # 图片显示区域
        self.image_label = tk.Label(root)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # 结果标签
        self.result_label = tk.Label(root, text="", font=("Arial", 20))
        self.result_label.pack()

    def select_folder(self):
        # 打开文件夹选择对话框
        folder_path = filedialog.askdirectory()
        if folder_path:
            print(f"Selected Folder: {folder_path}")

            # 获取文件夹中的所有图片文件
            image_extensions = ('.jpg', '.jpeg', '.png')
            image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

            if not image_paths:
                messagebox.showwarning("Warning", "No supported image files found in the folder!", font=("Arial", 20))
                return

            # 根据选择的模式处理图片
            if self.mode_var.get() == 1:
                self.process_images_mode1(image_paths)
            else:
                self.process_images_mode2(folder_path, image_paths)

    def process_images_mode1(self, image_paths):
        # 模式一：依次显示图片和结果
        for image_path in image_paths:
            # 预测图片
            predicted_class, confidence, image = predict_image(image_path, self.model, self.device)

            # 显示结果
            self.result_label.config(text=f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}%", font=("Arial", 20))

            # 显示图片（固定放大到整个窗口）
            image = image.resize((800, 600), Image.LANCZOS)  # 调整图片大小，使用 LANCZOS 滤镜进行缩放
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo  # 保持引用，避免被垃圾回收

            # 在单独的窗口中显示结果
            result_window = tk.Toplevel(self.root)
            result_window.title("Result")
            result_window.geometry("400x200")
            result_label = tk.Label(result_window, text=f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}%", font=("Arial", 20))
            result_label.pack(pady=20)
            continue_button = tk.Button(result_window, text="Continue", command=result_window.destroy, font=("Arial", 20), height=2, width=10)
            continue_button.pack(pady=10)

            # 等待用户点击继续
            self.root.wait_window(result_window)

    def process_images_mode2(self, folder_path, image_paths):
        # 模式二：生成结果文件
        results_file = os.path.join(folder_path, "results.txt")
        with open(results_file, "w", encoding="utf-8") as f:
            for image_path in image_paths:
                # 预测图片
                predicted_class, confidence, _ = predict_image(image_path, self.model, self.device)

                # 写入结果
                f.write(f"Image: {os.path.basename(image_path)}\n")
                f.write(f"Predicted: {predicted_class}\n")
                f.write(f"Confidence: {confidence:.2f}%\n")
                f.write("-" * 50 + "\n")

        messagebox.showinfo("Complete", f"Results saved to {results_file}", font=("Arial", 20))

# 主函数
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model_path = "/home/ccy/workspace/deeplearning/learning_conda/src/fruit_det/fruit_141/fruits-360_dataset_100x100/fruits-360/vit_fruit_classifier_5epochs_final.pth"  # 替换为你的模型路径
    model = load_model(model_path, num_classes=141)
    model.to(device)

    # 创建 Tkinter 窗口
    root = tk.Tk()
    root.option_add("*Font", ("Arial", 20))  # 全局字体放大5倍
    app = App(root, model, device)
    root.mainloop()

if __name__ == "__main__":
    main()
