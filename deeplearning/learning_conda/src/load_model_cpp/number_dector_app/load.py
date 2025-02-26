import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw

class ImageClassifier:
    def __init__(self, model_path: str):
        self.default_batch = 1
        self.default_size = 28
        self.picture_size = (self.default_size, self.default_size)
        self.model_path = model_path
        self.image = None
        self.load_model()

    def set_model_path(self, model_path: str):
        self.model_path = model_path

    def classify_image(self, image):
        # 确保输入图像是灰度图
        if len(image.shape) == 3:
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            self.image = image
            
        input_tensor = self.picture_fit_model()
        if input_tensor is None:
            return "Error"
            
        self.net.setInput(input_tensor)
        output = self.net.forward()
        output_vec = output.reshape(1, 10).flatten()
        max_index = 0
        max_value = output_vec[0]
        for i in range(1, 10):
            if output_vec[i] > max_value:
                max_index = i
                max_value = output_vec[i]
        
        return str(max_index)

    def change_picture_size(self, width: int, height: int):
        self.picture_size = (width, height)

    def change_picture_batch(self, batch: int):
        self.default_batch = batch

    def picture_fit_model(self):
        if self.image is None or self.image.size == 0:
            print("Could not read the image")
            return None

        # 调整图像大小
        resized_image = cv2.resize(self.image, self.picture_size)
        
        # 创建批次
        images = []
        for i in range(self.default_batch):
            images.append(resized_image)

        # 转换为blob
        input_blob = cv2.dnn.blobFromImages(
            images,
            1.0,
            self.picture_size,
            mean=(0, 0, 0),
            swapRB=False,
            crop=False
        )

        return input_blob

    def load_model(self):
        self.net = cv2.dnn.readNetFromONNX(self.model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


class DigitCanvas:
    def __init__(self, root, classifier):
        self.root = root
        self.classifier = classifier
        
        # 创建画布
        self.canvas_width = 280  # 设置大一点便于书写
        self.canvas_height = 280
        self.canvas = tk.Canvas(root, width=self.canvas_width, 
                              height=self.canvas_height, bg='black')
        self.canvas.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 创建控制按钮
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = tk.Button(self.control_frame, text="clear", command=self.clear_canvas)
        self.clear_button.pack(pady=5)
        
        self.recognize_button = tk.Button(self.control_frame, text="recognize", command=self.recognize_digit)
        self.recognize_button.pack(pady=5)
        
        self.result_label = tk.Label(self.control_frame, text="result:", font=('Arial', 18))
        self.result_label.pack(pady=5)
        
        # 绘图相关变量  
        self.last_x = None
        self.last_y = None
        self.line_width = 10
        
        # 绑定鼠标事件
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_draw)
        
        # 创建图像缓存
        self.image = Image.new('L', (self.canvas_width, self.canvas_height), 'black')
        self.draw = ImageDraw.Draw(self.image)

    def start_draw(self, event):
        self.last_x = event.x
        self.last_y = event.y
        
    def draw(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                  width=self.line_width, fill='white',
                                  capstyle=tk.ROUND, smooth=True)
            # 同时在PIL图像上绘制
            self.draw.line([self.last_x, self.last_y, event.x, event.y],
                         fill='white', width=self.line_width)
        self.last_x = event.x
        self.last_y = event.y
        
    def stop_draw(self, event):
        self.last_x = None
        self.last_y = None
        
    def clear_canvas(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (self.canvas_width, self.canvas_height), 'black')
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="result:")
        
    def recognize_digit(self):
        # 将PIL图像转换为OpenCV格式
        img_array = np.array(self.image)
        # 调整大小为模型所需的28x28
        resized = cv2.resize(img_array, (28, 28), interpolation=cv2.INTER_AREA)
        # 识别数字
        result = self.classifier.classify_image(resized)
        self.result_label.config(text=f"result: {result}")

# 主程序
if __name__ == "__main__":
    root = tk.Tk()
    root.title("number_recognition")
    
    # 创建分类器实例
    model = ImageClassifier("/home/ccy/workspace/deeplearning/model/"
                           "googlenet_100bitch_30_times9909.onnx")
    
    # 创建画布应用
    app = DigitCanvas(root, model)
    
    root.mainloop()

