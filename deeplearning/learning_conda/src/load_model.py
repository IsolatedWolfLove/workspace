import onnxruntime as ort
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
session = ort.InferenceSession('cnn.onnx')
input_name = session.get_inputs()[0].name
image = Image.open('/home/ccy/workspace/deeplearning/img9.png').convert('L')  # 确保是灰度图
image = image.resize((28, 28))  # 调整大小为 28x28
image_array = np.array(image)  # 将 PIL Image 转换为 NumPy 数组
image_array = 255 - image_array
input_tensor = np.repeat(image_array.reshape(1, -1).astype(np.float32), 100, axis=0)
outputs = session.run(None, {input_name: input_tensor})
predicted_class = np.argmax(outputs[0])
print(f"Predicted class: {predicted_class}")

