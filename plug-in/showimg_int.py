import sys
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

# 获取终端传入的参数
filename = sys.argv[1]

# 读取RGB格式的文本文件
with open(filename, 'r') as f:
    data = f.readlines()

# 将文本数据转换为NumPy数组
data = [line.strip().split() for line in data]
data = np.array(data, dtype=np.float32)
data = data.reshape(3, 224, 224)


zero_point = 114
scale = 0.018658448

data = (data - zero_point) * scale

# 创建反转换器
transform = transforms.Compose([
    transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.ToPILImage()
])

img = transform(torch.tensor(data).float())
# 转换为PIL图像对象
#img = Image.fromarray(np.uint8(data.transpose((1, 2, 0)) * 255))

# 展示图片
img.show()
