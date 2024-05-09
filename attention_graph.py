import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from src import UNet

# 加载模型

# get devices
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

# create model
model = UNet(in_channels=3, num_classes=2, base_c=64)

checkpoint = torch.load('./save_weights/best_model.pth')
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# model.load_state_dict(torch.load('./save_weights/best_model.pth'))
model.eval()

# 加载图像
image = plt.imread('./datasets/test/images/01_test.tif')
image = transforms.ToTensor()(image)
image = image.unsqueeze(0).to(device)

# 获取特征图
with torch.no_grad():
    feature_maps = model.get_feature_maps(image)

# 计算注意力权重
gap_layer = torch.nn.AdaptiveAvgPool2d(output_size=1)(feature_maps)
dense_layer = torch.nn.Linear(in_features=feature_maps.shape[1], out_features=feature_maps.shape[1])(gap_layer)
dense_layer = torch.nn.ReLU(inplace=True)(dense_layer)
attention_layer = torch.nn.Linear(in_features=feature_maps.shape[1], out_features=feature_maps.shape[1])(dense_layer)
attention_layer = torch.nn.Softmax(dim=1)(attention_layer)

# 生成热力图
attention_maps = feature_maps * attention_layer
attention_maps = torch.nn.functional.adaptive_avg_pool2d(attention_maps, output_size=(image.shape[-2], image.shape[-1]))
attention_maps = attention_maps.squeeze(0).squeeze(0).cpu().numpy()

# 可视化热力图
plt.imshow(attention_maps, cmap='jet')
plt.show()


