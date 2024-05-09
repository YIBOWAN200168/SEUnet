import torch

model = UNet()  # 替换为您自己的Unet模型
image = torch.randn(1, 3, 256, 256)  # 替换为您自己的输入图像

# 获取最后一层卷积层的输出
features = None
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        features = module(image)
assert features is not None, 'Failed to find the last convolutional layer'
