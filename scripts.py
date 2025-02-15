import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# tensor = torch.load("attn_map.pth").to(torch.float32)

# # 将tensor转换为numpy数组
# data = tensor.numpy()

# # 创建热力图
# plt.figure(figsize=(10, 10))  # 可以调整热力图大小
# sns.heatmap(data, cmap='YlGnBu', cbar=True)

# # 保存为jpg文件
# plt.savefig('heatmap.jpg', format='jpg', dpi=300, bbox_inches='tight')


import torch
import torch.nn as nn


class Conv1DModule(nn.Module):
    def __init__(self, input_channels, mid_channels, output_channels=None):
        super(Conv1DModule, self).__init__()
        output_channels = output_channels if output_channels else input_channels

        self.conv1 = nn.Conv1d(input_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(mid_channels, output_channels, kernel_size=3, padding=1, bias=False)

        self.init_param()
    
    def init_param(self):
        for param in self.conv1.parameters():
            nn.init.zeros_(param)
        for param in self.conv2.parameters():
            nn.init.normal_(param)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x

# 示例输入 (bhw, c, t)
bhw, c, t = 1350, 3072, 13  # 假设batch=32, 通道数=10, 时间步长=100
input_data = torch.randn(bhw, c, t)

# 创建模型
model = Conv1DModel(input_channels=c, mid_channels=128, output_channels=c)

# 前向传播
output = model(input_data)
print("输出形状:", output.sum())
