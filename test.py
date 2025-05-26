import torch
x = torch.randn(8, 3, 5, 4)
y = x.transpose(1,2)  # 交换第二与第三维度
print(y.shape)
