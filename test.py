import torch
import torch.nn as nn

x = torch.randn(3,1,5)
print(x)
print(x.shape)
x.unsqueeze_(2)
print(x)
print(x.shape)