import os
import sys
import torch
import torch.nn as nn
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

hello_pytorch_dir = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.append(hello_pytorch_dir)

path_tools = os.path.join(hello_pytorch_dir, "tools", "common_tools.py")
from tools.common_tools import set_seed

set_seed(1)

class MLP(nn.Module):
    def __init__(self, neural_num, layers):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(neural_num, neural_num, bias=False) for i in range(layers)])
        self.neural_num = neural_num

    def forward(self, x):
        for i, linear in enumerate(self.linears):
            x = linear(x)
            x = torch.relu(x)
            print(f"layer:{i}, mean:{x.mean()}, std:{x.std()}")
            if torch.isnan(x.std()):
                print(f"output is nan in layer{i}")
                break
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight.data, std=np.sqrt(1/self.neural_num))

                # tanh_gain = nn.init.calculate_gain("tanh")  # TODO: 为什么要计算增益?
                # nn.init.xavier_uniform_(m.weight.data, gain=tanh_gain)

                # nn.init.normal_(m.weight.data, std=np.sqrt(2/self.neural_num))
                nn.init.kaiming_normal_(m.weight.data)

layer_nums = 100
neural_nums = 256
batch_size = 16

net = MLP(neural_nums, layer_nums)
net.initialize()
inputs = torch.randn((batch_size, neural_nums))
output = net(inputs)
print(output)
