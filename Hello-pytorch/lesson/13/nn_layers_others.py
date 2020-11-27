import os
import sys
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

hello_pytorch_dir = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.append(hello_pytorch_dir)

path_tools = os.path.join(hello_pytorch_dir, "tools", "common_tools.py")
from tools.common_tools import transform_invert, set_seed

set_seed(1)

# ======================= 加载图片 =============================
path_img = "./lena.png"
img = Image.open(path_img).convert("RGB")
# 转为tensor
img_transform = transforms.Compose([transforms.ToTensor()])
img_tensor = img_transform(img)
img_tensor.unsqueeze_(dim=0)  # C*H*W to B*C*H*W

# ======================= 创建卷积层 =========================
# ===== maxpool
# flag = 1
flag = 0
if flag:
    maxpool_layer = nn.MaxPool2d((2,2), stride=(2,2))
    img_pool = maxpool_layer(img_tensor)

# ===== avgpool
# flag = 1
flag = 0
if flag:
    avgpool_layer = nn.AvgPool2d((2,2), stride=(2,2))
    img_pool = avgpool_layer(img_tensor)

# ===== avgpool divisor_override
# flag = 1
flag = 0
if flag:
    img_tensor = torch.ones((1,1,4,4))
    avgpool_layer = nn.AvgPool2d((2,2), stride=(2,2), divisor_override=3)
    img_pool = avgpool_layer(img_tensor)
    print(f"raw_img:{img_tensor}")
    print(f"pooling_img:{img_pool}")

# ===== max unpool
flag = 1
# flag = 0
if flag:
    # pooling
    img_tensor = torch.randint(high=5, size=(1,1,4,4), dtype=torch.float)
    maxpool_layer = nn.MaxPool2d((2,2), stride=(2,2), return_indices=True)
    img_pool, indices = maxpool_layer(img_tensor)

    # unpooling
    img_reconstruct = torch.randn_like(img_pool, dtype=torch.float)
    maxunpool_layer = nn.MaxUnpool2d((2, 2), stride=(2,2))
    img_unpool = maxunpool_layer(img_reconstruct, indices)

    print(f"img_raw:\n{img_tensor}")
    print(f"img_pool:\n{img_pool}")
    print(f"img_reconstruct:\n{img_reconstruct}")
    print(f"img_unpool:\n{img_unpool}")

# ===== linear
# flag = 1
flag = 0
if flag:
    inputs = torch.tensor([[1,2,3]], dtype=torch.float)
    linear_layer = nn.Linear(3,4)
    linear_layer.weight.data = torch.tensor([[1, 1, 1],
                                             [2, 2, 2],
                                             [3, 3, 3],
                                             [4, 4, 4]], dtype=torch.float)
    linear_layer.bias.data.fill_(0.5)
    output = linear_layer(inputs)
    print(inputs, inputs.shape)
    print(linear_layer.weight.data, linear_layer.weight.data.shape)
    print(output, output.shape)


# ======================== 可视化 =============================
print(f"池化前尺寸:{img_tensor.shape}\n池化后尺寸:{img_pool.shape}")
img_pool = transform_invert(img_pool[0, 0:3, ...], img_transform)
img_raw = transform_invert(img_tensor.squeeze(), img_transform)
plt.subplot(121).imshow(img_raw)
plt.subplot(122).imshow(img_pool)
plt.show()
