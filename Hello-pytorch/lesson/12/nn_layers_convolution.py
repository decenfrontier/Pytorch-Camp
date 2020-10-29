import os
import sys
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

# ======= 2d
# flag = 1
flag = 0
if flag:
    conv_layer = nn.Conv2d(3, 1, 3)
    nn.init.xavier_normal_(conv_layer.weight.data)
    img_conv = conv_layer(img_tensor)

# ======= 转置
flag = 1
# flag = 0
if flag:
    conv_layer = nn.ConvTranspose2d(3, 1, 3, stride=2)
    nn.init.xavier_normal_(conv_layer.weight.data)
    img_conv = conv_layer(img_tensor)

# ======================== 可视化 ============================
print("卷积前尺寸:{}\n卷积后尺寸:{}".format(img_tensor.shape, img_conv.shape))
img_conv = transform_invert(img_conv[0, 0:1, ...], img_transform)
img_raw = transform_invert(img_tensor.squeeze(), img_transform)
plt.subplot(122).imshow(img_conv, cmap="gray")
plt.subplot(121).imshow(img_raw)
plt.show()