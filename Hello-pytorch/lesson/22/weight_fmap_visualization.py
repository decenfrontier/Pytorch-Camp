import sys
import os
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.models as models
from tools.common_tools import set_seed
from PIL import Image

set_seed(1)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
hello_pytorch_dir = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.append(hello_pytorch_dir)

# -------------------------- kernel visualization -----------------------
flag = 0
# flag = 1
if flag:
    writer = SummaryWriter(comment='test_comment', filename_suffix="test_suffix")
    alexnet = models.alexnet(pretrained=True)
    # 当前遍历到第几层网络的卷积核了
    kernel_num = -1
    # 最多显示两层网络的卷积核:第 0 层和第 1 层
    vis_max = 1
    # 获取网络的每一层
    for sub_module in alexnet.modules():
        # 判断这一层是否为 2 维卷积层
        if isinstance(sub_module, nn.Conv2d):
            kernel_num += 1
            # 如果当前层大于1，则停止记录权值
            if kernel_num > vis_max:
                break
            # 获取这一层的权值
            kernels = sub_module.weight
            # 权值的形状是 [c_out, c_int, k_w, k_h]
            c_out, c_int, k_w, k_h = tuple(kernels.shape)
            # 根据输出的每个维度进行可视化
            for o_idx in range(c_out):
                # 取出的数据形状是 (c_int, k_w, k_h)，对应 BHW; 需要扩展为 (c_int, 1, k_w, k_h)，对应 BCHW
                kernel_idx = kernels[o_idx, :, :, :].unsqueeze(1)   # make_grid需要 BCHW，这里拓展C维度
                # 注意 nrow 设置为 c_int，所以行数为 1。在 for 循环中每 添加一个，就会多一个 global_step
                kernel_grid = vutils.make_grid(kernel_idx, normalize=True, scale_each=True, nrow=c_int)
                writer.add_image('{}_Convlayer_split_in_channel'.format(kernel_num), kernel_grid, global_step=o_idx)
            # 因为 channel 为 3 时才能进行可视化，所以这里 reshape
            kernel_all = kernels.view(-1, 3, k_h, k_w)  # 3, h, w
            kernel_grid = vutils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=8)  # c, h, w
            writer.add_image('{}_all'.format(kernel_num), kernel_grid, global_step=322)
            print("{}_convlayer shape:{}".format(kernel_num, tuple(kernels.shape)))
    writer.close()

# ------------------------ feature map visualization -------------------------
# flag = 0
flag = 1
if flag:
    writer = SummaryWriter(comment='test_comment', filename_suffix="test_suffix")
    # 数据
    path_img = "./lena.png"  # your path to image
    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]

    norm_transform = transforms.Normalize(normMean, normStd)
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        norm_transform
    ])

    img_pil = Image.open(path_img).convert('RGB')
    if img_transforms is not None:
        img_tensor = img_transforms(img_pil)
    img_tensor.unsqueeze_(0)  # chw --> bchw

    # 模型
    alexnet = models.alexnet(pretrained=True)

    # forward
    convlayer1 = alexnet.features[0]
    fmap_1 = convlayer1(img_tensor)

    # 预处理
    fmap_1.transpose_(0, 1)  # bchw=(1, 64, 55, 55) --> (64, 1, 55, 55)
    fmap_1_grid = vutils.make_grid(fmap_1, normalize=True, scale_each=True, nrow=8)

    writer.add_image('feature map in conv1', fmap_1_grid, global_step=322)
    writer.close()