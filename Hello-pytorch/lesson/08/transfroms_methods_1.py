import os, sys
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

hello_pytorch_dir = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.append(hello_pytorch_dir)

path_lenet = os.path.join(hello_pytorch_dir, "model", "lenet.py")
path_tools = os.path.join(hello_pytorch_dir, "tools", "common_tools.py")


from tools.my_dataset import RMBDataset
from tools.common_tools import set_seed, transform_invert

set_seed(1)  # 设置随机种子

# 参数设置
MAX_EPOCH = 10
BATCH_SIZE = 1
LR = 0.01
log_interval = 10
val_interval = 1
rmb_label = {"1": 0, "100": 1}

# ============================ step 1/5 数据 ============================
split_dir = os.path.join(hello_pytorch_dir, "data", "rmb_split")
train_dir = os.path.join(split_dir, "train")
valid_dir = os.path.join(split_dir, "valid")

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # 1 CenterCrop
    # transforms.CenterCrop(128),     # 512

    # 2 RandomCrop
    # transforms.RandomCrop(224, padding=16),
    # transforms.RandomCrop(224, padding=(16, 64)),
    # transforms.RandomCrop(224, padding=16, fill=(255, 0, 0)),
    # transforms.RandomCrop(512, pad_if_needed=True),   # pad_if_needed=True
    # transforms.RandomCrop(224, padding=64, padding_mode='edge'),
    # transforms.RandomCrop(512, padding=512, padding_mode='reflect'),
    # transforms.RandomCrop(512, padding=512, padding_mode='symmetric'),

    # 3 RandomResizedCrop
    # transforms.RandomResizedCrop(size=224, scale=(0.5, 0.5)),

    # 4 FiveCrop
    # transforms.FiveCrop(112),
    # transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops])),

    # 5 TenCrop
    # transforms.TenCrop(112, vertical_flip=False),
    # transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops])),

    # 1 Horizontal Flip
    # transforms.RandomHorizontalFlip(p=1),

    # 2 Vertical Flip
    # transforms.RandomVerticalFlip(p=0.5),

    # 3 RandomRotation
    # transforms.RandomRotation(90),
    # transforms.RandomRotation((90), expand=True),
    # transforms.RandomRotation(30, center=(0, 0)),
    # transforms.RandomRotation(30, center=(0, 0), expand=True),   # expand only for center rotation

    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

# 构建MyDataset实例
train_data = RMBDataset(data_dir=train_dir, transform=train_transform)
valid_data = RMBDataset(data_dir=valid_dir, transform=valid_transform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

# ============================ step 5/5 训练 ============================
for epoch in range(MAX_EPOCH):
    for i, data in enumerate(train_loader):

        inputs, labels = data   # B C H W

        img_tensor = inputs[0, ...]     # C H W
        img = transform_invert(img_tensor, train_transform)
        plt.imshow(img)
        plt.show()
        plt.pause(0.5)
        plt.close()

        # bs, ncrops, c, h, w = inputs.shape
        # for n in range(ncrops):
        #     img_tensor = inputs[0, n, ...]  # C H W
        #     img = transform_invert(img_tensor, train_transform)
        #     plt.imshow(img)
        #     plt.show()
        #     plt.pause(1)
