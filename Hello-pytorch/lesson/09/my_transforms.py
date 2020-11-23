import os, sys
import numpy as np
import random
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader
from tools.my_dataset import RMBDataset
from tools.common_tools import set_seed, transform_invert

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
hello_pytorch_dir = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.append(hello_pytorch_dir)

path_lenet = os.path.join(hello_pytorch_dir, "model", "lenet.py")
path_tools = os.path.join(hello_pytorch_dir, "tools", "common_tools.py")

set_seed(1)  # 设置随机种子

# 参数设置
MAX_EPOCH = 10
BATCH_SIZE = 1
LR = 0.01
log_interval = 10
val_interval = 1
rmb_label = {"1": 0, "100": 1}


class AddPepperNoise(object):
    """增加椒盐噪声
    Args:
        snr （float）: Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) and (isinstance(p, float))    # 2020 07 26 or --> and
        self.snr = snr
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255   # 盐噪声
            img_[mask == 2] = 0     # 椒噪声
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img


# ============================ step 1/5 数据 ============================
split_dir = os.path.join(hello_pytorch_dir, "data", "rmb_split")
train_dir = os.path.join(split_dir, "train")
valid_dir = os.path.join(split_dir, "valid")

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    AddPepperNoise(0.9, p=0.5),
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


