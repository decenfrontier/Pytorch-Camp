import sys
import os
import time
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils

from tools.my_dataset import RMBDataset
from tools.common_tools import set_seed
from model.lenet import LeNet
from torchsummary import summary

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
hello_pytorch_dir = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.append(hello_pytorch_dir)



set_seed(1)  # 设置随机种子

# --------------------------- 3 image --------------------------
flag = 0
# flag = 1
if flag:
    writer = SummaryWriter(comment="test_comment", filename_suffix="test_suffix")

    # img 1     random
    fake_img = torch.randn(3, 512, 512)
    writer.add_image("fake_img", fake_img, 1)
    time.sleep(1)

    # img 2     ones
    fake_img = torch.ones(3, 512, 512)
    time.sleep(1)
    writer.add_image("fake_img", fake_img, 2)

    # img 3     1.1
    fake_img = torch.ones(3, 512, 512) * 1.1
    time.sleep(1)
    writer.add_image("fake_img", fake_img, 3)

    # img 4     HW
    fake_img = torch.rand(512, 512)
    writer.add_image("fake_img", fake_img, 4, dataformats="HW")

    # img 5     HWC
    fake_img = torch.rand(512, 512, 3)
    writer.add_image("fake_img", fake_img, 5, dataformats="HWC")

    writer.close()

# ----------------------------------- 4 make_grid -----------------------------------
flag = 0
# flag = 1
if flag:
    writer = SummaryWriter(comment='test_comment', filename_suffix="test_suffix")

    split_dir = os.path.join("..", "..", "data", "rmb_split")
    train_dir = os.path.join(split_dir, "train")

    transform_compose = transforms.Compose([transforms.Resize((32, 64)), transforms.ToTensor()])
    train_data = RMBDataset(data_dir=train_dir, transform=transform_compose)
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
    data_batch, label_batch = next(iter(train_loader))

    img_grid = vutils.make_grid(data_batch, nrow=4, normalize=True, scale_each=True)
    writer.add_image("input img", img_grid, 0)

    writer.close()


# ----------------------------------- 5 add_graph -----------------------------------
# flag = 0
flag = 1
if flag:
    writer = SummaryWriter(comment='test_comment', filename_suffix="test_suffix")

    # 模型
    fake_img = torch.randn(1, 3, 32, 32)
    lenet = LeNet(classes=2)
    writer.add_graph(lenet, fake_img)
    writer.close()

    print(summary(lenet, (3, 32, 32), device="cpu"))