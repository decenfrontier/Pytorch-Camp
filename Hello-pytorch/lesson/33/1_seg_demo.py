import os
import time
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt

hello_pytorch_dir = os.path.join(os.path.dirname(__file__), "..", "..")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # path_img = os.path.join(hello_pytorch_dir, "data", "demo_img1.png")
    # path_img = os.path.join(hello_pytorch_dir, "data", "demo_img2.png")
    path_img = os.path.join(hello_pytorch_dir, "data", "demo_img3.png")

    # config
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 1. load data & model
    input_image = Image.open(path_img).convert("RGB")
    print("\n 注意，由于网络问题，cache的获取可能要较长时间，可自行下载并解压，放到↓↓↓的路径中")
    model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet101', pretrained=True)
    model.eval()

    # 2. preprocess
    input_tensor = preprocess(input_image)
    input_bchw = input_tensor.unsqueeze(0)

    # 3. to device
    if torch.cuda.is_available():
        input_bchw = input_bchw.to(device)
        model.to(device)

    # 4. forward
    with torch.no_grad():
        tic = time.time()
        print("input img tensor shape:{}".format(input_bchw.shape))
        output_4d = model(input_bchw)['out']
        output = output_4d[0]
        print("pass: {:.3f}s use: {}".format(time.time() - tic, device))
        print("output img tensor shape:{}".format(output.shape))
    output_predictions = output.argmax(0)

    # 5. visualization
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    r.putpalette(colors)
    plt.subplot(121).imshow(r)
    plt.subplot(122).imshow(input_image)
    plt.show()

    # appendix
    classes = ['__background__',
                       'aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair',
                       'cow', 'diningtable', 'dog', 'horse',
                       'motorbike', 'person', 'pottedplant',
                       'sheep', 'sofa', 'train', 'tvmonitor']