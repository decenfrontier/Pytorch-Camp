import time
import torch
import torchvision.transforms as transforms
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
import os

hello_pytorch_dir = os.path.join(os.path.dirname(__file__), "..", "..")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


if __name__ == "__main__":
    path_img = os.path.join(hello_pytorch_dir, "data", "det_demo_img1.png")
    # path_img = os.path.join(hello_pytorch_dir, "data", "det_demo_img2.png")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 1. load data & model
    input_image = Image.open(path_img).convert("RGB")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # 2. preprocess
    img_chw = preprocess(input_image)

    # 3. to device
    if torch.cuda.is_available():
        img_chw = img_chw.to('cuda')
        model.to('cuda')

    # 4. forward
    input_list = [img_chw]
    with torch.no_grad():
        tic = time.time()
        print("input img tensor shape:{}".format(input_list[0].shape))
        output_list = model(input_list)
        output_dict = output_list[0]
        print("pass: {:.3f}s".format(time.time() - tic))
        for k, v in output_dict.items():
            print("key:{}, value:{}".format(k, v))

    # 5. visualization
    out_boxes = output_dict["boxes"].cpu()
    out_scores = output_dict["scores"].cpu()
    out_labels = output_dict["labels"].cpu()

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(input_image, aspect='equal')

    num_boxes = out_boxes.shape[0]
    max_vis = 40
    thres = 0.5

    for idx in range(0, min(num_boxes, max_vis)):

        score = out_scores[idx].numpy()
        bbox = out_boxes[idx].numpy()
        class_name = COCO_INSTANCE_CATEGORY_NAMES[out_labels[idx]]

        if score < thres:
            continue

        ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False,
                                   edgecolor='red', linewidth=3.5))
        ax.text(bbox[0], bbox[1] - 2, '{:s} {:.3f}'.format(class_name, score), bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
    plt.show()
    plt.close()

