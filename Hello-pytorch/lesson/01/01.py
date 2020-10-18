import torch

print("Pytorch version:{}".format(torch.__version__))

print("CUDA is available:{}, version is:{}".format(torch.cuda.is_available(),
                                                   torch.version.cuda))

#print("device_name:{}".format(torch.cuda.get_device_name(0)))