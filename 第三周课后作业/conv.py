import torch.nn as nn
import torch

# conv_layer_1 = nn.Conv2d(3, 1, 3, bias=False)
conv_layer_1 = nn.Conv2d(3, 1, 3, bias=False, padding=1)
conv_layer_1.weight.data = torch.ones(conv_layer_1.weight.shape)

# conv_layer_2 = nn.Conv2d(3, 1, 3, bias=False)
conv_layer_2 = nn.Conv2d(3, 1, 3, bias=False, padding=1)
conv_layer_2.weight.data = torch.ones(conv_layer_2.weight.shape)*2

img_tensor = torch.ones((1, 3, 5, 5))
img_tensor[:, 1, :, :] = img_tensor[:, 1, :, :]*2
img_tensor[:, 2, :, :] = img_tensor[:, 2, :, :]*3


img_conv_1 = conv_layer_1(img_tensor)
print("卷积前尺寸:{}\n卷积后尺寸:{}".format(img_tensor.shape, img_conv_1.shape))
print("像素值大小:1x1x9+2x1x9+3x1x9 = {}".format(img_conv_1[0, 0, 3, 3].data))

print("=======================================")
img_conv_2 = conv_layer_2(img_tensor)
print("卷积前尺寸:{}\n卷积后尺寸:{}".format(img_tensor.shape, img_conv_2.shape))
print("像素值大小:1x2x9+2x2x9+3x2x9 = {}".format(img_conv_2[0, 0, 3, 3].data))
