import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tools.common_tools import transform_invert, set_seed
from matplotlib import pyplot as plt
set_seed(2)

# ================================= load img ==================================
path_img = "lena.png"
img = Image.open(path_img).convert('RGB')  # 0~255

# convert to tensor
img_transform = transforms.Compose([transforms.ToTensor()])
img_tensor = img_transform(img)
img_tensor.unsqueeze_(dim=0)    # C*H*W to B*C*H*W

# ================ 3d kernel (1, 3, 3)
# flag = 1
flag = 0
if flag:
    conv_layer = nn.Conv3d(3, 1, (1, 3, 3), padding=(1, 0, 0), bias=False)
    nn.init.xavier_normal_(conv_layer.weight.data)

    # calculation
    img_tensor.unsqueeze_(dim=2)    # B*C*H*W to B*C*D*H*W
    img_conv = conv_layer(img_tensor)

# ================================= visualization ==================================
    print("卷积前尺寸:{}\n卷积后尺寸:{}".format(img_tensor.shape, img_conv.shape))
    img_conv = transform_invert(img_conv.squeeze(), img_transform)
    img_raw = transform_invert(img_tensor.squeeze(), img_transform)
    plt.subplot(122).imshow(img_conv, cmap='gray')
    plt.subplot(121).imshow(img_raw)
    plt.show()

# ================ 3d kernel (3, 3, 3)
flag = 1
# flag = 0
if flag:
    conv_layer = nn.Conv3d(3, 1, (3, 3, 3), padding=(1, 0, 0), bias=False)
    nn.init.xavier_normal_(conv_layer.weight.data)

    # calculation
    img_tensor.unsqueeze_(dim=2)    # B*C*H*W to B*C*D*H*W
    img_conv = conv_layer(img_tensor)

# ================================= visualization ==================================
    print("卷积前尺寸:{}\n卷积后尺寸:{}".format(img_tensor.shape, img_conv.shape))
    img_conv = transform_invert(img_conv[:, :, ...], img_transform)
    img_raw = transform_invert(img_tensor.squeeze(), img_transform)
    plt.subplot(122).imshow(img_conv, cmap='gray')
    plt.subplot(121).imshow(img_raw)
    plt.show()
