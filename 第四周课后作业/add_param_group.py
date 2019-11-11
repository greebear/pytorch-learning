import torch
import torch.optim as optim
from tools.common_tools import set_seed
set_seed(1)  # 设置随机种子

w1 = torch.randn((2, 2), requires_grad=True)
w2 = torch.randn((2, 2), requires_grad=True)
w3 = torch.randn((2, 2), requires_grad=True)

optimizer = optim.SGD([w1], lr=0.01, momentum=0.9)
optimizer.add_param_group({"params": w2, 'lr': 0.02, 'momentum': 0.8})
optimizer.add_param_group({"params": w3, 'lr': 0.03, 'momentum': 0.7})

for index, group in enumerate(optimizer.param_groups):
    params = group["params"]
    lr = group["lr"]
    momentum = group["momentum"]
    print("第【{}】组参数 params 为:\n{} \n学习率 lr 为:{} \n动量 momentum 为:{}".format(index, params, lr, momentum))
    print("==============================================")
